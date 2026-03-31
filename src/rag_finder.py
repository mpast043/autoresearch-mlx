"""
RAG Finder — semantic search over the findings corpus.

Provides a lightweight hybrid search (keyword + embedding) over all
validated findings and build briefs so that agents can retrieve
relevant prior work without re-running the pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.runtime.paths import resolve_project_path

logger = logging.getLogger(__name__)

# ── Embedding model (lightweight, CPU-friendly) ────────────────────────────────

EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM: int = 384  # all-MiniLM-L6-v2 output dimension


def _get_embedding_model():
    """Lazily import and cache the sentence-transformers model."""
    if not hasattr(_get_embedding_model, "_model"):
        try:
            from sentence_transformers import SentenceTransformer
            _get_embedding_model._model = SentenceTransformer(EMBED_MODEL)
            logger.info("[RAG] loaded embedding model=%s dim=%d", EMBED_MODEL, EMBED_DIM)
        except ImportError:
            logger.warning("[RAG] sentence-transformers not installed — using TF-IDF fallback")
            _get_embedding_model._model = None
    return _get_embedding_model._model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using the configured embedding model.
    Falls back to TF-IDF vectors if sentence-transformers is unavailable.
    Returns list of normalized float vectors.
    """
    model = _get_embedding_model()
    if model is not None:
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    # Fallback: TF-IDF
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=EMBED_DIM, stop_words="english")
        matrices = vec.fit_transform(texts).toarray()
        # Normalize rows
        norms = np.linalg.norm(matrices, axis=1, keepdims=True)
        norms[norms == 0] = 1
        matrices = matrices / norms
        return [row.tolist() for row in matrices]
    except Exception as exc:
        logger.error("[RAG] TF-IDF fallback also failed: %s", exc)
        # Return zero vectors as last resort
        return [([0.0] * EMBED_DIM) for _ in texts]


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return max(0.0, min(1.0, dot))  # already normalized


# ── SQLite corpus store ────────────────────────────────────────────────────────

DEFAULT_DB_PATH = Path("data/rag_corpus.db")


def _db_path() -> Path:
    configured = os.getenv("RAG_CORPUS_DB_PATH")
    return resolve_project_path(configured, default=DEFAULT_DB_PATH)


def _get_connection() -> sqlite3.Connection:
    db_path = _db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA mmap_size=268435456")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS corpus (
            chunk_id   TEXT PRIMARY KEY,
            chunk_text TEXT NOT NULL,
            chunk_vec  BLOB NOT NULL,
            source_type TEXT NOT NULL,   -- 'finding' | 'build_brief' | 'validation_note'
            source_id  INTEGER NOT NULL,
            metadata   TEXT,              -- JSON
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_source ON corpus(source_type, source_id);
    """)
    conn.commit()


# ── Chunking ──────────────────────────────────────────────────────────────────

CHUNK_SIZE = 500  # characters per chunk


def _chunk(text: str, chunk_id_prefix: str, max_len: int = CHUNK_SIZE) -> list[tuple[str, str]]:
    """Split text into overlapping chunks. Returns [(chunk_id, chunk_text)]."""
    chunks = []
    sentences = text.replace("\n", ". ").split(". ")
    buf = ""
    for sent in sentences:
        if len(buf) + len(sent) > max_len and buf:
            chunk_id = hashlib.sha256(buf.encode()).hexdigest()[:16]
            chunks.append((f"{chunk_id_prefix}_{chunk_id}", buf.strip()))
            buf = sent
        else:
            buf = (buf + " " + sent).strip()
    if buf:
        chunk_id = hashlib.sha256(buf.encode()).hexdigest()[:16]
        chunks.append((f"{chunk_id_prefix}_{chunk_id}", buf.strip()))
    return chunks


# ── Indexing ─────────────────────────────────────────────────────────────────

def index_findings(db: "Database") -> int:
    """Index all validated/parked findings into the RAG corpus. Returns count."""
    from datetime import datetime, timezone
    import json

    conn = _get_connection()
    indexed = 0
    for finding in db.get_findings(limit=10000):
        prefix = f"finding_{finding.id}"
        text = f"{finding.product_built or ''} {finding.outcome_summary or ''}".strip()
        if not text:
            continue
        chunk_texts = [ct for _, ct in _chunk(text, prefix)]
        if not chunk_texts:
            continue
        vectors = embed_texts(chunk_texts)
        for (chunk_id, chunk_text), vec in zip(chunk_texts, vectors):
            conn.execute(
                """
                INSERT OR REPLACE INTO corpus
                    (chunk_id, chunk_text, chunk_vec, source_type, source_id, metadata, created_at)
                VALUES (?, ?, ?, 'finding', ?, ?, ?)
                """,
                (
                    chunk_id,
                    chunk_text,
                    np.array(vec, dtype=np.float32).tobytes(),
                    int(finding.id),
                    json.dumps({"product_built": finding.product_built, "status": finding.status}),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        indexed += 1
    conn.commit()
    conn.close()
    logger.info("[RAG] indexed %d findings", indexed)
    return indexed


def index_build_briefs(db: "Database") -> int:
    """Index all build briefs."""
    from datetime import datetime, timezone
    import json

    conn = _get_connection()
    indexed = 0
    for brief in db.get_build_briefs(limit=1000) if hasattr(db, "get_build_briefs") else []:
        prefix = f"brief_{brief.id}"
        text = f"{getattr(brief, 'title', '') or ''} {getattr(brief, 'description', '') or ''}".strip()
        if not text:
            continue
        chunk_texts = [ct for _, ct in _chunk(text, prefix)]
        vectors = embed_texts(chunk_texts)
        for (chunk_id, chunk_text), vec in zip(chunk_texts, vectors):
            conn.execute(
                """
                INSERT OR REPLACE INTO corpus
                    (chunk_id, chunk_text, chunk_vec, source_type, source_id, metadata, created_at)
                VALUES (?, ?, ?, 'build_brief', ?, ?, ?)
                """,
                (
                    chunk_id,
                    chunk_text,
                    np.array(vec, dtype=np.float32).tobytes(),
                    int(brief.id),
                    json.dumps({"title": getattr(brief, "title", "")}),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        indexed += 1
    conn.commit()
    conn.close()
    logger.info("[RAG] indexed %d build briefs", indexed)
    return indexed


def rebuild_corpus(db: "Database") -> dict[str, int]:
    """Full rebuild of the RAG corpus from DB. Returns counts."""
    counts = {"findings": index_findings(db), "build_briefs": index_build_briefs(db)}
    logger.info("[RAG] corpus rebuild complete: %s", counts)
    return counts


# ── Search ───────────────────────────────────────────────────────────────────

@dataclass
class SearchHit:
    chunk_id: str
    chunk_text: str
    source_type: str
    source_id: int
    score: float
    metadata: dict


def hybrid_search(
    query: str,
    top_k: int = 5,
    source_type: Optional[str] = None,
    min_score: float = 0.3,
) -> list[SearchHit]:
    """
    Hybrid (keyword + semantic) search over the corpus.

    1. Embed query → ANN search against all vectors
    2. Keyword filter on chunk_text
    3. Return top-k with score >= min_score
    """
    conn = _get_connection()
    try:
        query_vec = embed_texts([query])[0]
        query_arr = np.frombuffer(
            np.array(query_vec, dtype=np.float32).tobytes(), dtype=np.float32
        )

        if source_type:
            rows = conn.execute(
                "SELECT chunk_id, chunk_text, chunk_vec, source_type, source_id, metadata FROM corpus WHERE source_type = ?",
                (source_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT chunk_id, chunk_text, chunk_vec, source_type, source_id, metadata FROM corpus"
            ).fetchall()

        scored: list[tuple[float, dict]] = []
        for chunk_id, chunk_text, chunk_vec, source_type, source_id, metadata in rows:
            vec = np.frombuffer(chunk_vec, dtype=np.float32)
            score = cosine_sim(query_vec, vec.tolist())
            # Light keyword boost
            if query.lower() in chunk_text.lower():
                score = min(1.0, score + 0.05)
            if score >= min_score:
                import json
                scored.append((score, {
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text,
                    "source_type": source_type,
                    "source_id": source_id,
                    "score": round(score, 4),
                    "metadata": json.loads(metadata or "{}"),
                }))

        scored.sort(key=lambda x: -x[0])
        return [SearchHit(**item) for _, item in scored[:top_k]]
    finally:
        conn.close()


def format_hits_for_context(hits: list[SearchHit]) -> str:
    """Format search hits as a readable context string for LLM consumption."""
    if not hits:
        return "No relevant prior findings found."
    lines = ["=== Relevant Prior Findings ==="]
    for i, hit in enumerate(hits, 1):
        meta = hit.metadata or {}
        lines.append(f"\n[{i}] ({hit.source_type} #{hit.source_id}) score={hit.score:.3f}")
        if meta.get("product_built"):
            lines.append(f"  Product: {meta['product_built']}")
        if meta.get("title"):
            lines.append(f"  Title: {meta['title']}")
        lines.append(f"  Text: {hit.chunk_text[:300]}")
    return "\n".join(lines)
