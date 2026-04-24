"""Re-evidence parked findings with LLM augmentation to rescue candidates."""
import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
import sys
for p in [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.database import Database
from src.agents.evidence import EvidenceAgent
from src.messaging import MessageQueue, MessageType, Message


async def main():
    db = Database(str(PROJECT_ROOT / "data" / "autoresearch.db"))
    db.init_schema()

    import yaml
    with open(PROJECT_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    queue = MessageQueue()
    agent = EvidenceAgent(db, queue, config=config)

    # Find parked pain_signal findings with decent corroboration
    conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "autoresearch.db"))
    parked = conn.execute("""
        SELECT f.id, f.product_built, c.corroboration_score, c.recurrence_score,
               json_extract(c.evidence_json, '$.generalizability_class') as gc,
               json_extract(c.evidence_json, '$.source_family_diversity') as sfd
        FROM findings f
        JOIN corroborations c ON c.finding_id = f.id
        WHERE f.status = 'parked'
        AND f.source_class = 'pain_signal'
        AND c.corroboration_score >= 0.3
        ORDER BY c.corroboration_score DESC
    """).fetchall()
    conn.close()

    print(f"Found {len(parked)} parked pain_signal findings with corr >= 0.3")

    run_id = f"re_evidence_parked_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    db.set_active_run_id(run_id)

    target_ids = [row[0] for row in parked]
    results = {}

    for fid in target_ids:
        finding = db.get_finding(fid)
        if not finding:
            print(f"  Finding {fid} not found, skipping")
            continue

        atoms_dict = db.get_problem_atoms_for_findings([fid])
        atoms = atoms_dict.get(fid, [])
        if not atoms:
            print(f"  No atoms for finding {fid}, skipping")
            continue

        atom = atoms[0]
        title = (finding.product_built or "")[:55]
        print(f"\n--- Re-evidencing finding {fid}: {title}... ---")

        try:
            msg = Message(
                msg_id=f"re-ev-parked-{fid}",
                from_agent="script",
                to_agent="evidence",
                msg_type=MessageType.FINDING,
                payload={"finding_id": fid, "atom_id": atom.id},
                timestamp=datetime.now(timezone.utc),
                priority=2,
            )
            result = await agent.process(msg)
            results[fid] = result
            print(f"  Result: {json.dumps(result, default=str)[:200]}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Now check updated selection status for the opportunities linked to these findings
    print("\n\n=== UPDATED CORROBORATION RESULTS ===")
    conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "autoresearch.db"))
    for fid in target_ids:
        rows = conn.execute(
            "SELECT finding_id, corroboration_score, recurrence_score, "
            "json_extract(evidence_json, '$.source_family_diversity') as sfd, "
            "json_extract(evidence_json, '$.effective_family_diversity') as efd, "
            "json_extract(evidence_json, '$.generalizability_class') as gc, "
            "json_extract(evidence_json, '$.generalizability_score') as gs "
            "FROM corroborations WHERE finding_id = ? ORDER BY id DESC LIMIT 1",
            (fid,)
        ).fetchall()
        if rows:
            r = rows[0]
            print(f"Finding {fid}: corr={r[1]:.4f} rec={r[2]:.4f} sfd={r[3]} efd={r[4]} gc={r[5]} gs={r[6]}")
        else:
            print(f"Finding {fid}: no corroboration record")
    conn.close()

    # Check opportunities
    print("\n=== OPPORTUNITY SELECTION STATUS ===")
    conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "autoresearch.db"))
    opps = conn.execute("""
        SELECT o.id, o.cluster_id, o.title, o.selection_status, o.selection_reason,
               o.composite_score, o.evidence_quality, o.frequency_score
        FROM opportunities o
        ORDER BY o.composite_score DESC
    """).fetchall()
    for o in opps:
        title = (o[2] or "")[:50]
        print(f"  opp={o[0]} c={o[1]} status={o[3]:>22} reason={o[4] or '?':>30} cs={o[5]:.3f} eq={o[6]:.3f} fs={o[7]:.3f} | {title}")
    conn.close()

    db.close()


if __name__ == "__main__":
    asyncio.run(main())