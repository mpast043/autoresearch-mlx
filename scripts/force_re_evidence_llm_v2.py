"""Force re-evidence on top findings with LLM augmentation."""
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
for p in [str(SRC_ROOT), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.database import Database
from src.agents.evidence import EvidenceAgent
from src.messaging import MessageQueue, MessageType


async def main():
    db = Database("data/autoresearch.db")
    db.init_schema()
    
    import yaml
    with open(PROJECT_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    
    queue = MessageQueue()
    agent = EvidenceAgent(db, queue, config=config)
    
    target_finding_ids = [27, 2191, 2197, 46, 2, 30]
    
    run_id = "re_evidence_llm_v2"
    db.set_active_run_id(run_id)
    
    from datetime import datetime, timezone
    from src.messaging import Message

    results = {}
    for fid in target_finding_ids:
        finding = db.get_finding(fid)
        if not finding:
            print(f"Finding {fid} not found, skipping")
            continue

        atoms_dict = db.get_problem_atoms_for_findings([fid])
        atoms = atoms_dict.get(fid, [])
        if not atoms:
            print(f"No atoms for finding {fid}, skipping")
            continue

        atom = atoms[0]
        print(f"\n--- Re-evidencing finding {fid}: {(finding.product_built or '')[:60]}... ---")

        try:
            msg = Message(
                msg_id=f"re-ev-{fid}",
                from_agent="script",
                to_agent="evidence",
                msg_type=MessageType.FINDING,
                payload={
                    "finding_id": fid,
                    "atom_id": atom.id,
                },
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
    
    print("\n\n=== UPDATED CORROBORATION RESULTS ===")
    import sqlite3
    conn = sqlite3.connect("data/autoresearch.db")
    for fid in target_finding_ids:
        rows = conn.execute(
            "SELECT finding_id, corroboration_score, recurrence_score, "
            "json_extract(evidence_json, '$.source_families') as src_families, "
            "json_extract(evidence_json, '$.source_family_diversity') as fam_div, "
            "json_extract(evidence_json, '$.generalizability_class') as gen_class, "
            "json_extract(evidence_json, '$.generalizability_score') as gen_score "
            "FROM corroborations WHERE finding_id = ? AND run_id = ? ORDER BY id DESC LIMIT 1",
            (fid, run_id)
        ).fetchall()
        if rows:
            r = rows[0]
            print(f"Finding {fid}: corr={r[1]:.4f} rec={r[2]:.4f} families={r[3]} fam_div={r[4]} gen_class={r[5]} gen_score={r[6]}")
        else:
            print(f"Finding {fid}: no corroboration record for this run")
    
    conn.close()
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
