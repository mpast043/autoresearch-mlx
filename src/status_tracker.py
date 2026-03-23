"""Simple JSON status tracker for runtime/operator visibility."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class StatusTracker:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.status = {
            "runId": "",
            "stage": "idle",
            "status": "idle",
            "startTime": "",
            "discoveries": 0,
            "rawSignals": 0,
            "problemAtoms": 0,
            "clusters": 0,
            "experiments": 0,
            "ledgerEntries": 0,
            "validated": 0,
            "ideas": 0,
            "built": 0,
            "logs": [],
            "opportunities": [],
            "validatedIdeas": [],
            "generatedIdeas": [],
            "mvps": [],
            "sourceHealth": [],
            "discoveryStrategy": [],
            "learningInsights": [],
            "decisionBreakdown": {},
            "recentValidationReview": [],
        }
        self._save()

    def reset(self) -> None:
        self.status.update(
            {
                "runId": "",
                "stage": "idle",
                "status": "idle",
                "startTime": "",
                "discoveries": 0,
                "rawSignals": 0,
                "problemAtoms": 0,
                "clusters": 0,
                "experiments": 0,
                "ledgerEntries": 0,
                "validated": 0,
                "ideas": 0,
                "built": 0,
                "logs": [],
                "opportunities": [],
                "validatedIdeas": [],
                "generatedIdeas": [],
                "mvps": [],
                "sourceHealth": [],
                "discoveryStrategy": [],
                "learningInsights": [],
                "decisionBreakdown": {},
                "recentValidationReview": [],
            }
        )
        self._save()

    def start_run(self, run_id: str) -> None:
        self.status["runId"] = run_id
        self._save()

    def set_stage(self, stage: str) -> None:
        if not self.status["startTime"]:
            self.status["startTime"] = datetime.now().isoformat()
        self.status["stage"] = stage
        self.status["status"] = "running"
        self.log(f"stage={stage}")

    def update(self, **kwargs) -> None:
        self.status.update(kwargs)
        self._save()

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status["logs"] = (self.status["logs"] + [f"{timestamp} {message}"])[-200:]
        self._save()

    def complete(self) -> None:
        self.status["status"] = "completed"
        self.status["stage"] = "completed"
        self.log("pipeline completed")

    def fail(self, message: str) -> None:
        self.status["status"] = "failed"
        self.log(f"error={message}")

    def _save(self) -> None:
        path = self.output_dir / "pipeline_status.json"
        path.write_text(json.dumps(self.status, indent=2))
