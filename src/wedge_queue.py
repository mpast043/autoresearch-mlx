"""Persistent wedge queue for continuous builder pipeline.

This module replaces the one-time winner/backup selector with an ongoing
wedge discovery builder pipeline that continuously:
1. discovers new pain signals
2. extracts wedge-ready failures
3. validates and scores them
4. emits builder-ready commercialization cards
5. ranks active build candidates
6. iterates for additional narrow wedges
7. updates the queue as new evidence appears
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Queue status definitions
class BuilderStatus:
    WATCHING = "watching"
    VALIDATING = "validating"
    BUILDABLE = "buildable"
    BUILDING = "building"
    DEPRIORITIZED = "deprioritized"
    REJECTED = "rejected"


# Wedge classification for iteration
class WedgeClassification:
    NEW_QUEUE_ITEM = "new_queue_item"
    DUPLICATE_OF_EXISTING = "duplicate_of_existing"
    NARROWER_CHILD = "narrower_child"
    BROADER_PARENT = "broader_parent"
    INSUFFICIENTLY_DISTINCT = "insufficiently_distinct"
    REJECT = "reject"


# Queue priority tiers
class QueuePriority:
    IMMEDIATE = 1  # Ready to build now
    HIGH = 2      # Strong candidate, minor validation needed
    MEDIUM = 3    # Good candidate, needs evidence
    LOW = 4       # Weak candidate, needs significant evidence
    REJECT = 5   # Not a real opportunity


@dataclass
class WedgeQueueItem:
    """Persistent queue item for the wedge build queue."""
    wedge_id: str
    wedge_title: str
    exact_user: str
    exact_workflow: str
    exact_trigger: str
    exact_failure: str
    exact_consequence: str
    host_platform: str
    product_shape: str
    evidence_summary: str
    software_fit_score: float = 0.0
    monetization_fit_score: float = 0.0
    trust_risk: str = "low"
    recurrence_confidence: float = 0.0
    builder_status: str = BuilderStatus.WATCHING
    queue_priority: int = QueuePriority.MEDIUM
    last_updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    next_required_evidence: str = ""
    next_builder_action: str = ""
    # MVP and monetization
    mvp_in_scope: list[str] = field(default_factory=list)
    mvp_out_of_scope: list[str] = field(default_factory=list)
    first_paid_offer: str = ""
    pricing_hypothesis: str = ""
    first_customer: str = ""
    first_channel: str = ""
    why_this_is_narrow: str = ""
    why_this_could_make_money: str = ""
    # Parent/child relationships
    parent_wedge_id: Optional[str] = None
    child_wedge_ids: list[str] = field(default_factory=list)
    # Classification
    classification_reason: str = ""
    deprioritization_reason: str = ""


class WedgeQueue:
    """Persistent queue for managing wedges through the builder pipeline."""

    def __init__(self, db_path: str = "data/wedge_queue.json"):
        self.db_path = Path(db_path)
        self.queue: dict[str, WedgeQueueItem] = {}
        self._load()

    def _load(self) -> None:
        """Load queue from disk."""
        if self.db_path.exists():
            try:
                data = json.loads(self.db_path.read_text())
                self.queue = {
                    k: WedgeQueueItem(**v)
                    for k, v in data.get("queue", {}).items()
                }
                logger.info(f"Loaded wedge queue with {len(self.queue)} items")
            except Exception as e:
                logger.warning(f"Failed to load wedge queue: {e}")
                self.queue = {}

    def _save(self) -> None:
        """Save queue to disk."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "queue": {k: asdict(v) for k, v in self.queue.items()},
            "last_updated": datetime.utcnow().isoformat(),
            "total_items": len(self.queue),
            "buildable_count": len([v for v in self.queue.values() if v.builder_status == BuilderStatus.BUILDABLE]),
        }
        self.db_path.write_text(json.dumps(data, indent=2))

    def add_or_update_wedge(
        self,
        wedge_id: str,
        wedge_data: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> tuple[WedgeClassification, Optional[WedgeQueueItem]]:
        """Add or update a wedge in the queue.

        Returns:
            - Classification of how this wedge relates to existing queue
            - The queue item (new or updated)
        """
        # Check for duplicates
        existing = self._find_exact_duplicate(wedge_data)
        if existing:
            # Update existing with new evidence
            self._update_existing(existing, evidence)
            return WedgeClassification.DUPLICATE_OF_EXISTING, self.queue[existing]

        # Check if this is a narrower child of existing
        parent = self._find_broader_parent(wedge_data)
        if parent:
            child = self._create_child_wedge(wedge_id, wedge_data, evidence, parent)
            return WedgeClassification.NARROWER_CHILD, child

        # Check if this is a broader parent of existing
        child = self._find_narrower_child(wedge_data)
        if child:
            return WedgeClassification.BROADER_PARENT, None

        # Check if insufficiently distinct
        if not self._is_sufficiently_distinct(wedge_data):
            return WedgeClassification.INSUFFICIENTLY_DISTINCT, None

        # Check if should reject
        if self._should_reject(wedge_data):
            return WedgeClassification.REJECT, None

        # Create new queue item
        item = self._create_queue_item(wedge_id, wedge_data, evidence)
        self.queue[wedge_id] = item
        self._save()
        return WedgeClassification.NEW_QUEUE_ITEM, item

    def _find_exact_duplicate(self, wedge_data: dict[str, Any]) -> Optional[str]:
        """Find exact duplicate by comparing core fields."""
        for wedge_id, item in self.queue.items():
            if (item.exact_failure == wedge_data.get("exact_failure") and
                item.exact_user == wedge_data.get("exact_user")):
                return wedge_id
        return None

    def _find_broader_parent(self, wedge_data: dict[str, Any]) -> Optional[str]:
        """Find an existing wedge that is broader than this one."""
        for wedge_id, item in self.queue.items():
            # If same failure but we have more specific user/workflow
            if item.exact_failure == wedge_data.get("exact_failure"):
                if (len(wedge_data.get("exact_user", "")) > len(item.exact_user) or
                    len(wedge_data.get("exact_workflow", "")) > len(item.exact_workflow)):
                    return wedge_id
        return None

    def _find_narrower_child(self, wedge_data: dict[str, Any]) -> Optional[str]:
        """Find an existing wedge that is narrower than this one."""
        for wedge_id, item in self.queue.items():
            if item.parent_wedge_id:
                continue  # Skip already children
            # If we are broader and there's a more specific child
            if (item.exact_failure == wedge_data.get("exact_failure") and
                len(item.exact_user) < len(wedge_data.get("exact_user", ""))):
                return wedge_id
        return None

    def _is_sufficiently_distinct(self, wedge_data: dict[str, Any]) -> bool:
        """Check if wedge is sufficiently distinct from existing queue items."""
        for item in self.queue.values():
            # Must differ in at least two core fields
            differences = 0
            if item.exact_user != wedge_data.get("exact_user"):
                differences += 1
            if item.exact_workflow != wedge_data.get("exact_workflow"):
                differences += 1
            if item.host_platform != wedge_data.get("host_platform"):
                differences += 1

            if differences < 2:
                return False
        return True

    def _should_reject(self, wedge_data: dict[str, Any]) -> bool:
        """Determine if wedge should be rejected."""
        # Reject if not software-first
        software_score = wedge_data.get("software_fit_score", 0)
        if software_score < 0.3:
            return True

        # Reject if too broad
        workflow = wedge_data.get("exact_workflow", "").lower()
        broad_words = ["platform", "all systems", "any csv", "universal", "everything"]
        if any(w in workflow for w in broad_words):
            return True

        # Reject if monetization is weak
        if wedge_data.get("monetization_fit_score", 0) < 0.2:
            return True

        return False

    def _create_queue_item(
        self,
        wedge_id: str,
        wedge_data: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> WedgeQueueItem:
        """Create a new queue item."""
        # Determine status and priority
        software_score = wedge_data.get("software_fit_score", 0)
        monetization_score = wedge_data.get("monetization_fit_score", 0)
        trust_risk = wedge_data.get("trust_risk", "low")

        if software_score >= 0.7 and monetization_score >= 0.5 and trust_risk != "high":
            status = BuilderStatus.BUILDABLE
            priority = QueuePriority.IMMEDIATE
        elif software_score >= 0.5:
            status = BuilderStatus.VALIDATING
            priority = QueuePriority.HIGH
        else:
            status = BuilderStatus.WATCHING
            priority = QueuePriority.MEDIUM

        # Build action based on status
        if status == BuilderStatus.BUILDABLE:
            next_action = "Build MVP now"
        elif status == BuilderStatus.VALIDATING:
            next_action = "Gather additional evidence to confirm buildability"
        else:
            next_action = "Monitor for more evidence"

        return WedgeQueueItem(
            wedge_id=wedge_id,
            wedge_title=wedge_data.get("wedge_title", "")[:80],
            exact_user=wedge_data.get("exact_user", ""),
            exact_workflow=wedge_data.get("exact_workflow", ""),
            exact_trigger=wedge_data.get("exact_trigger", ""),
            exact_failure=wedge_data.get("exact_failure", ""),
            exact_consequence=wedge_data.get("exact_consequence", ""),
            host_platform=wedge_data.get("host_platform", ""),
            product_shape=wedge_data.get("product_shape", ""),
            evidence_summary=wedge_data.get("evidence_summary", ""),
            software_fit_score=software_score,
            monetization_fit_score=monetization_score,
            trust_risk=trust_risk,
            recurrence_confidence=wedge_data.get("recurrence_confidence", 0.5),
            builder_status=status,
            queue_priority=priority,
            next_required_evidence=wedge_data.get("next_required_evidence", ""),
            next_builder_action=next_action,
            mvp_in_scope=wedge_data.get("mvp_in_scope", []),
            mvp_out_of_scope=wedge_data.get("mvp_out_of_scope", []),
            first_paid_offer=wedge_data.get("first_paid_offer", ""),
            pricing_hypothesis=wedge_data.get("pricing_hypothesis", ""),
            first_customer=wedge_data.get("first_customer", ""),
            first_channel=wedge_data.get("first_channel", ""),
            why_this_is_narrow=wedge_data.get("why_this_is_narrow", ""),
            why_this_could_make_money=wedge_data.get("why_this_could_make_money", ""),
        )

    def _create_child_wedge(
        self,
        wedge_id: str,
        wedge_data: dict[str, Any],
        evidence: list[dict[str, Any]],
        parent_id: str,
    ) -> WedgeQueueItem:
        """Create a narrower child wedge."""
        item = self._create_queue_item(wedge_id, wedge_data, evidence)
        item.parent_wedge_id = parent_id
        self.queue[parent_id].child_wedge_ids.append(wedge_id)
        self.queue[wedge_id] = item
        self._save()
        return item

    def _update_existing(self, wedge_id: str, evidence: list[dict[str, Any]]) -> None:
        """Update existing wedge with new evidence."""
        item = self.queue[wedge_id]
        item.last_updated_at = datetime.utcnow().isoformat()
        # Could update scores, evidence summary, etc.
        self._save()

    def transition_status(
        self,
        wedge_id: str,
        new_status: str,
        reason: str = "",
    ) -> bool:
        """Transition a wedge to a new status."""
        if wedge_id not in self.queue:
            return False

        item = self.queue[wedge_id]
        old_status = item.builder_status

        # Validate transition
        allowed = {
            (BuilderStatus.WATCHING, BuilderStatus.VALIDATING),
            (BuilderStatus.VALIDATING, BuilderStatus.BUILDABLE),
            (BuilderStatus.BUILDABLE, BuilderStatus.BUILDING),
            (BuilderStatus.VALIDATING, BuilderStatus.DEPRIORITIZED),
            (BuilderStatus.BUILDABLE, BuilderStatus.DEPRIORITIZED),
            (BuilderStatus.WATCHING, BuilderStatus.REJECTED),
            (BuilderStatus.VALIDATING, BuilderStatus.REJECTED),
            (BuilderStatus.BUILDABLE, BuilderStatus.REJECTED),
        }

        if (old_status, new_status) not in allowed:
            logger.warning(f"Invalid transition {old_status} -> {new_status} for {wedge_id}")
            return False

        item.builder_status = new_status
        item.last_updated_at = datetime.utcnow().isoformat()

        if new_status == BuilderStatus.DEPRIORITIZED:
            item.deprioritization_reason = reason
            item.queue_priority = QueuePriority.REJECT
        elif new_status == BuilderStatus.BUILDABLE:
            item.queue_priority = QueuePriority.IMMEDIATE
            item.next_builder_action = "Build MVP now"

        self._save()
        return True

    def get_buildable_wedges(self) -> list[WedgeQueueItem]:
        """Get all buildable wedges."""
        return [
            item for item in self.queue.values()
            if item.builder_status == BuilderStatus.BUILDABLE
        ]

    def get_watching_wedges(self) -> list[WedgeQueueItem]:
        """Get all watching wedges."""
        return [
            item for item in self.queue.values()
            if item.builder_status == BuilderStatus.WATCHING
        ]

    def get_validating_wedges(self) -> list[WedgeQueueItem]:
        """Get all validating wedges."""
        return [
            item for item in self.queue.values()
            if item.builder_status == BuilderStatus.VALIDATING
        ]

    def get_deprioritized_wedges(self) -> list[WedgeQueueItem]:
        """Get all deprioritized/rejected wedges."""
        return [
            item for item in self.queue.values()
            if item.builder_status in (BuilderStatus.DEPRIORITIZED, BuilderStatus.REJECTED)
        ]

    def get_queue_summary(self) -> dict[str, Any]:
        """Get queue summary for output."""
        return {
            "total_items": len(self.queue),
            "buildable": len(self.get_buildable_wedges()),
            "validating": len(self.get_validating_wedges()),
            "watching": len(self.get_watching_wedges()),
            "deprioritized": len(self.get_deprioritized_wedges()),
            "last_updated": max(
                (item.last_updated_at for item in self.queue.values()),
                default=None
            ),
        }


def process_wedges_for_queue(
    db,
    wedge_queue: WedgeQueue,
) -> dict[str, Any]:
    """Process current wedges and update the queue.

    This is the main entry point for the builder queue output stage.
    It runs after wedge validation and emits builder-ready queue items.
    """

    from src.builder_output import generate_builder_outputs
    from src.wedge_queue import WedgeClassification

    # Generate builder cards from current wedges
    builder_cards = generate_builder_outputs(db)

    results = {
        "new_queue_items": [],
        "updated_queue_items": [],
        "buildable_wedges": [],
        "validating_wedges": [],
        "deprioritized_rejected": [],
        "additional_narrow_wedges": [],
        "next_actions": [],
    }

    # Process each builder card
    for card in builder_cards:
        wedge_data = {
            "wedge_title": card.wedge_title,
            "exact_user": card.exact_user,
            "exact_workflow": card.exact_workflow,
            "exact_trigger": card.exact_trigger,
            "exact_failure": card.exact_failure,
            "exact_consequence": card.exact_consequence,
            "host_platform": card.host_platform,
            "product_shape": card.product_shape,
            "evidence_summary": card.evidence_summary,
            "software_fit_score": card.software_fit_score,
            "monetization_fit_score": card.monetization_fit_score,
            "trust_risk": card.trust_risk,
            "mvp_in_scope": card.mvp_in_scope,
            "mvp_out_of_scope": card.mvp_out_of_scope,
            "first_paid_offer": card.first_paid_offer,
            "pricing_hypothesis": card.pricing_hypothesis,
            "first_customer": card.first_customer,
            "first_channel": card.first_channel,
            "why_this_is_narrow": card.why_this_is_narrow,
            "why_this_could_make_money": card.why_this_could_make_money,
        }

        wedge_id = f"wedge_{card.wedge_id}"
        evidence = []  # Would fetch actual evidence

        classification, item = wedge_queue.add_or_update_wedge(
            wedge_id, wedge_data, evidence
        )

        if classification == WedgeClassification.NEW_QUEUE_ITEM:
            results["new_queue_items"].append({
                "wedge_id": wedge_id,
                "title": card.wedge_title,
                "status": item.builder_status,
                "action": item.next_builder_action,
            })
        elif classification == WedgeClassification.NARROWER_CHILD:
            results["additional_narrow_wedges"].append({
                "wedge_id": wedge_id,
                "parent": item.parent_wedge_id,
                "title": card.wedge_title,
            })

    # Get current state
    buildable = wedge_queue.get_buildable_wedges()
    validating = wedge_queue.get_validating_wedges()
    watching = wedge_queue.get_watching_wedges()
    deprioritized = wedge_queue.get_deprioritized_wedges()

    results["buildable_wedges"] = [
        {
            "wedge_id": w.wedge_id,
            "title": w.wedge_title,
            "product_shape": w.product_shape,
            "mvp_in_scope": w.mvp_in_scope,
            "next_action": w.next_builder_action,
        }
        for w in buildable
    ]

    results["validating_wedges"] = [
        {
            "wedge_id": w.wedge_id,
            "title": w.wedge_title,
            "needed": w.next_required_evidence,
        }
        for w in validating
    ]

    results["deprioritized_rejected"] = [
        {
            "wedge_id": w.wedge_id,
            "title": w.wedge_title,
            "reason": w.deprioritization_reason,
        }
        for w in deprioritized
    ]

    # Add next actions for buildable
    for w in buildable:
        results["next_actions"].append({
            "wedge_id": w.wedge_id,
            "action": w.next_builder_action,
            "customer": w.first_customer,
            "channel": w.first_channel,
        })

    results["queue_summary"] = wedge_queue.get_queue_summary()

    return results


# Run-level output structure
def generate_run_output(wedge_queue: WedgeQueue) -> dict[str, Any]:
    """Generate the complete run-level output."""

    buildable = wedge_queue.get_buildable_wedges()
    validating = wedge_queue.get_validating_wedges()
    watching = wedge_queue.get_watching_wedges()
    deprioritized = wedge_queue.get_deprioritized_wedges()

    return {
        "queue_state": {
            "total_items": len(wedge_queue.queue),
            "buildable": len(buildable),
            "validating": len(validating),
            "watching": len(watching),
            "deprioritized_rejected": len(deprioritized),
        },
        "buildable_wedges": [
            {
                "id": w.wedge_id,
                "title": w.wedge_title,
                "user": w.exact_user,
                "workflow": w.exact_workflow,
                "failure": w.exact_failure,
                "product_shape": w.product_shape,
                "mvp_in": w.mvp_in_scope,
                "mvp_out": w.mvp_out_of_scope,
                "pricing": w.pricing_hypothesis,
                "first_customer": w.first_customer,
                "first_channel": w.first_channel,
                "next_action": w.next_builder_action,
            }
            for w in buildable
        ],
        "validating_wedges": [
            {
                "id": w.wedge_id,
                "title": w.wedge_title,
                "needed_evidence": w.next_required_evidence,
            }
            for w in validating
        ],
        "does_anything_else_exist": {
            "additional_narrow_found": len(watching) + len([w for w in validating if w.recurrence_confidence > 0.5]),
            "are_they_distinct": len(watching) > 0,
            "are_they_better_than_buildable": False,  # Would need comparison logic
            "are_they_broader_variants": len([w for w in watching if "platform" in w.exact_workflow.lower()]) > 0,
        },
    }