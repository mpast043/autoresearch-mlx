# v4 Calibration Artifact

**Date:** 2026-04-03
**Version:** v4_calibrated

## Final Threshold Values

| Parameter | Value | Source |
|------------|-------|--------|
| `promotion_threshold` | 0.18 | P90 of decision_score distribution |
| `park_threshold` | 0.15 | Unchanged |
| `PTS hard floor` | 0.10 | Below P50 (0.107) |
| `PTS promote floor` | 0.11 | P50 of PTS distribution |
| `RRS promote floor` | 0.22 | P50 of RRS distribution |
| `frequency gate` | 0.25 | Unchanged (not a bottleneck) |

## Rationale

- **promotion_threshold (0.18):** P90 of decision_score distribution in current data
- **PTS/RRS floors (0.11/0.22):** P50 - original floors (0.35/0.30) were far above distribution (>P95)
- **Hard PTS floor (0.10):** Previous value (0.20) was killing candidates above P90; lowered to below P50
- **Frequency (0.25):** Not the bottleneck - 29.5% pass; kept as-is

## Distribution Expectations

| Decision | Expected | % |
|----------|----------|---|
| Promote | 7 | 6.2% |
| Park | 24 | 21.4% |
| Kill | 81 | 72.3% |
| **Total** | 112 | 100% |

## Current Database State

| Status | Count |
|--------|-------|
| promoted | 1 (ID 6 - old thresholds) |
| parked | 1 (ID 28 - old thresholds) |
| killed | 110 |

**Note:** Existing rows were scored with old thresholds. New validation runs will use calibrated values.

## Promoted Candidates (simulated)

| ID | Title | Decision | PTS | RRS | Freq | Notes |
|----|-------|----------|-----|-----|------|-------|
| 28 | Smartsheet spreadsheet pain | 0.347 | 0.334 | 0.363 | 0.410 | High-quality |
| 6 | Month-end reconciliation | 0.341 | 0.355 | 0.325 | 0.418 | High-quality |
| 112 | Resume/interview script | 0.241 | 0.178 | 0.319 | 0.252 | Legit |
| 14 | Listing setup/shipping | 0.192 | 0.132 | 0.265 | 0.360 | Borderline |
| 86 | Operations workflow | 0.186 | 0.137 | 0.246 | 0.480 | Borderline |
| 5 | Finance reconciliation | 0.184 | 0.152 | 0.223 | 0.360 | Borderline |
| 64 | Template onboarding | 0.182 | 0.147 | 0.225 | 0.360 | Borderline |

**Sanity Check:** Top 2 are clearly legitimate operational pain. Next 5 are borderline but valid signals.

## Files Updated

- `config.yaml`: promotion_threshold: 0.18, park_threshold: 0.15
- `src/opportunity_engine.py`: SCORING_THRESHOLDS["v4"], stage_decision defaults, diagnose_stage_decision defaults

## Version Metadata

- `threshold_version`: 2025_q2
- `formula_version`: pts_rrs_v1

## Risks

1. **Rescoring needed:** Existing rows were scored with old thresholds. Full recalibration requires re-running validation.
2. **Borderline candidates:** 5 of 7 promoted candidates are just above P50 floors. This is intentional (based on P50) but may include false positives.
3. **Test inconsistency:** behavior_eval.py uses old defaults; may not match production.

## Verdict

**Ready for operational use** with minor follow-up:
- New validation runs will use calibrated thresholds
- Existing rows need rescoring for full alignment
- Monitor borderline candidates for quality