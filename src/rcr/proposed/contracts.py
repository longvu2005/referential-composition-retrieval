"""Checkpoint names and validation selection fields; no model runtime imports."""

ARCHITECTURE_VERSION = "fafa-subject-reasoner-v3"

FIXED_FINE_KEYS = (
    "pooling",
    "fine_score_mode",
)

CALIBRATION_PARAMETERS = {
    "coverage_weights": "coverage_weight",
    "context_weights": "context_weight",
    "fine_coarse_weights": "fine_coarse_weight",
}
CALIBRATED_RETRIEVAL_KEYS = tuple(CALIBRATION_PARAMETERS.values())

SELECTED_RETRIEVAL_KEYS = (
    *FIXED_FINE_KEYS,
    *CALIBRATED_RETRIEVAL_KEYS,
    "coarse_beta",
    "top_m",
)
