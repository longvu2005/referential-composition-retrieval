"""Checkpoint names and validation selection fields; no model runtime imports."""

ARCHITECTURE_VERSION = "fafa-partial-match-v2"
# Checkpoint parameter layouts shared by these architecture identifiers.
COMPATIBLE_ARCHITECTURE_VERSIONS = (
    ARCHITECTURE_VERSION,
    "clip-b32-fafa-partial-match-v2",
)

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
