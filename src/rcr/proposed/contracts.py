"""Checkpoint names and validation selection fields; no model runtime imports."""

ARCHITECTURE_VERSION = "fafa-partial-match-v2"
# Checkpoint parameter layouts shared by these architecture identifiers.
COMPATIBLE_ARCHITECTURE_VERSIONS = (
    ARCHITECTURE_VERSION,
    "clip-b32-fafa-partial-match-v2",
)

SELECTED_RETRIEVAL_KEYS = (
    "pooling",
    "fine_score_mode",
    "context_weight",
    "coarse_beta",
    "fine_coarse_weight",
    "top_m",
)
