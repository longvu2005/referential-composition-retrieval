"""Checkpoint names and validation selection fields; no model runtime imports."""

ARCHITECTURE_VERSION = "fafa-partial-match-v2"
# Naming changed, but tensor layouts and model equations did not.
COMPATIBLE_ARCHITECTURE_VERSIONS = (
    ARCHITECTURE_VERSION,
    "clip-b32-fafa-partial-match-v2",
)

SELECTED_RETRIEVAL_KEYS = (
    "fine_score_mode",
    "context_weight",
    "coarse_beta",
    "fine_coarse_weight",
    "top_m",
)
