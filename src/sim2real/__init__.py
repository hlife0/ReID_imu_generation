"""sim2real: downstream-task evaluation of synthetic IMU quality.

Ranks the generation pipelines by how useful their synthetic IMU is for a
real downstream task (train-on-synthetic, test-on-real retrieval) instead of
signal similarity alone.
"""

from .contracts import (
    ImuSequence,
    MotionSequence,
    REAL_SOURCE,
    SCHEMA_VERSION,
    canonical_json,
    config_hash,
    load_manifest,
    parse_source,
    source_to_token,
    synth_source,
    write_manifest,
)
from .splits import (
    SPLIT_NAMES,
    SplitSpec,
    find_leakage,
    load_split,
    shard_split,
    subject_of_sequence,
)

__all__ = [
    "ImuSequence",
    "MotionSequence",
    "REAL_SOURCE",
    "SCHEMA_VERSION",
    "SPLIT_NAMES",
    "SplitSpec",
    "canonical_json",
    "config_hash",
    "find_leakage",
    "load_manifest",
    "load_split",
    "parse_source",
    "shard_split",
    "source_to_token",
    "subject_of_sequence",
    "synth_source",
    "write_manifest",
]
