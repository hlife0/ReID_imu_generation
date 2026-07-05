"""Frozen subject-level split registry plus leakage checks.

Rules (see docs/sim2real_design.md, "切分与泄漏"):

- Splits are frozen JSON files under ``configs/sim2real/splits/`` and are
  committed to git BEFORE any corpus generation. A frozen split is never
  edited; changes mean a new file (``..._v2.json``).
- Splits are subject-disjoint. In the retrieval task the motion content is
  half of the label, so any motion shared across train/test inflates results;
  splitting by subject removes sequence- and take-level leakage in one cut.
- Window shards are named ``<split>__<source_token>.npz``. ``find_leakage``
  verifies every shard only contains subjects belonging to that shard's
  split; ``tests/test_no_split_leakage.py`` keeps this executable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

SPLIT_NAMES = ("train", "val", "test")

_SUBJECT_RE = re.compile(r"^([Ss]\d+)(?:[_\-.].*)?$")


def subject_of_sequence(sequence_name: str) -> str:
    """Extract the normalized subject id from a TotalCapture sequence name.

    ``"S1_freestyle3"`` and ``"s1_freestyle3"`` both map to ``"S1"``.
    """
    match = _SUBJECT_RE.match(sequence_name.strip())
    if not match:
        raise ValueError(
            f"cannot parse subject from sequence name {sequence_name!r} "
            "(expected e.g. 'S1_freestyle3')"
        )
    return match.group(1).upper()


def shard_split(shard_name: str) -> str:
    """Map a shard name to its split: ``train__synth_globalpose_a3f2`` -> ``train``."""
    stem = shard_name.split(".")[0]
    prefix = stem.split("__", 1)[0]
    if prefix not in SPLIT_NAMES:
        raise ValueError(
            f"shard name {shard_name!r} must start with one of {SPLIT_NAMES} "
            "(convention: '<split>__<source_token>')"
        )
    return prefix


@dataclass(frozen=True)
class SplitSpec:
    name: str
    dataset: str
    train_subjects: tuple
    val_subjects: tuple
    test_subjects: tuple
    frozen: str
    rationale: str

    def subjects(self, split: str) -> tuple:
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown split {split!r}, expected one of {SPLIT_NAMES}")
        return getattr(self, f"{split}_subjects")

    def all_subjects(self) -> frozenset:
        return frozenset(self.train_subjects + self.val_subjects + self.test_subjects)

    def split_of_subject(self, subject: str) -> str:
        subject = subject.upper()
        for split in SPLIT_NAMES:
            if subject in self.subjects(split):
                return split
        raise KeyError(
            f"subject {subject!r} is not covered by split {self.name!r} "
            f"(covered: {sorted(self.all_subjects())})"
        )

    def split_of_sequence(self, sequence_name: str) -> str:
        return self.split_of_subject(subject_of_sequence(sequence_name))


def _normalized_subjects(raw: Iterable[str], field: str, path: Path) -> tuple:
    subjects = tuple(str(s).strip().upper() for s in raw)
    for subject in subjects:
        if not re.fullmatch(r"S\d+", subject):
            raise ValueError(f"{path}: {field} contains invalid subject id {subject!r}")
    if len(set(subjects)) != len(subjects):
        raise ValueError(f"{path}: {field} contains duplicates: {subjects}")
    return subjects


def load_split(path: Path) -> SplitSpec:
    """Load and validate a frozen split file."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    for key in ("name", "dataset", "train_subjects", "val_subjects", "test_subjects", "frozen"):
        if key not in raw:
            raise ValueError(f"{path}: missing required key {key!r}")

    train = _normalized_subjects(raw["train_subjects"], "train_subjects", path)
    val = _normalized_subjects(raw["val_subjects"], "val_subjects", path)
    test = _normalized_subjects(raw["test_subjects"], "test_subjects", path)

    if not train or not test:
        raise ValueError(f"{path}: train_subjects and test_subjects must be non-empty")
    overlap = (set(train) & set(val)) | (set(train) & set(test)) | (set(val) & set(test))
    if overlap:
        raise ValueError(f"{path}: split is not subject-disjoint, overlap: {sorted(overlap)}")

    return SplitSpec(
        name=str(raw["name"]),
        dataset=str(raw["dataset"]),
        train_subjects=train,
        val_subjects=val,
        test_subjects=test,
        frozen=str(raw["frozen"]),
        rationale=str(raw.get("rationale", "")),
    )


def find_leakage(spec: SplitSpec, shard_subjects: Mapping[str, Iterable[str]]) -> list:
    """Check window shards against a split spec.

    ``shard_subjects`` maps shard names (``'<split>__<source_token>'``) to the
    subjects whose windows the shard contains. Returns a list of violation
    messages; an empty list means the shards are clean. Note this checks
    subject placement only — "val/test shards must be real" is enforced at the
    protocol layer, not here.
    """
    violations = []
    for shard_name in sorted(shard_subjects):
        try:
            split = shard_split(shard_name)
        except ValueError as exc:
            violations.append(str(exc))
            continue
        for subject in sorted({str(s).upper() for s in shard_subjects[shard_name]}):
            try:
                expected = spec.split_of_subject(subject)
            except KeyError:
                violations.append(
                    f"shard {shard_name!r}: subject {subject!r} is not in split {spec.name!r}"
                )
                continue
            if expected != split:
                violations.append(
                    f"shard {shard_name!r}: subject {subject!r} belongs to "
                    f"{expected!r}, not {split!r} — LEAKAGE"
                )
    return violations
