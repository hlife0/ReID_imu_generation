"""Append-only results ledger, same discipline as sim2real's results.jsonl.

One JSON object per line; aggregation only ever reads the ledger; matrix
drivers skip cells whose key is already present (resume-first, shared-GPU
reality). The cell key for sim2pipe P1 is
(protocol, imu_stream, motion_source, seed).
"""

from __future__ import annotations

import json
from pathlib import Path

KEY_FIELDS = ("protocol", "imu_stream", "motion_source", "seed")


def cell_key(row: dict) -> tuple:
    try:
        return tuple(row[k] for k in KEY_FIELDS)
    except KeyError as exc:
        raise KeyError(f"ledger row missing key field {exc}; row keys: {sorted(row)}") from None


def load_rows(ledger_path: Path) -> list[dict]:
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return []
    rows = []
    for i, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{ledger_path}:{i + 1}: corrupt ledger line: {exc}") from None
    return rows


def done_keys(ledger_path: Path) -> set:
    return {cell_key(row) for row in load_rows(ledger_path)}


def append_row(ledger_path: Path, row: dict) -> None:
    cell_key(row)  # validate before writing
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
