"""Render main-project yaml configs from a template + explicit overrides.

The template lives in ``configs/sim2pipe/overlays/`` and holds everything
static about a pipe-probe run; the driver injects per-cell values (export
root, work dir, seed, split subjects) via a flat dot-path override dict,
e.g. ``{"preprocess.synthetic_imu_root": "/abs/path"}``.
"""

from __future__ import annotations

from pathlib import Path


def _set_dotted(cfg: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        nxt = node.get(part)
        if nxt is None:
            nxt = {}
            node[part] = nxt
        elif not isinstance(nxt, dict):
            raise TypeError(f"cannot descend into {dotted_key!r}: {part!r} is {type(nxt).__name__}")
        node = nxt
    node[parts[-1]] = value


def render_overlay(template_path: Path, overrides: dict, out_path: Path) -> Path:
    import yaml

    cfg = yaml.safe_load(Path(template_path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{template_path}: expected a yaml mapping")
    for key, value in overrides.items():
        _set_dotted(cfg, key, value)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_path
