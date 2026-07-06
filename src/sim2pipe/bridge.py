"""Boundary to the main project: paths config, dependency checks, subprocess CLI.

Machine-specific locations live in ``configs/sim2pipe/paths.yaml`` (git keeps
only ``paths.yaml.example``). Nothing here imports main-project code — every
interaction is a subprocess into the main repo's own python environment.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipePaths:
    main_repo: Path
    main_python: Path
    motionbert_ckpt: Path
    imu_ckpt: Path
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "PipePaths":
        import yaml  # deferred: PyYAML only needed on the driver path

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a mapping")
        required = ("main_repo", "main_python", "motionbert_ckpt", "imu_ckpt")
        missing = [k for k in required if not raw.get(k)]
        if missing:
            raise ValueError(f"{path}: missing required keys {missing}")
        known = {k: Path(str(raw[k])).expanduser() for k in required}
        extra = {k: v for k, v in raw.items() if k not in required}
        return cls(**known, extra=extra)


def check_dependencies(paths: PipePaths, corpus_root: Path | None = None) -> list[str]:
    """Return a human-readable list of missing prerequisites (empty = OK)."""
    problems: list[str] = []

    def need(cond: bool, msg: str) -> None:
        if not cond:
            problems.append(msg)

    need(paths.main_repo.is_dir(), f"main repo not found: {paths.main_repo}")
    need(
        (paths.main_repo / "src" / "engine" / "train.py").is_file(),
        f"main repo layout unexpected (src/engine/train.py missing): {paths.main_repo}",
    )
    need(paths.main_python.is_file(), f"main-project python not found: {paths.main_python}")
    need(paths.motionbert_ckpt.is_file(), f"MotionBERT ckpt not found: {paths.motionbert_ckpt}")
    need(paths.imu_ckpt.is_file(), f"DeSPITE IMU ckpt not found: {paths.imu_ckpt}")
    if corpus_root is not None:
        corpus_root = Path(corpus_root)
        need(corpus_root.is_dir(), f"sim2real corpus not found: {corpus_root}")

    if paths.main_python.is_file():
        probe = run_main_python(paths, ["-c", "import torch, yaml, scipy"], capture=True)
        need(
            probe.returncode == 0,
            f"main-project python cannot import torch/yaml/scipy: {probe.stderr.strip()[-200:]}",
        )
    return problems


def run_main_python(
    paths: PipePaths,
    argv: list[str],
    log_path: Path | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Run the main repo's python with its repo as cwd and on PYTHONPATH."""
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{paths.main_repo}:{paths.main_repo / 'src'}"
    cmd = [str(paths.main_python), *argv]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            return subprocess.run(cmd, cwd=paths.main_repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    return subprocess.run(cmd, cwd=paths.main_repo, env=env, capture_output=capture, text=True)


def run_main_module(
    paths: PipePaths,
    module: str,
    args: list[str],
    log_path: Path | None = None,
) -> subprocess.CompletedProcess:
    """``python -m <module> <args...>`` inside the main repo."""
    return run_main_python(paths, ["-m", module, *args], log_path=log_path)
