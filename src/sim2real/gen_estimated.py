"""Estimated-skeleton generator — corpus layer A (estskel variant).

Turns the clean 21-joint SMPL-X ground-truth skeleton (``motion.npz``) into a
17-joint H36M skeleton that mimics what the main project's real pipeline
actually consumes: ``video -> ByteTrack+AlphaPose -> 2D->3D lift (MotionBERT)
-> H36M-17 3D``. The main matcher never sees pixels — it sees this estimated
skeleton — so the faithful sim2real proxy is a *degraded* skeleton, not raw
images. See ``docs/sim2real_findings_v1.md`` and the estskel plan.

Two independent transforms, applied in order:

1. **Retarget 21 -> H36M-17.** A pure index remap; every H36M joint has a
   direct pipeline source (hands/feet are dropped, exactly as the real
   pipeline drops them). No joints are fabricated.
2. **Estimation noise.** Deterministic given ``seed``: per-joint Gaussian
   jitter (larger at end-effectors), light temporal low-pass (MotionBERT
   smoothing), and occlusion runs where a joint is dropped and linearly
   interpolated back (missed detections). Literature-default magnitudes,
   fully parameterized via config — recalibrate against real AlphaPose output
   when it is available.

Pure numpy; no SMPL-X, no torch, no external venv. Runs under the repo python.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

# Source layout: the 21-joint pipeline order produced by
# data_generation/src/smplx_ops/joints.py::JOINT_NAMES. Kept as an explicit
# list here so this module has no dependency on the generation repo.
PIPELINE_JOINT_NAMES = (
    "pelvis", "spine1", "spine2", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist", "left_hand",
    "right_shoulder", "right_elbow", "right_wrist", "right_hand",
    "left_hip", "left_knee", "left_ankle", "left_foot",
    "right_hip", "right_knee", "right_ankle", "right_foot",
)
_PIPE_IDX = {name: i for i, name in enumerate(PIPELINE_JOINT_NAMES)}

# Target layout: standard H36M-17 order consumed by MotionBERT.
H36M17_JOINT_NAMES = (
    "hip", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "spine", "thorax", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
)
ESTIMATED_LAYOUT = "estimated_h36m17_v1"

# H36M joint -> pipeline joint it is sourced from. Every target is covered.
_RETARGET = {
    "hip": "pelvis",
    "right_hip": "right_hip", "right_knee": "right_knee", "right_ankle": "right_ankle",
    "left_hip": "left_hip", "left_knee": "left_knee", "left_ankle": "left_ankle",
    "spine": "spine1", "thorax": "spine2", "neck": "neck", "head": "head",
    "left_shoulder": "left_shoulder", "left_elbow": "left_elbow", "left_wrist": "left_wrist",
    "right_shoulder": "right_shoulder", "right_elbow": "right_elbow", "right_wrist": "right_wrist",
}
# Precomputed source-index vector: output[:, k] = input[:, _SRC_IDX[k]].
_SRC_IDX = np.array([_PIPE_IDX[_RETARGET[name]] for name in H36M17_JOINT_NAMES], dtype=np.int64)

# End-effectors get larger estimation error than the torso (documented pose
# estimator behavior). Values are jitter multipliers on ``jitter_sigma_m``.
_END_EFFECTORS = frozenset({"right_wrist", "left_wrist", "right_ankle", "left_ankle", "head"})

DEFAULT_CONFIG = {
    "seed": 0,
    "jitter_sigma_m": 0.015,          # torso per-joint Gaussian sigma (metres)
    "end_effector_sigma_mult": 2.3,   # end-effectors ~35mm vs torso ~15mm
    "temporal_smooth_alpha": 0.35,    # EMA weight for MotionBERT-like smoothing
    "occlusion_rate_per_joint": 0.02, # expected occlusion runs per joint per second
    "occlusion_len_mean_frames": 6.0, # mean run length (geometric)
    "fps_assumed": 60.0,              # only used to scale occlusion_rate
}


def retarget_21_to_h36m17(joints21: np.ndarray) -> np.ndarray:
    """(T, 21, 3) pipeline skeleton -> (T, 17, 3) H36M skeleton (pure remap)."""
    joints21 = np.asarray(joints21, dtype=np.float32)
    if joints21.ndim != 3 or joints21.shape[1] != len(PIPELINE_JOINT_NAMES) or joints21.shape[2] != 3:
        raise ValueError(
            f"expected (T, {len(PIPELINE_JOINT_NAMES)}, 3), got {joints21.shape}"
        )
    return joints21[:, _SRC_IDX, :].copy()


def _per_joint_sigma(cfg: Mapping) -> np.ndarray:
    base = float(cfg["jitter_sigma_m"])
    mult = float(cfg["end_effector_sigma_mult"])
    sigma = np.full(len(H36M17_JOINT_NAMES), base, dtype=np.float32)
    for j, name in enumerate(H36M17_JOINT_NAMES):
        if name in _END_EFFECTORS:
            sigma[j] = base * mult
    return sigma


def _apply_jitter(joints: np.ndarray, cfg: Mapping, rng: np.random.Generator) -> np.ndarray:
    sigma = _per_joint_sigma(cfg)  # (J,)
    noise = rng.normal(0.0, 1.0, size=joints.shape).astype(np.float32) * sigma[None, :, None]
    return joints + noise


def _apply_temporal_smooth(joints: np.ndarray, cfg: Mapping) -> np.ndarray:
    """Causal EMA over time — mimics the temporal smoothing of a 3D lifter."""
    alpha = float(cfg["temporal_smooth_alpha"])
    if alpha <= 0.0:
        return joints
    out = np.empty_like(joints)
    out[0] = joints[0]
    for t in range(1, joints.shape[0]):
        out[t] = alpha * joints[t] + (1.0 - alpha) * out[t - 1]
    return out


def _apply_occlusions(joints: np.ndarray, cfg: Mapping, rng: np.random.Generator) -> np.ndarray:
    """Drop random per-joint runs and linearly interpolate across the gap."""
    rate = float(cfg["occlusion_rate_per_joint"])
    if rate <= 0.0:
        return joints
    T, J = joints.shape[0], joints.shape[1]
    fps = float(cfg["fps_assumed"])
    mean_len = max(1.0, float(cfg["occlusion_len_mean_frames"]))
    p_start = rate / fps  # per-frame start probability
    out = joints.copy()
    for j in range(J):
        t = 0
        while t < T:
            if rng.random() < p_start:
                run = int(rng.geometric(1.0 / mean_len))
                run = max(1, min(run, T - t))
                lo, hi = t - 1, t + run  # anchors either side of the gap
                if lo >= 0 and hi < T:
                    for k in range(run):
                        w = (k + 1) / (run + 1)
                        out[t + k, j] = (1.0 - w) * joints[lo, j] + w * joints[hi, j]
                # gaps at the very start/end keep the (jittered) value
                t += run
            else:
                t += 1
    return out


def degrade_skeleton(joints21: np.ndarray, config: Mapping | None = None) -> np.ndarray:
    """Full estimated-skeleton transform: retarget then inject estimation noise.

    Deterministic given ``config['seed']``. Returns (T, 17, 3) float32.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    rng = np.random.default_rng(int(cfg["seed"]))

    est = retarget_21_to_h36m17(joints21)
    est = _apply_jitter(est, cfg, rng)
    est = _apply_temporal_smooth(est, cfg)
    est = _apply_occlusions(est, cfg, rng)
    return np.ascontiguousarray(est, dtype=np.float32)
