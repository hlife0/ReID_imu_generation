"""Probe retrieval evaluation — M3 (implemented). Torch; venv only."""

from __future__ import annotations

import numpy as np
import torch

EMBED_BATCH = 1024


@torch.no_grad()
def _embed_all(tower, data: torch.Tensor) -> torch.Tensor:
    tower.eval()
    chunks = [tower(data[i : i + EMBED_BATCH]) for i in range(0, len(data), EMBED_BATCH)]
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def retrieval_r_at_k(imu_tower, motion_tower, imu: torch.Tensor, motion: torch.Tensor,
                     ks: tuple = (1, 5)) -> dict:
    """Full-gallery IMU->motion retrieval. Returns {k: R@k}."""
    emb_i = _embed_all(imu_tower, imu)
    emb_m = _embed_all(motion_tower, motion)
    sim = emb_i @ emb_m.T
    target = torch.arange(len(emb_i), device=sim.device)
    result = {}
    max_k = max(ks)
    topk = sim.topk(max_k, dim=1).indices
    for k in ks:
        hits = (topk[:, :k] == target[:, None]).any(dim=1)
        result[k] = float(hits.float().mean())
    return result


@torch.no_grad()
def retrieval_metrics(imu_tower, motion_tower, imu: torch.Tensor, motion: torch.Tensor,
                      sequences: np.ndarray) -> dict:
    """R@1/R@5 plus per-sequence breakdown and chance level."""
    emb_i = _embed_all(imu_tower, imu)
    emb_m = _embed_all(motion_tower, motion)
    sim = emb_i @ emb_m.T
    target = torch.arange(len(emb_i), device=sim.device)
    topk = sim.topk(5, dim=1).indices
    hit1 = (topk[:, :1] == target[:, None]).any(dim=1).cpu().numpy()
    hit5 = (topk == target[:, None]).any(dim=1).cpu().numpy()

    per_sequence = {}
    for seq in sorted(set(sequences.tolist())):
        mask = sequences == seq
        per_sequence[seq] = {"r1": round(float(hit1[mask].mean()), 4),
                             "windows": int(mask.sum())}

    gallery = len(emb_i)
    return {
        "r_at_1": round(float(hit1.mean()), 4),
        "r_at_5": round(float(hit5.mean()), 4),
        "gallery_size": gallery,
        "chance_r_at_1": round(1.0 / gallery, 6),
        "per_sequence": per_sequence,
    }
