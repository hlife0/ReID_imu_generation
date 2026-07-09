"""Probe training. Runs in the external venv (torch + GPU).

Symmetric in-batch InfoNCE; early stopping on val R@1 (val is always real).
Fixed budget across all matrix cells. ``shuffle_pairs`` is the leakage
tripwire: it permutes the imu<->motion pairing in the TRAIN set only and must
drive test retrieval back to chance.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import build_probe
from .retrieve import retrieval_r_at_k

TEMPERATURE = 0.07
BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE = 8


def train_probe(
    train_imu: np.ndarray,      # (N, T, C), already normalized
    train_motion: np.ndarray,   # (N, T, J, 3)
    val_imu: np.ndarray,
    val_motion: np.ndarray,
    seed: int,
    device: str = "cuda",
    shuffle_pairs: bool = False,
    lr: float = 1e-3,
    max_epochs: int = MAX_EPOCHS,
    init_state: dict | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    imu_tower, motion_tower = build_probe(train_imu.shape[2], train_motion.shape[2])
    if init_state is not None:
        imu_tower.load_state_dict(init_state["imu_tower"])
        motion_tower.load_state_dict(init_state["motion_tower"])
    imu_tower.to(dev)
    motion_tower.to(dev)

    x_imu = torch.tensor(train_imu, dtype=torch.float32, device=dev)
    x_mot = torch.tensor(train_motion, dtype=torch.float32, device=dev)
    if shuffle_pairs:
        perm = torch.randperm(len(x_mot), generator=torch.Generator().manual_seed(seed + 999))
        x_mot = x_mot[perm.to(dev)]
    v_imu = torch.tensor(val_imu, dtype=torch.float32, device=dev)
    v_mot = torch.tensor(val_motion, dtype=torch.float32, device=dev)

    params = list(imu_tower.parameters()) + list(motion_tower.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    best = {"val_r1": -1.0, "epoch": -1, "state": None}
    history = []
    n = len(x_imu)

    for epoch in range(max_epochs):
        imu_tower.train()
        motion_tower.train()
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 10007 + epoch))
        epoch_loss, batches = 0.0, 0
        for i in range(0, n - 1, BATCH_SIZE):
            idx = order[i : i + BATCH_SIZE].to(dev)
            if len(idx) < 8:
                continue
            emb_i = imu_tower(x_imu[idx])
            emb_m = motion_tower(x_mot[idx])
            logits = emb_i @ emb_m.T / TEMPERATURE
            target = torch.arange(len(idx), device=dev)
            loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            batches += 1

        val_r = retrieval_r_at_k(imu_tower, motion_tower, v_imu, v_mot, ks=(1,))
        history.append({"epoch": epoch, "loss": epoch_loss / max(batches, 1), "val_r1": val_r[1]})

        if val_r[1] > best["val_r1"]:
            best = {
                "val_r1": val_r[1],
                "epoch": epoch,
                "state": {
                    "imu_tower": {k: v.detach().cpu().clone() for k, v in imu_tower.state_dict().items()},
                    "motion_tower": {k: v.detach().cpu().clone() for k, v in motion_tower.state_dict().items()},
                },
            }
        elif epoch - best["epoch"] >= PATIENCE:
            break

    imu_tower.load_state_dict(best["state"]["imu_tower"])
    motion_tower.load_state_dict(best["state"]["motion_tower"])
    return {
        "imu_tower": imu_tower,
        "motion_tower": motion_tower,
        "state": best["state"],
        "best_val_r1": best["val_r1"],
        "best_epoch": best["epoch"],
        "epochs_ran": len(history),
        "history": history,
    }
