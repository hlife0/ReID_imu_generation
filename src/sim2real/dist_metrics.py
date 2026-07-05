"""L1 distribution distances between real and synthetic window sets — M2 (implemented).

All metrics operate on frozen-encoder embeddings of the SAME split's windows
(real vs one synthetic source). Embeddings are standardized with the REAL
side's mean/std first, so scales are comparable across generators.

- ``frechet_distance``: Gaussian-approximation Fréchet distance ("IMU FID"),
  numpy-only via symmetric eigendecomposition.
- ``mmd_rbf``: unbiased RBF-kernel MMD^2, median-heuristic bandwidth,
  seeded subsampling for tractability.
- ``c2st_auc``: classifier two-sample test — small numpy logistic regression,
  held-out AUC. ~0.5 = indistinguishable in feature space, ~1.0 = trivially
  separable.
"""

from __future__ import annotations

import numpy as np


def standardize_by_reference(reference: np.ndarray, other: np.ndarray) -> tuple:
    mean = reference.mean(axis=0, keepdims=True)
    std = reference.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (reference - mean) / std, (other - mean) / std


def frechet_distance(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> float:
    mu_a, mu_b = embeddings_a.mean(axis=0), embeddings_b.mean(axis=0)
    cov_a = np.cov(embeddings_a, rowvar=False)
    cov_b = np.cov(embeddings_b, rowvar=False)

    # sqrtm(cov_a) via symmetric eigendecomposition
    vals_a, vecs_a = np.linalg.eigh(cov_a)
    sqrt_a = (vecs_a * np.sqrt(np.clip(vals_a, 0, None))) @ vecs_a.T
    inner = sqrt_a @ cov_b @ sqrt_a
    vals_i = np.linalg.eigvalsh(inner)
    tr_sqrt = np.sqrt(np.clip(vals_i, 0, None)).sum()

    diff = mu_a - mu_b
    return float(diff @ diff + np.trace(cov_a) + np.trace(cov_b) - 2.0 * tr_sqrt)


def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)


def mmd_rbf(embeddings_a: np.ndarray, embeddings_b: np.ndarray,
            max_samples: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    a = embeddings_a[rng.choice(len(embeddings_a), min(max_samples, len(embeddings_a)), replace=False)]
    b = embeddings_b[rng.choice(len(embeddings_b), min(max_samples, len(embeddings_b)), replace=False)]

    d_ab = _pairwise_sq_dists(a, b)
    median = float(np.median(d_ab))
    gamma = 1.0 / max(median, 1e-12)

    k_aa = np.exp(-gamma * _pairwise_sq_dists(a, a))
    k_bb = np.exp(-gamma * _pairwise_sq_dists(b, b))
    k_ab = np.exp(-gamma * d_ab)

    n, m = len(a), len(b)
    np.fill_diagonal(k_aa, 0.0)
    np.fill_diagonal(k_bb, 0.0)
    mmd2 = k_aa.sum() / (n * (n - 1)) + k_bb.sum() / (m * (m - 1)) - 2.0 * k_ab.mean()
    return float(mmd2)


def c2st_auc(embeddings_a: np.ndarray, embeddings_b: np.ndarray,
             seed: int = 0, train_ratio: float = 0.7,
             iterations: int = 500, lr: float = 0.5, l2: float = 1e-3) -> float:
    """Held-out AUC of a logistic regression separating A (label 0) from B (label 1)."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([embeddings_a, embeddings_b], axis=0)
    y = np.concatenate([np.zeros(len(embeddings_a)), np.ones(len(embeddings_b))])

    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0.0] = 1.0
    x = (x - mean) / std

    order = rng.permutation(len(x))
    x, y = x[order], y[order]
    cut = int(len(x) * train_ratio)
    x_tr, y_tr, x_te, y_te = x[:cut], y[:cut], x[cut:], y[cut:]

    w = np.zeros(x.shape[1])
    b = 0.0
    for _ in range(iterations):
        z = x_tr @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad_w = x_tr.T @ (p - y_tr) / len(x_tr) + l2 * w
        grad_b = float((p - y_tr).mean())
        w -= lr * grad_w
        b -= lr * grad_b

    scores = x_te @ w + b
    pos, neg = scores[y_te == 1], scores[y_te == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # rank-based AUC (Mann-Whitney U)
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    return auc
