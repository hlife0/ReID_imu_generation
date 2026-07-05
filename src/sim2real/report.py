"""Results aggregation and reporting — M4 (implemented).

Reads the append-only ledger (one JSON row per completed cell) and produces
the headline tables: TRTR anchor, TSTR generator ranking with sim-to-real
gaps, mix (augmentation) comparison, the sim-boost curve over the real-data
budget, and the shuffled-pairs control. Aggregation is mean +- std over
seeds. The ledger is the single source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_ledger(results_jsonl: Path) -> list:
    rows = []
    for line in Path(results_jsonl).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _group_key(row: dict) -> tuple:
    return (
        row["protocol"],
        tuple(row["train"]),
        tuple(row["pretrain"] or []),
        row["real_fraction"],
    )


def aggregate(rows: list) -> dict:
    """Group non-control rows by cell config; mean+-std over seeds."""
    groups = {}
    for row in rows:
        if row.get("shuffle_pairs"):
            continue
        groups.setdefault(_group_key(row), []).append(row)

    out = {}
    for key, members in groups.items():
        r1 = np.array([m["r_at_1"] for m in members], dtype=float)
        r5 = np.array([m["r_at_5"] for m in members], dtype=float)
        out[key] = {
            "protocol": key[0], "train": list(key[1]), "pretrain": list(key[2]),
            "real_fraction": key[3], "seeds": sorted(m["seed"] for m in members),
            "r1_mean": float(r1.mean()), "r1_std": float(r1.std()),
            "r5_mean": float(r5.mean()), "r5_std": float(r5.std()),
            "chance": members[0]["chance_r_at_1"], "gallery": members[0]["gallery_size"],
        }
    return out


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def _token_label(tokens: list) -> str:
    return "+".join(tokens)


def render_report_md(rows: list, benchmark_id: str) -> str:
    agg = aggregate(rows)
    controls = [r for r in rows if r.get("shuffle_pairs")]

    def find(protocol, train=None, pretrain=(), rf=1.0):
        for key, g in agg.items():
            if g["protocol"] != protocol or g["real_fraction"] != rf:
                continue
            if train is not None and tuple(g["train"]) != tuple(train):
                continue
            if tuple(g["pretrain"]) != tuple(pretrain):
                continue
            yield g

    trtr = next(find("trtr", train=["real"]), None)
    lines = [f"# sim2real L2 Report — {benchmark_id}", ""]
    if trtr:
        lines += [
            f"Testbed: IMU→motion retrieval, 24-frame windows, subject-disjoint split, "
            f"test gallery {trtr['gallery']} real windows (chance R@1 = {trtr['chance']:.4f}).",
            "",
            "## Anchors",
            "",
            f"- **TRTR (train real, test real): R@1 = {_fmt(trtr['r1_mean'], trtr['r1_std'])}, "
            f"R@5 = {_fmt(trtr['r5_mean'], trtr['r5_std'])}** — "
            f"{trtr['r1_mean'] / trtr['chance']:.0f}× above chance.",
        ]
    for c in controls:
        lines.append(f"- Shuffled-pairs control: R@1 = {c['r_at_1']} (chance {c['chance_r_at_1']:.4f}) — "
                     + ("**at chance, no leakage** ✅" if c["r_at_1"] <= 3 * c["chance_r_at_1"] else "**ABOVE CHANCE — investigate leakage!** ⚠️"))
    lines.append("")

    tstr_rows = sorted(find("tstr"), key=lambda g: -g["r1_mean"])
    if tstr_rows:
        lines += ["## TSTR generator ranking (train synthetic, test real)", "",
                  "| generator (train source) | R@1 | R@5 | sim-to-real gap (TRTR−TSTR R@1) |",
                  "|---|---|---|---|"]
        for g in tstr_rows:
            gap = f"{trtr['r1_mean'] - g['r1_mean']:.4f}" if trtr else "—"
            lines.append(f"| {_token_label(g['train'])} | {_fmt(g['r1_mean'], g['r1_std'])} "
                         f"| {_fmt(g['r5_mean'], g['r5_std'])} | {gap} |")
        lines.append("")

    mix_rows = sorted(find("mix"), key=lambda g: -g["r1_mean"])
    if mix_rows:
        lines += ["## Mix (real + synthetic as augmentation)", "",
                  "| train composition | R@1 | R@5 | Δ vs TRTR |",
                  "|---|---|---|---|"]
        for g in mix_rows:
            delta = f"{g['r1_mean'] - trtr['r1_mean']:+.4f}" if trtr else "—"
            lines.append(f"| {_token_label(g['train'])} | {_fmt(g['r1_mean'], g['r1_std'])} "
                         f"| {_fmt(g['r5_mean'], g['r5_std'])} | {delta} |")
        lines.append("")

    fractions = sorted({g["real_fraction"] for g in agg.values()})
    pf_pretrains = sorted({tuple(g["pretrain"]) for g in agg.values() if g["protocol"] == "pretrain_finetune"})
    if pf_pretrains:
        lines += ["## Sim-boost curve (synthetic pretraining vs real-data budget)", "",
                  "| real fraction | real-only R@1 | " +
                  " | ".join(f"pretrain {_token_label(list(p))} R@1 (boost)" for p in pf_pretrains) + " |",
                  "|---|---|" + "---|" * len(pf_pretrains)]
        for rf in fractions:
            base = next(find("trtr", train=["real"], rf=rf), None)
            cells = [f"{_fmt(base['r1_mean'], base['r1_std'])}" if base else "—"]
            for p in pf_pretrains:
                g = next(find("pretrain_finetune", train=["real"], pretrain=p, rf=rf), None)
                if g and base:
                    cells.append(f"{_fmt(g['r1_mean'], g['r1_std'])} ({g['r1_mean'] - base['r1_mean']:+.4f})")
                elif g:
                    cells.append(_fmt(g["r1_mean"], g["r1_std"]))
                else:
                    cells.append("—")
            lines.append(f"| {rf:g} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## Notes", "",
              "- All values mean ± std over seeds; ledger: `results.jsonl` (single source of truth).",
              "- Normalization: channel stats from the training source, applied unchanged to val/test (honest TSTR).",
              "- val = S4 (hyperparameter surface), test = S5 (touched once per cell)."]
    return "\n".join(lines) + "\n"


def write_report(results_jsonl: Path, out_path: Path, benchmark_id: str) -> Path:
    rows = read_ledger(results_jsonl)
    out_path = Path(out_path)
    out_path.write_text(render_report_md(rows, benchmark_id), encoding="utf-8")
    return out_path
