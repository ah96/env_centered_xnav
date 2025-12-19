#!/usr/bin/env python3
"""
plot_results.py

Loads your provided CSVs:
  - metrics_aff_remove.csv
  - metrics_aff_move.csv
  - transfer_matrix_aff_remove.csv
  - transfer_matrix_aff_move.csv

and writes figures into ./figures/

Figures produced (PDF):
  - success_at_k_combined.pdf
  - success_at_k_remove.pdf
  - success_at_k_move.pdf
  - agreement_heatmap_remove.pdf Holden? (only if pairwise rows exist)
  - agreement_heatmap_move.pdf
  - minimality_gap_remove.pdf
  - minimality_gap_move.pdf
  - robustness_remove.pdf
  - robustness_move.pdf
  - cross_planner_transfer_remove.pdf
  - cross_planner_transfer_move.pdf

Notes:
- Uses matplotlib only (no seaborn).
- If some metrics are missing (e.g., oracle gaps), it will warn and skip that plot.

python plot_results.py \
  --metrics_remove /path/to/metrics_aff_remove.csv \
  --metrics_move /path/to/metrics_aff_move.csv \
  --transfer_remove /path/to/transfer_matrix_aff_remove.csv \
  --transfer_move /path/to/transfer_matrix_aff_move.csv
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------- helpers -----------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def _extract_k(metric_str: str) -> int:
    # metric like "success_at_3"
    return int(str(metric_str).replace("success_at_", ""))

def _nice_mode_title(mode: str) -> str:
    return "Remove" if mode == "remove" else "Move"


# ----------------- plotting: Success@k -----------------

def plot_success_at_k(metrics: pd.DataFrame, out_dir: str, mode: str) -> None:
    df = metrics[metrics["metric"].astype(str).str.startswith("success_at_")].copy()
    if df.empty:
        print(f"[WARN] No success_at_k rows found for mode={mode}; skipping.")
        return

    df["k"] = df["metric"].apply(_extract_k)
    agg = df.groupby(["method", "k"], as_index=False)["value"].mean().sort_values(["method", "k"])

    plt.figure()
    for method, sub in agg.groupby("method"):
        # keep only core methods and baselines if present
        sub = sub.sort_values("k")
        plt.plot(sub["k"], sub["value"], marker="o", label=str(method))

    plt.xlabel("k (top-k entities intervened on)")
    plt.ylabel("Success@k")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.title(f"Attribution faithfulness (mode: {_nice_mode_title(mode)})")
    plt.legend()
    savefig(os.path.join(out_dir, f"success_at_k_{mode}.pdf"))
    print(f"[OK] Wrote success_at_k_{mode}.pdf")


def plot_success_at_k_combined(metrics_remove: pd.DataFrame, metrics_move: pd.DataFrame, out_dir: str) -> None:
    def prep(df: pd.DataFrame, mode: str) -> pd.DataFrame:
        d = df[df["metric"].astype(str).str.startswith("success_at_")].copy()
        if d.empty:
            return d
        d["k"] = d["metric"].apply(_extract_k)
        d["mode"] = mode
        return d

    a = prep(metrics_remove, "remove")
    b = prep(metrics_move, "move")
    df = pd.concat([a, b], ignore_index=True)

    if df.empty:
        print("[WARN] No success_at_k rows in either remove or move; skipping combined plot.")
        return

    # only show lime/shap by default (cleaner for paper)
    df = df[df["method"].isin(["lime", "shap"])].copy()
    if df.empty:
        print("[WARN] Combined plot: expected methods lime/shap not found; plotting all methods.")
        df = pd.concat([a, b], ignore_index=True)

    agg = df.groupby(["mode", "method", "k"], as_index=False)["value"].mean().sort_values(["mode", "method", "k"])

    plt.figure()
    for (mode, method), sub in agg.groupby(["mode", "method"]):
        sub = sub.sort_values("k")
        label = f"{method} ({mode})"
        plt.plot(sub["k"], sub["value"], marker="o", label=label)

    plt.xlabel("k (top-k entities intervened on)")
    plt.ylabel("Success@k")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.title("Success@k comparison (remove vs move)")
    plt.legend()
    savefig(os.path.join(out_dir, "success_at_k_combined.pdf"))
    print("[OK] Wrote success_at_k_combined.pdf")


# ----------------- plotting: Agreement heatmap -----------------

def plot_agreement_heatmap(metrics: pd.DataFrame, out_dir: str, mode: str, k: int = 5) -> None:
    """
    Uses rows where metric == jaccard_top_k and method is like "A_vs_B".
    If you only have lime_vs_shap, it will make a 2x2 heatmap.
    """
    df = metrics[metrics["metric"].astype(str) == f"jaccard_top_{k}"].copy()
    if df.empty:
        # fallback: pick most common k
        df2 = metrics[metrics["metric"].astype(str).str.startswith("jaccard_top_")].copy()
        if df2.empty:
            print(f"[WARN] No jaccard_top_k found for mode={mode}; skipping agreement heatmap.")
            return
        df2["k"] = df2["metric"].astype(str).str.replace("jaccard_top_", "", regex=False).astype(int)
        k = int(df2["k"].mode().iloc[0])
        df = df2[df2["k"] == k].copy()

    pairs = []
    for m in df["method"].astype(str).unique():
        if "_vs_" in m:
            a, b = m.split("_vs_", 1)
            pairs.append((a, b))
    pairs = sorted(set(pairs))

    if not pairs:
        print(f"[WARN] No A_vs_B method pairs for agreement (mode={mode}); skipping.")
        return

    methods = sorted(set([a for a, _ in pairs] + [b for _, b in pairs]))
    idx = {m: i for i, m in enumerate(methods)}
    M = np.full((len(methods), len(methods)), np.nan, dtype=float)
    np.fill_diagonal(M, 1.0)

    agg = df.groupby("method", as_index=False)["value"].mean()
    for _, row in agg.iterrows():
        name = str(row["method"])
        if "_vs_" not in name:
            continue
        a, b = name.split("_vs_", 1)
        i, j = idx[a], idx[b]
        M[i, j] = float(row["value"])
        M[j, i] = float(row["value"])

    plt.figure()
    im = plt.imshow(M, interpolation="nearest")
    plt.colorbar(im)
    plt.xticks(range(len(methods)), methods, rotation=45, ha="right")
    plt.yticks(range(len(methods)), methods)
    plt.title(f"Agreement (Jaccard top-{k}) — mode: {_nice_mode_title(mode)}")

    for i in range(len(methods)):
        for j in range(len(methods)):
            if np.isfinite(M[i, j]):
                plt.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center")

    savefig(os.path.join(out_dir, f"agreement_heatmap_{mode}.pdf"))
    print(f"[OK] Wrote agreement_heatmap_{mode}.pdf")


# ----------------- plotting: Minimality gap -----------------

def plot_minimality_gap(metrics: pd.DataFrame, out_dir: str, mode: str) -> None:
    df = metrics[(metrics["method"].astype(str) == "cose") &
                 (metrics["metric"].astype(str) == "min_gap_oracle")].copy()
    if df.empty:
        print(f"[WARN] No min_gap_oracle rows for mode={mode}; skipping minimality gap plot.")
        return

    vals = df["value"].astype(float).to_numpy()

    plt.figure()
    plt.boxplot(vals, vert=True, showmeans=True)
    plt.ylabel(r"$|O_c| - |O_c^{oracle}|$")
    plt.title(f"COSE minimality gap vs oracle — mode: {_nice_mode_title(mode)}")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    savefig(os.path.join(out_dir, f"minimality_gap_{mode}.pdf"))
    print(f"[OK] Wrote minimality_gap_{mode}.pdf")


# ----------------- plotting: Robustness -----------------

def plot_robustness(metrics: pd.DataFrame, out_dir: str, mode: str) -> None:
    """
    Produces a bar plot for kendall_tau_jitter for lime/shap if present.
    If you later add more robustness metrics, you can extend here.
    """
    df = metrics[metrics["metric"].astype(str) == "kendall_tau_jitter"].copy()
    df = df[df["method"].isin(["lime", "shap"])].copy()

    if df.empty:
        print(f"[WARN] No kendall_tau_jitter rows for mode={mode}; skipping robustness plot.")
        return

    agg = df.groupby("method", as_index=False)["value"].mean().sort_values("method")
    labels = agg["method"].astype(str).tolist()
    vals = agg["value"].astype(float).tolist()

    x = np.arange(len(vals))
    plt.figure()
    plt.bar(x, vals)
    plt.xticks(x, labels)
    plt.ylim(-1.0, 1.0)
    plt.ylabel("Kendall's τ (ranking stability)")
    plt.title(f"Robustness under jitter — mode: {_nice_mode_title(mode)}")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    savefig(os.path.join(out_dir, f"robustness_{mode}.pdf"))
    print(f"[OK] Wrote robustness_{mode}.pdf")


# ----------------- plotting: Cross-planner transfer -----------------

def plot_cross_planner_transfer(transfer: pd.DataFrame, out_dir: str, mode: str) -> None:
    df = transfer[transfer["metric"].astype(str) == "cose_sufficiency"].copy()
    if df.empty:
        print(f"[WARN] No cose_sufficiency rows for mode={mode}; skipping transfer heatmap.")
        return

    agg = df.groupby(["source_planner", "eval_planner"], as_index=False)["value"].mean()
    src = sorted(agg["source_planner"].astype(str).unique())
    evl = sorted(agg["eval_planner"].astype(str).unique())

    mat = np.full((len(src), len(evl)), np.nan, dtype=float)
    src_i = {s: i for i, s in enumerate(src)}
    evl_i = {e: i for i, e in enumerate(evl)}

    for _, r in agg.iterrows():
        i = src_i[str(r["source_planner"])]
        j = evl_i[str(r["eval_planner"])]
        mat[i, j] = float(r["value"])

    plt.figure()
    im = plt.imshow(mat, interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.colorbar(im)
    plt.xticks(range(len(evl)), evl, rotation=45, ha="right")
    plt.yticks(range(len(src)), src)
    plt.xlabel("Evaluation planner")
    plt.ylabel("Source planner")
    plt.title(f"Cross-planner transfer (COSE sufficiency) — mode: {_nice_mode_title(mode)}")

    for i in range(len(src)):
        for j in range(len(evl)):
            if np.isfinite(mat[i, j]):
                plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center")

    savefig(os.path.join(out_dir, f"cross_planner_transfer_{mode}.pdf"))
    print(f"[OK] Wrote cross_planner_transfer_{mode}.pdf")


# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_remove", default="results/metrics_aff_remove.csv")
    ap.add_argument("--metrics_move", default="results/metrics_aff_move.csv")
    ap.add_argument("--transfer_remove", default="results/transfer_matrix_aff_remove.csv")
    ap.add_argument("--transfer_move", default="results/transfer_matrix_aff_move.csv")
    ap.add_argument("--figures_dir", default="figures")
    ap.add_argument("--agreement_k", type=int, default=5)
    args = ap.parse_args()

    ensure_dir(args.figures_dir)

    metrics_remove = read_csv(args.metrics_remove)
    metrics_move = read_csv(args.metrics_move)
    transfer_remove = read_csv(args.transfer_remove)
    transfer_move = read_csv(args.transfer_move)

    # Combined + separate Success@k
    plot_success_at_k_combined(metrics_remove, metrics_move, args.figures_dir)
    plot_success_at_k(metrics_remove, args.figures_dir, mode="remove")
    plot_success_at_k(metrics_move, args.figures_dir, mode="move")

    # Agreement heatmaps (if available)
    plot_agreement_heatmap(metrics_remove, args.figures_dir, mode="remove", k=args.agreement_k)
    plot_agreement_heatmap(metrics_move, args.figures_dir, mode="move", k=args.agreement_k)

    # Minimality gap plots (if available)
    plot_minimality_gap(metrics_remove, args.figures_dir, mode="remove")
    plot_minimality_gap(metrics_move, args.figures_dir, mode="move")

    # Robustness plots
    plot_robustness(metrics_remove, args.figures_dir, mode="remove")
    plot_robustness(metrics_move, args.figures_dir, mode="move")

    # Transfer heatmaps
    plot_cross_planner_transfer(transfer_remove, args.figures_dir, mode="remove")
    plot_cross_planner_transfer(transfer_move, args.figures_dir, mode="move")

    print(f"[DONE] Figures written to: {args.figures_dir}/")


if __name__ == "__main__":
    main()
