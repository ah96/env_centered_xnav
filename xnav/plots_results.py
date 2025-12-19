#!/usr/bin/env python3
"""
plot_results.py

Generates the exact figures referenced in the paper from:
  - results/metrics.csv
  - results/transfer_matrix.csv

Outputs PDFs into: figures/
  - success_at_k.pdf
  - agreement_heatmap.pdf
  - minimality_gap.pdf
  - robustness.pdf
  - cross_planner_transfer.pdf

Notes:
- Uses matplotlib only (no seaborn).
- If some metrics are missing (e.g., no oracle gaps), the corresponding figure is still produced
  with a clear warning and will contain whatever is available.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _savefig(out_path: str) -> None:
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_success_at_k(metrics: pd.DataFrame, out_dir: str, filename: str = "success_at_k.pdf") -> None:
    df = metrics[metrics["metric"].astype(str).str.startswith("success_at_")].copy()
    if df.empty:
        print("[WARN] No success_at_k rows found; skipping success_at_k plot.")
        return

    # Extract k
    df["k"] = df["metric"].astype(str).str.replace("success_at_", "", regex=False).astype(int)

    # Average across env/planners
    agg = (
        df.groupby(["method", "k"], as_index=False)["value"]
        .mean()
        .sort_values(["method", "k"])
    )

    plt.figure()
    for method, sub in agg.groupby("method"):
        sub = sub.sort_values("k")
        plt.plot(sub["k"], sub["value"], marker="o", label=method)

    plt.xlabel("k (top-k entities intervened on)")
    plt.ylabel("Success@k")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    _savefig(os.path.join(out_dir, filename))
    print(f"[OK] Wrote {filename}")


def plot_agreement_heatmap(
    metrics: pd.DataFrame,
    out_dir: str,
    filename: str = "agreement_heatmap.pdf",
    k: int = 5,
) -> None:
    """
    Agreement heatmap is ideally pairwise Jaccard(top-k) for multiple methods.
    The provided runner stores only 'lime_vs_shap' Jaccard. This function:
      - If full pairwise rows exist in metrics (method like "A_vs_B"), uses them.
      - Else, falls back to a minimal 2x2 heatmap for lime vs shap if available.
    """
    df = metrics[metrics["metric"].astype(str) == f"jaccard_top_{k}"].copy()
    if df.empty:
        # Try any jaccard_top_* available
        df2 = metrics[metrics["metric"].astype(str).str.startswith("jaccard_top_")].copy()
        if df2.empty:
            print("[WARN] No jaccard_top_k rows found; skipping agreement heatmap.")
            return
        # pick the most common k
        df2["k"] = df2["metric"].astype(str).str.replace("jaccard_top_", "", regex=False).astype(int)
        k_mode = int(df2["k"].mode().iloc[0])
        k = k_mode
        df = df2[df2["k"] == k_mode].copy()

    # Parse "method" column expected like "lime_vs_shap"
    pairs = []
    for m in df["method"].astype(str).unique():
        if "_vs_" in m:
            a, b = m.split("_vs_", 1)
            pairs.append((a, b))
    pairs = sorted(set(pairs))

    if not pairs:
        print("[WARN] No pairwise 'A_vs_B' methods found; skipping agreement heatmap.")
        return

    # Determine methods involved
    methods = sorted(set([a for a, _ in pairs] + [b for _, b in pairs]))
    idx = {m: i for i, m in enumerate(methods)}

    # Build matrix with diagonal = 1
    M = np.full((len(methods), len(methods)), np.nan, dtype=float)
    np.fill_diagonal(M, 1.0)

    # Average across env/planners for each pair
    agg = df.groupby("method", as_index=False)["value"].mean()
    for _, row in agg.iterrows():
        m = str(row["method"])
        if "_vs_" not in m:
            continue
        a, b = m.split("_vs_", 1)
        i, j = idx[a], idx[b]
        M[i, j] = float(row["value"])
        M[j, i] = float(row["value"])

    plt.figure()
    im = plt.imshow(M, interpolation="nearest")
    plt.colorbar(im)
    plt.xticks(range(len(methods)), methods, rotation=45, ha="right")
    plt.yticks(range(len(methods)), methods)
    plt.title(f"Jaccard similarity of top-{k} sets")

    # Annotate
    for i in range(len(methods)):
        for j in range(len(methods)):
            if np.isfinite(M[i, j]):
                plt.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center")

    _savefig(os.path.join(out_dir, filename))
    print(f"[OK] Wrote {filename}")


def plot_minimality_gap(metrics: pd.DataFrame, out_dir: str, filename: str = "minimality_gap.pdf") -> None:
    df = metrics[(metrics["method"].astype(str) == "cose") & (metrics["metric"].astype(str) == "min_gap_oracle")].copy()
    if df.empty:
        print("[WARN] No cose min_gap_oracle rows found; skipping minimality gap plot.")
        return

    # Aggregate across env/planners by (H, density) optionally
    # For plotting, show distribution overall as boxplot.
    vals = df["value"].astype(float).to_numpy()

    plt.figure()
    plt.boxplot(vals, vert=True, showmeans=True)
    plt.ylabel(r"$|O_c| - |O_c^{oracle}|$")
    plt.title("COSE minimality gap vs oracle (smaller is better)")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    _savefig(os.path.join(out_dir, filename))
    print(f"[OK] Wrote {filename}")


def plot_robustness(metrics: pd.DataFrame, out_dir: str, filename: str = "robustness.pdf") -> None:
    """
    Produces a grouped bar plot:
      - Kendall tau for attribution methods (lime, shap) under jitter
      - Jaccard for counterfactual sets if present (optional), otherwise omitted
    """
    # Kendall tau
    tau = metrics[metrics["metric"].astype(str) == "kendall_tau_jitter"].copy()
    tau = tau[tau["method"].isin(["lime", "shap"])].copy()

    # Optional: set stability for COSE (not stored by default)
    # If you later add it, name it "jaccard_cose_jitter"
    cose_j = metrics[metrics["metric"].astype(str).isin(["jaccard_cose_jitter", "jaccard_cose_distractor"])].copy()

    if tau.empty and cose_j.empty:
        print("[WARN] No robustness rows found; skipping robustness plot.")
        return

    # Aggregate
    bars = []
    labels = []

    if not tau.empty:
        tau_agg = tau.groupby("method", as_index=False)["value"].mean()
        for _, r in tau_agg.iterrows():
            labels.append(f"{r['method']} (Kendall τ)")
            bars.append(float(r["value"]))

    # If COSE stability exists
    if not cose_j.empty:
        cj_agg = cose_j.groupby("metric", as_index=False)["value"].mean()
        for _, r in cj_agg.iterrows():
            labels.append(str(r["metric"]))
            bars.append(float(r["value"]))

    x = np.arange(len(bars))

    plt.figure()
    plt.bar(x, bars)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(-1.0, 1.0)  # Kendall τ range; Jaccard will still fit in [0,1]
    plt.ylabel("Stability")
    plt.title("Robustness under environment perturbations")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    _savefig(os.path.join(out_dir, filename))
    print(f"[OK] Wrote {filename}")


def plot_cross_planner_transfer(transfer: pd.DataFrame, out_dir: str, filename: str = "cross_planner_transfer.pdf") -> None:
    df = transfer[transfer["metric"].astype(str) == "cose_sufficiency"].copy()
    if df.empty:
        print("[WARN] No cose_sufficiency rows found; skipping cross-planner transfer plot.")
        return

    # Average across env configs
    agg = df.groupby(["source_planner", "eval_planner"], as_index=False)["value"].mean()
    src = sorted(agg["source_planner"].unique())
    evl = sorted(agg["eval_planner"].unique())

    mat = np.full((len(src), len(evl)), np.nan, dtype=float)
    src_i = {s: i for i, s in enumerate(src)}
    evl_i = {e: i for i, e in enumerate(evl)}

    for _, r in agg.iterrows():
        i = src_i[str(r["source_planner"])]
        j = evl_i[str(r["eval_planner"])]
        mat[i, j] = float(r["value"])

    plt.figure()
    im = plt.imshow(mat, interpolation="nearest")
    plt.colorbar(im)
    plt.xticks(range(len(evl)), evl, rotation=45, ha="right")
    plt.yticks(range(len(src)), src)
    plt.xlabel("Evaluation planner")
    plt.ylabel("Source planner (explanation generated under)")
    plt.title("COSE sufficiency across planners")

    for i in range(len(src)):
        for j in range(len(evl)):
            if np.isfinite(mat[i, j]):
                plt.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center")

    _savefig(os.path.join(out_dir, filename))
    print(f"[OK] Wrote {filename}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results", help="Directory containing metrics.csv and transfer_matrix.csv")
    ap.add_argument("--figures_dir", default="figures", help="Output directory for PDFs")
    ap.add_argument("--agreement_k", type=int, default=5, help="k for Jaccard(top-k) agreement heatmap")
    args = ap.parse_args()

    _ensure_dir(args.figures_dir)

    metrics_path = os.path.join(args.results_dir, "metrics.csv")
    transfer_path = os.path.join(args.results_dir, "transfer_matrix.csv")

    metrics = _read_csv(metrics_path)
    transfer = _read_csv(transfer_path)

    plot_success_at_k(metrics, args.figures_dir, "success_at_k.pdf")
    plot_agreement_heatmap(metrics, args.figures_dir, "agreement_heatmap.pdf", k=args.agreement_k)
    plot_minimality_gap(metrics, args.figures_dir, "minimality_gap.pdf")
    plot_robustness(metrics, args.figures_dir, "robustness.pdf")
    plot_cross_planner_transfer(transfer, args.figures_dir, "cross_planner_transfer.pdf")


if __name__ == "__main__":
    main()
