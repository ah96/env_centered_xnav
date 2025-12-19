from __future__ import annotations
import argparse
from dataclasses import asdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from xnav.env import make_environment
from xnav.planners import PLANNERS
from xnav.explainers import lime_attribution, shap_attribution, cose_counterfactual
from xnav.metrics import success_at_k, jaccard, kendall_tau
from xnav.oracle import oracle_min_counterfactual_bruteforce
from xnav.perturbations import jitter_obstacles, add_distractor


def rank_from_scores(entity_ids: List[int], scores: np.ndarray) -> List[int]:
    order = np.argsort(-scores, kind="stable")
    return [entity_ids[i] for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["remove", "move"], default="remove")

    ap.add_argument("--grid_sizes", default="20,30,40")
    ap.add_argument("--densities", default="0.20,0.28,0.35")
    ap.add_argument("--envs_per_cfg", type=int, default=30)

    ap.add_argument("--lime_budget", type=int, default=400)
    ap.add_argument("--shap_budget", type=int, default=800)

    ap.add_argument("--k_list", default="1,2,3,5")
    ap.add_argument("--oracle_max_entities", type=int, default=18)
    ap.add_argument("--oracle_max_k", type=int, default=6)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    grid_sizes = [int(x) for x in args.grid_sizes.split(",")]
    densities = [float(x) for x in args.densities.split(",")]
    k_list = [int(x) for x in args.k_list.split(",")]

    rows = []
    transfer_rows = []

    planner_names = list(PLANNERS.keys())

    for H in grid_sizes:
        for dens in densities:
            for n in tqdm(range(args.envs_per_cfg), desc=f"Generating envs H={H} dens={dens}"):
                env = make_environment(H, H, dens, rng=rng)

                # Evaluate each planner; we focus on failures
                base_outcomes = {p: PLANNERS[p](env) for p in planner_names}

                for p_name in planner_names:
                    planner = PLANNERS[p_name]
                    if base_outcomes[p_name]:
                        continue  # only failures to explain

                    # Explain with LIME/SHAP
                    lime = lime_attribution(env, planner, mode=args.mode, rng=rng, budget=args.lime_budget)
                    shap = shap_attribution(env, planner, mode=args.mode, rng=rng, budget=args.shap_budget)

                    lime_rank = rank_from_scores(lime.entity_ids, lime.scores)
                    shap_rank = rank_from_scores(shap.entity_ids, shap.scores)

                    # COSE ranking-guided by SHAP (common choice)
                    cose = cose_counterfactual(env, planner, mode=args.mode, rng=rng, ranking=shap_rank)

                    # Oracle (optional)
                    oracle_set = oracle_min_counterfactual_bruteforce(
                        env, planner, mode=args.mode, rng=rng,
                        max_entities=args.oracle_max_entities, max_k=args.oracle_max_k
                    )

                    # Q1: Success@k for attribution
                    for k in k_list:
                        rows.append({
                            "H": H, "density": dens, "planner": p_name, "method": "lime", "metric": f"success_at_{k}",
                            "value": float(success_at_k(env, planner, args.mode, rng, lime_rank, k))
                        })
                        rows.append({
                            "H": H, "density": dens, "planner": p_name, "method": "shap", "metric": f"success_at_{k}",
                            "value": float(success_at_k(env, planner, args.mode, rng, shap_rank, k))
                        })

                    # Agreement (Jaccard top-k)
                    k_agree = min(5, len(lime_rank), len(shap_rank))
                    rows.append({
                        "H": H, "density": dens, "planner": p_name, "method": "lime_vs_shap", "metric": f"jaccard_top_{k_agree}",
                        "value": jaccard(lime_rank[:k_agree], shap_rank[:k_agree])
                    })

                    # Q2: COSE set size, success
                    rows.append({"H": H, "density": dens, "planner": p_name, "method": "cose", "metric": "set_size", "value": len(cose.set_entities)})
                    rows.append({"H": H, "density": dens, "planner": p_name, "method": "cose", "metric": "success", "value": float(PLANNERS[p_name](env))})
                    # Actually evaluate intervention success for COSE:
                    from xnav.interventions import apply_interventions
                    env_c = apply_interventions(env, cose.set_entities, mode=args.mode, rng=rng)
                    rows[-1]["value"] = float(planner(env_c))

                    if oracle_set is not None:
                        rows.append({"H": H, "density": dens, "planner": p_name, "method": "oracle", "metric": "set_size", "value": len(oracle_set)})
                        rows.append({"H": H, "density": dens, "planner": p_name, "method": "cose", "metric": "min_gap_oracle", "value": len(cose.set_entities) - len(oracle_set)})

                    # Q3: Robustness (jitter + distractor)
                    env_j = jitter_obstacles(env, rng=rng, p_move=0.2)
                    env_d = add_distractor(env, rng=rng, size=5)

                    lime_j = lime_attribution(env_j, planner, args.mode, rng, budget=max(150, args.lime_budget//2))
                    shap_j = shap_attribution(env_j, planner, args.mode, rng, budget=max(200, args.shap_budget//2))

                    rows.append({"H": H, "density": dens, "planner": p_name, "method": "lime", "metric": "kendall_tau_jitter",
                                 "value": kendall_tau(lime.scores, lime_j.scores) if len(lime.scores)==len(lime_j.scores) else 0.0})
                    rows.append({"H": H, "density": dens, "planner": p_name, "method": "shap", "metric": "kendall_tau_jitter",
                                 "value": kendall_tau(shap.scores, shap_j.scores) if len(shap.scores)==len(shap_j.scores) else 0.0})

                    # Transfer: explanations generated for planner p_name, evaluated under other planners
                    # We test sufficiency: apply COSE set, see if other planner succeeds.
                    for q_name in planner_names:
                        q_planner = PLANNERS[q_name]
                        env_cf = apply_interventions(env, cose.set_entities, mode=args.mode, rng=rng)
                        transfer_rows.append({
                            "H": H, "density": dens, "source_planner": p_name, "eval_planner": q_name,
                            "metric": "cose_sufficiency", "value": float(q_planner(env_cf))
                        })

    df = pd.DataFrame(rows)
    tf = pd.DataFrame(transfer_rows)

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(f"{args.out_dir}/metrics.csv", index=False)
    tf.to_csv(f"{args.out_dir}/transfer_matrix.csv", index=False)

    print(f"Wrote {len(df)} metric rows to {args.out_dir}/metrics.csv")
    print(f"Wrote {len(tf)} transfer rows to {args.out_dir}/transfer_matrix.csv")


if __name__ == "__main__":
    main()
