from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from .env import GridEnvironment
from .interventions import apply_interventions

PlannerFn = Callable[[GridEnvironment], bool]


def success_at_k(
    env: GridEnvironment,
    planner: PlannerFn,
    mode: str,
    rng: np.random.Generator,
    ranked_entities: List[int],
    k: int,
) -> bool:
    chosen = ranked_entities[:k]
    env_p = apply_interventions(env, chosen, mode=mode, rng=rng)
    return bool(planner(env_p))


def jaccard(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def kendall_tau(a_scores: np.ndarray, b_scores: np.ndarray) -> float:
    """
    Simple Kendall's tau for rankings induced by scores (ties broken by index).
    Uses O(m^2) implementation for small m typical here.
    """
    m = len(a_scores)
    if m <= 1:
        return 1.0
    a = np.argsort(-a_scores, kind="stable")
    b = np.argsort(-b_scores, kind="stable")
    pos_b = np.zeros(m, dtype=int)
    for i, idx in enumerate(b):
        pos_b[idx] = i

    concordant = 0
    discordant = 0
    for i in range(m):
        for j in range(i + 1, m):
            ai, aj = a[i], a[j]
            # In a: ai before aj by construction
            if pos_b[ai] < pos_b[aj]:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom > 0 else 0.0
