from __future__ import annotations
from typing import Callable, List, Optional, Tuple
import itertools
import numpy as np
from .env import GridEnvironment
from .interventions import apply_interventions

PlannerFn = Callable[[GridEnvironment], bool]


def oracle_min_counterfactual_bruteforce(
    env: GridEnvironment,
    planner: PlannerFn,
    mode: str,
    rng: np.random.Generator,
    max_entities: int = 18,
    max_k: int = 6,
) -> Optional[List[int]]:
    """
    Exact oracle by brute force over subsets up to size max_k.
    Only feasible for small number of entities. Use for evaluation only.
    Returns one minimal set (not all).
    """
    eids = sorted(env.entities.keys())
    if len(eids) > max_entities:
        return None

    for k in range(1, min(max_k, len(eids)) + 1):
        for subset in itertools.combinations(eids, k):
            env_p = apply_interventions(env, list(subset), mode=mode, rng=rng)
            if planner(env_p):
                return list(subset)
    return None
