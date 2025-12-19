from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
import numpy as np
from sklearn.linear_model import Ridge
from .env import GridEnvironment
from .interventions import apply_interventions

PlannerFn = Callable[[GridEnvironment], bool]


@dataclass
class AttributionResult:
    scores: np.ndarray            # length m (entities)
    entity_ids: List[int]         # mapping indices->entity ids
    budget_used: int


def _entity_id_list(env: GridEnvironment) -> List[int]:
    # Stable ordering
    return sorted(env.entities.keys())


def lime_attribution(
    env: GridEnvironment,
    planner: PlannerFn,
    mode: str,
    rng: np.random.Generator,
    budget: int = 400,
    p_on: float = 0.3,
    kernel_width: float = 0.25,
) -> AttributionResult:
    """
    LIME-style: sample binary masks over entities, apply interventions to masked-on entities,
    fit weighted linear surrogate: y ~ w0 + sum w_i * mask_i
    We explain failure, so label is success (1) after intervention; coefficients indicate which
    entities help success when intervened upon (i.e., were responsible for failure).
    """
    eids = _entity_id_list(env)
    m = len(eids)
    if m == 0:
        return AttributionResult(scores=np.array([]), entity_ids=[], budget_used=0)

    X = np.zeros((budget, m), dtype=float)
    y = np.zeros((budget,), dtype=float)
    w = np.zeros((budget,), dtype=float)

    for b in range(budget):
        mask = (rng.random(m) < p_on).astype(float)
        # Ensure not all zeros (or allow; but LIME usually varies)
        if mask.sum() == 0:
            mask[rng.integers(0, m)] = 1.0
        chosen = [eids[i] for i in range(m) if mask[i] > 0.5]
        env_p = apply_interventions(env, chosen, mode=mode, rng=rng)
        y[b] = 1.0 if planner(env_p) else 0.0
        X[b, :] = mask

        # LIME kernel weight based on Hamming distance to "instance" (all-zeros = original failure)
        # distance = fraction of toggled features
        dist = mask.mean()  # since original is all zeros
        w[b] = np.exp(-(dist ** 2) / (kernel_width ** 2))

    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X, y, sample_weight=w)
    scores = model.coef_.copy()  # higher => intervening this entity increases success probability
    return AttributionResult(scores=scores, entity_ids=eids, budget_used=budget)


def shap_attribution(
    env: GridEnvironment,
    planner: PlannerFn,
    mode: str,
    rng: np.random.Generator,
    budget: int = 800,
) -> AttributionResult:
    """
    Monte Carlo Shapley approximation:
    For each sample, take a random permutation; traverse adding entities and measure marginal gain
    in outcome (success after interventions). This yields Shapley values over entities.
    """
    eids = _entity_id_list(env)
    m = len(eids)
    if m == 0:
        return AttributionResult(scores=np.array([]), entity_ids=[], budget_used=0)

    phi = np.zeros(m, dtype=float)
    # Each permutation consumes up to m planner calls; we cap total calls by budget.
    perms = max(1, budget // max(1, m))
    calls = 0

    for _ in range(perms):
        perm = rng.permutation(m)
        current_set: List[int] = []
        # baseline: no intervention
        env0 = apply_interventions(env, current_set, mode=mode, rng=rng)
        prev = 1.0 if planner(env0) else 0.0
        calls += 1

        for idx in perm:
            current_set.append(eids[idx])
            env1 = apply_interventions(env, current_set, mode=mode, rng=rng)
            cur = 1.0 if planner(env1) else 0.0
            calls += 1
            phi[idx] += (cur - prev)
            prev = cur
            if calls >= budget:
                break
        if calls >= budget:
            break

    phi /= max(1, perms)
    return AttributionResult(scores=phi, entity_ids=eids, budget_used=calls)


@dataclass
class CounterfactualResult:
    set_entities: List[int]
    budget_used: int


def cose_counterfactual(
    env: GridEnvironment,
    planner: PlannerFn,
    mode: str,
    rng: np.random.Generator,
    ranking: Optional[List[int]] = None,
) -> CounterfactualResult:
    """
    COSE: forward selection until success, then redundancy pruning.
    Greedy is justified by monotonicity under removal; for move, it is heuristic but works well.
    """
    eids = _entity_id_list(env)
    order = ranking if ranking is not None else eids
    chosen: List[int] = []
    calls = 0

    # forward selection
    for eid in order:
        chosen.append(eid)
        env_p = apply_interventions(env, chosen, mode=mode, rng=rng)
        calls += 1
        if planner(env_p):
            break

    # pruning
    pruned = chosen.copy()
    for eid in chosen:
        trial = [x for x in pruned if x != eid]
        env_p = apply_interventions(env, trial, mode=mode, rng=rng)
        calls += 1
        if planner(env_p):
            pruned = trial

    return CounterfactualResult(set_entities=pruned, budget_used=calls)
