from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from .env import GridEnvironment, extract_entities_from_grid, Coord


def remove_entity(env: GridEnvironment, eid: int) -> GridEnvironment:
    new = env.copy()
    coords = new.entities.get(eid, [])
    for (r, c) in coords:
        new.grid[r, c] = 0
    new_entities = extract_entities_from_grid(new.grid)
    return GridEnvironment(new.grid, new_entities, new.start, new.goal)


def move_entity(
    env: GridEnvironment,
    eid: int,
    rng: np.random.Generator,
    max_tries: int = 200,
) -> GridEnvironment:
    """
    Move entity as a rigid shape by translating all its cells (dr, dc).
    Must remain within bounds and not overlap existing occupied cells (excluding itself).
    """
    if eid not in env.entities:
        return env

    shape = env.entities[eid]
    coords = np.array(shape, dtype=int)
    h, w = env.H, env.W

    # Create occupancy mask excluding this entity
    base_grid = env.grid.copy()
    for r, c in shape:
        base_grid[r, c] = 0

    # bounding box of shape
    rmin, cmin = coords.min(axis=0)
    rmax, cmax = coords.max(axis=0)

    for _ in range(max_tries):
        # random translation range that keeps bbox inside
        dr = int(rng.integers(-rmin, h - 1 - rmax + 1))
        dc = int(rng.integers(-cmin, w - 1 - cmax + 1))

        moved = coords + np.array([dr, dc])
        # check overlap
        ok = True
        for rr, cc in moved:
            if base_grid[rr, cc] == 1:
                ok = False
                break
        if not ok:
            continue

        # apply move
        new_grid = base_grid.copy()
        for rr, cc in moved:
            new_grid[rr, cc] = 1

        # keep start/goal free
        new_grid[env.start] = 0
        new_grid[env.goal] = 0

        new_entities = extract_entities_from_grid(new_grid)
        return GridEnvironment(new_grid, new_entities, env.start, env.goal)

    # If cannot move, fall back to no-op (or removal if you prefer)
    return env


def apply_interventions(
    env: GridEnvironment,
    entity_ids: List[int],
    mode: str,
    rng: np.random.Generator,
) -> GridEnvironment:
    """
    mode: "remove" or "move"
    Applies sequentially to keep it simple and deterministic.
    """
    out = env
    for eid in entity_ids:
        if mode == "remove":
            out = remove_entity(out, eid)
        elif mode == "move":
            out = move_entity(out, eid, rng)
        else:
            raise ValueError(f"Unknown intervention mode: {mode}")
    return out
