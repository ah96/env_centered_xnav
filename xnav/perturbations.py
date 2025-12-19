from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .env import GridEnvironment, extract_entities_from_grid
from .interventions import move_entity

def jitter_obstacles(env: GridEnvironment, rng: np.random.Generator, p_move: float = 0.2) -> GridEnvironment:
    """
    Randomly move a fraction of entities (small jitter). Uses the same move primitive.
    """
    out = env
    for eid in sorted(env.entities.keys()):
        if rng.random() < p_move:
            out = move_entity(out, eid, rng=rng, max_tries=50)
    # ensure entities consistent
    ents = extract_entities_from_grid(out.grid)
    return GridEnvironment(out.grid, ents, out.start, out.goal)

def add_distractor(env: GridEnvironment, rng: np.random.Generator, size: int = 5, max_tries: int = 200) -> GridEnvironment:
    """
    Insert a small random blob far from start-goal corridor (heuristic).
    """
    grid = env.grid.copy()
    h, w = grid.shape

    # avoid start/goal neighborhood
    forbidden = {env.start, env.goal}
    for _ in range(max_tries):
        r0 = int(rng.integers(0, h))
        c0 = int(rng.integers(0, w))
        if (r0, c0) in forbidden:
            continue
        # place a small random walk if free
        if grid[r0, c0] == 1:
            continue
        r, c = r0, c0
        placed = []
        ok = True
        for _ in range(size):
            if 0 <= r < h and 0 <= c < w and grid[r, c] == 0 and (r, c) not in forbidden:
                grid[r, c] = 1
                placed.append((r, c))
            dr, dc = rng.choice([(1,0),(-1,0),(0,1),(0,-1)])
            r = int(np.clip(r + dr, 0, h - 1))
            c = int(np.clip(c + dc, 0, w - 1))
        if placed:
            break

    ents = extract_entities_from_grid(grid)
    return GridEnvironment(grid, ents, env.start, env.goal)
