from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import deque

Coord = Tuple[int, int]


@dataclass(frozen=True)
class GridEnvironment:
    """
    Binary occupancy grid environment with explicit obstacle entities.
    grid[r, c] == 1 means occupied, 0 means free.
    entities maps entity_id -> list of occupied coordinates.
    """
    grid: np.ndarray
    entities: Dict[int, List[Coord]]
    start: Coord
    goal: Coord

    @property
    def H(self) -> int:
        return int(self.grid.shape[0])

    @property
    def W(self) -> int:
        return int(self.grid.shape[1])

    def is_free(self, p: Coord) -> bool:
        r, c = p
        return 0 <= r < self.H and 0 <= c < self.W and self.grid[r, c] == 0

    def copy(self) -> "GridEnvironment":
        # Deep copy grid; entities lists are rebuilt by interventions typically.
        return GridEnvironment(self.grid.copy(), {k: v.copy() for k, v in self.entities.items()},
                               self.start, self.goal)


def _neighbors4(h: int, w: int, r: int, c: int):
    if r > 0: yield (r - 1, c)
    if r + 1 < h: yield (r + 1, c)
    if c > 0: yield (r, c - 1)
    if c + 1 < w: yield (r, c + 1)


def extract_entities_from_grid(grid: np.ndarray) -> Dict[int, List[Coord]]:
    """
    Connected components of occupied cells (4-connectivity).
    Returns entity_id -> coords.
    """
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    entities: Dict[int, List[Coord]] = {}
    eid = 0

    for r in range(h):
        for c in range(w):
            if grid[r, c] != 1 or visited[r, c]:
                continue
            # BFS component
            q = deque([(r, c)])
            visited[r, c] = True
            comp: List[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(h, w, rr, cc):
                    if not visited[nr, nc] and grid[nr, nc] == 1:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            entities[eid] = comp
            eid += 1
    return entities


def generate_random_grid(
    h: int,
    w: int,
    obstacle_density: float,
    rng: np.random.Generator,
    blob_min: int = 2,
    blob_max: int = 10,
) -> np.ndarray:
    """
    Generate an occupancy grid with blob-like obstacles by placing random rectangles
    and random walks. This tends to create connected entities rather than salt-and-pepper.
    """
    grid = np.zeros((h, w), dtype=np.uint8)

    # Target occupied cells
    target = int(h * w * obstacle_density)
    occupied = 0

    while occupied < target:
        # Choose blob size
        size = int(rng.integers(blob_min, blob_max + 1))
        r0 = int(rng.integers(0, h))
        c0 = int(rng.integers(0, w))

        # Random walk blob
        r, c = r0, c0
        for _ in range(size):
            if 0 <= r < h and 0 <= c < w:
                if grid[r, c] == 0:
                    grid[r, c] = 1
                    occupied += 1
                    if occupied >= target:
                        break
            # step
            dr, dc = rng.choice([(1,0),(-1,0),(0,1),(0,-1)])
            r = int(np.clip(r + dr, 0, h - 1))
            c = int(np.clip(c + dc, 0, w - 1))

    return grid


def sample_start_goal(grid: np.ndarray, rng: np.random.Generator, min_manhattan: int = 10) -> Tuple[Coord, Coord]:
    h, w = grid.shape
    free = np.argwhere(grid == 0)
    if len(free) < 2:
        raise ValueError("Not enough free cells to sample start/goal.")
    for _ in range(5000):
        s = tuple(free[rng.integers(0, len(free))])
        g = tuple(free[rng.integers(0, len(free))])
        if abs(s[0]-g[0]) + abs(s[1]-g[1]) >= min_manhattan:
            return (int(s[0]), int(s[1])), (int(g[0]), int(g[1]))
    # fallback
    s = tuple(free[0]); g = tuple(free[-1])
    return (int(s[0]), int(s[1])), (int(g[0]), int(g[1]))


def make_environment(
    h: int,
    w: int,
    obstacle_density: float,
    rng: np.random.Generator,
    min_entities: int = 5,
    max_tries: int = 100,
) -> GridEnvironment:
    """
    Generate environment with extracted entities and sampled start/goal.
    Ensures at least min_entities exist (otherwise regenerate).
    """
    for _ in range(max_tries):
        grid = generate_random_grid(h, w, obstacle_density, rng)
        entities = extract_entities_from_grid(grid)
        if len(entities) < min_entities:
            continue
        start, goal = sample_start_goal(grid, rng)
        # Ensure start/goal are free
        grid[start] = 0
        grid[goal] = 0
        # Re-extract entities because we may have cleared cells
        entities = extract_entities_from_grid(grid)
        return GridEnvironment(grid=grid, entities=entities, start=start, goal=goal)
    raise RuntimeError("Failed to generate environment with enough entities.")
