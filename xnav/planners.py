from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import heapq
import numpy as np
from .env import GridEnvironment, Coord

def _neighbors4(env: GridEnvironment, p: Coord):
    r, c = p
    for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
        if 0 <= rr < env.H and 0 <= cc < env.W and env.grid[rr, cc] == 0:
            yield (rr, cc)

def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def bfs_success(env: GridEnvironment) -> bool:
    from collections import deque
    s, g = env.start, env.goal
    if not env.is_free(s) or not env.is_free(g):
        return False
    q = deque([s])
    seen = {s}
    while q:
        p = q.popleft()
        if p == g:
            return True
        for n in _neighbors4(env, p):
            if n not in seen:
                seen.add(n)
                q.append(n)
    return False

def dijkstra_success(env: GridEnvironment) -> bool:
    s, g = env.start, env.goal
    if not env.is_free(s) or not env.is_free(g):
        return False
    dist = {s: 0}
    pq = [(0, s)]
    while pq:
        d, p = heapq.heappop(pq)
        if p == g:
            return True
        if d != dist.get(p, None):
            continue
        for n in _neighbors4(env, p):
            nd = d + 1
            if nd < dist.get(n, 10**18):
                dist[n] = nd
                heapq.heappush(pq, (nd, n))
    return False

def astar_success(env: GridEnvironment) -> bool:
    s, g = env.start, env.goal
    if not env.is_free(s) or not env.is_free(g):
        return False
    gscore = {s: 0}
    fscore = {s: _manhattan(s, g)}
    pq = [(fscore[s], s)]
    while pq:
        _, p = heapq.heappop(pq)
        if p == g:
            return True
        for n in _neighbors4(env, p):
            tentative = gscore[p] + 1
            if tentative < gscore.get(n, 10**18):
                gscore[n] = tentative
                fscore[n] = tentative + _manhattan(n, g)
                heapq.heappush(pq, (fscore[n], n))
    return False

PLANNERS = {
    "bfs": bfs_success,
    "dijkstra": dijkstra_success,
    "astar": astar_success,
}
