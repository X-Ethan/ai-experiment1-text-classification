import heapq
from typing import Callable, Dict, List, Optional, Tuple

Pos = Tuple[int, int]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def astar_grid(
    n_rows: int,
    n_cols: int,
    start: Pos,
    goal: Pos,
    step_cost: Callable[[Pos], Optional[float]],
    heuristic: Callable[[Pos], float],
    weight: float = 1.0,
) -> Dict:
    counter = 0
    open_heap = [(weight * heuristic(start), heuristic(start), counter, start)]
    g_score = {start: 0}
    parent: Dict[Pos, Optional[Pos]] = {start: None}
    closed = set()
    order: List[Pos] = []
    generated = 1

    while open_heap:
        _, _, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        order.append(cur)
        if cur == goal:
            path = []
            node = cur
            while node is not None:
                path.append(node)
                node = parent[node]
            return dict(path=path[::-1], cost=g_score[cur], expanded=len(closed),
                        generated=generated, closed=closed, order=order)
        r, c = cur
        for dr, dc in DIRS:
            nxt = (r + dr, c + dc)
            if not (0 <= nxt[0] < n_rows and 0 <= nxt[1] < n_cols) or nxt in closed:
                continue
            w = step_cost(nxt)
            if w is None:
                continue
            ng = g_score[cur] + w
            if ng < g_score.get(nxt, float("inf")):
                g_score[nxt] = ng
                parent[nxt] = cur
                counter += 1
                generated += 1
                h = heuristic(nxt)
                heapq.heappush(open_heap, (ng + weight * h, h, counter, nxt))
    return dict(path=[], cost=-1, expanded=len(closed), generated=generated,
                closed=closed, order=order)


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
