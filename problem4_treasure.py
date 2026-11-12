from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from astar import astar_grid, manhattan

RAW_MAP = [
    ['A', '1', '1', 'X', '2', '1', 'T1'],
    ['1', 'X', '6', 'X', '1', 'X', '1'],
    ['1', '1', 'T2', '1', '1', '1', '1'],
    ['X', '1', 'X', '8', 'X', '1', 'X'],
    ['1', '1', '1', '1', 'T3', '1', '1'],
    ['1', 'X', '9', 'X', '1', 'X', '1'],
    ['T4', '1', '1', 'X', '1', '1', 'B'],
]
VALUES = {'T1': 9, 'T2': 6, 'T3': 7, 'T4': 7}
NAMES = ['A', 'B', 'T1', 'T2', 'T3', 'T4']


def locate(grid) -> Dict[str, Tuple[int, int]]:
    return {v: (r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v in NAMES}


def cell_cost(grid, p):
    v = grid[p[0]][p[1]]
    if v == 'X':
        return None
    if v in NAMES:
        return 1
    return int(v)


def distance_matrix(grid=RAW_MAP) -> Dict[str, Dict[str, int]]:
    loc = locate(grid)
    rows, cols = len(grid), len(grid[0])
    wmin = min(cell_cost(grid, (r, c)) for r in range(rows) for c in range(cols)
               if cell_cost(grid, (r, c)) is not None)
    D = {a: {} for a in NAMES}
    for a in NAMES:
        for b in NAMES:
            if a == b:
                D[a][b] = 0
                continue
            goal = loc[b]
            info = astar_grid(rows, cols, loc[a], goal, lambda p: cell_cost(grid, p),
                              lambda p: manhattan(p, goal) * wmin)
            D[a][b] = info["cost"]
    return D


@dataclass(frozen=True)
class GameState:
    pos_a: str
    pos_b: str
    remaining: Tuple[str, ...]
    score_a: int
    score_b: int
    a_to_move: bool

    def utility(self) -> int:
        return self.score_a - self.score_b

    def move(self, t: str, D) -> "GameState":
        rem = tuple(x for x in self.remaining if x != t)
        if self.a_to_move:
            g = VALUES[t] - D[self.pos_a][t]
            return GameState(t, self.pos_b, rem, self.score_a + g, self.score_b, False)
        g = VALUES[t] - D[self.pos_b][t]
        return GameState(self.pos_a, t, rem, self.score_a, self.score_b + g, True)


def initial_state() -> GameState:
    return GameState('A', 'B', ('T1', 'T2', 'T3', 'T4'), 0, 0, True)


class Stats:
    def __init__(self):
        self.nodes = 0
        self.leaves = 0
        self.cutoffs = 0
        self.tt_hits = 0


def ordered_moves(s: GameState, D, order: str) -> List[str]:
    moves = list(s.remaining)
    if order == "reverse":
        moves.reverse()
    elif order in ("gain", "worst"):
        pos = s.pos_a if s.a_to_move else s.pos_b
        moves.sort(key=lambda t: VALUES[t] - D[pos][t], reverse=(order == "gain"))
    return moves


def minimax(s: GameState, depth: int, D, st: Stats, order="default") -> int:
    st.nodes += 1
    if depth == 0 or not s.remaining:
        st.leaves += 1
        return s.utility()
    vals = [minimax(s.move(t, D), depth - 1, D, st, order) for t in ordered_moves(s, D, order)]
    return max(vals) if s.a_to_move else min(vals)


def alphabeta(s: GameState, depth: int, alpha, beta, D, st: Stats, order="default",
              tt: Optional[dict] = None) -> int:
    st.nodes += 1
    u = s.utility()
    if depth == 0 or not s.remaining:
        st.leaves += 1
        return u
    a0, b0 = alpha, beta
    key = (s.pos_a, s.pos_b, s.remaining, s.a_to_move, depth)
    if tt is not None and key in tt:
        flag, rel = tt[key]
        val = u + rel
        if flag == 'exact' or (flag == 'lower' and val >= beta) or (flag == 'upper' and val <= alpha):
            st.tt_hits += 1
            return val
    moves = ordered_moves(s, D, order)
    if s.a_to_move:
        best = float("-inf")
        for i, t in enumerate(moves):
            best = max(best, alphabeta(s.move(t, D), depth - 1, alpha, beta, D, st, order, tt))
            alpha = max(alpha, best)
            if alpha >= beta:
                st.cutoffs += i < len(moves) - 1
                break
    else:
        best = float("inf")
        for i, t in enumerate(moves):
            best = min(best, alphabeta(s.move(t, D), depth - 1, alpha, beta, D, st, order, tt))
            beta = min(beta, best)
            if alpha >= beta:
                st.cutoffs += i < len(moves) - 1
                break
    if tt is not None:
        flag = 'upper' if best <= a0 else 'lower' if best >= b0 else 'exact'
        tt[key] = (flag, best - u)
    return best


def best_first_move(H: int, method: str = "alphabeta", order: str = "default",
                    use_tt: bool = False, D=None):
    D = D or distance_matrix()
    s0 = initial_state()
    st = Stats()
    st.nodes += 1
    tt = {} if use_tt else None
    best_t, best_v = None, float("-inf")
    alpha, beta = float("-inf"), float("inf")
    child_vals = {}
    for t in ordered_moves(s0, D, order):
        child = s0.move(t, D)
        if method == "minimax":
            v = minimax(child, H - 1, D, st, order)
        else:
            v = alphabeta(child, H - 1, alpha, beta, D, st, order, tt)
            alpha = max(alpha, v)
        child_vals[t] = v
        if v > best_v:
            best_t, best_v = t, v
    return {"best_target": best_t, "best_value": best_v}, st, child_vals


def principal_variation(H: int, D=None) -> List[Tuple[str, str, int]]:
    D = D or distance_matrix()
    s, pv = initial_state(), []
    for d in range(H, 0, -1):
        if not s.remaining:
            break
        st = Stats()
        scored = [(minimax(s.move(t, D), d - 1, D, st), t) for t in s.remaining]
        v, t = (max if s.a_to_move else min)(scored, key=lambda x: x[0])
        pos = s.pos_a if s.a_to_move else s.pos_b
        pv.append(('A' if s.a_to_move else 'B', t, VALUES[t] - D[pos][t]))
        s = s.move(t, D)
    return pv


if __name__ == "__main__":
    D = distance_matrix()
    print("      " + "".join(f"{n:>5}" for n in NAMES))
    for a in NAMES:
        print(f"{a:>5} " + "".join(f"{D[a][b]:>5}" for b in NAMES))
    for H in (2, 4):
        for m in ("minimax", "alphabeta"):
            res, st, cv = best_first_move(H, m, D=D)
            print(H, m, res, st.nodes, cv, principal_variation(H, D))
