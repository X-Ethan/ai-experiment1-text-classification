from typing import List, Tuple

from astar import astar_grid, manhattan

Pos = Tuple[int, int]


def parse_cell(x) -> str:
    return str(x).strip()


def make_cost_fn(grid, end):
    def cost(p):
        v = parse_cell(grid[p[0]][p[1]])
        if p == end or v == 'E':
            return 0
        if v == 'S':
            return 1
        if v == 'X':
            return None
        return int(v)
    return cost


def min_weight(grid) -> int:
    ws = [int(parse_cell(v)) for row in grid for v in row if parse_cell(v).isdigit()]
    return min(ws) if ws else 1


def avg_weight(grid) -> float:
    ws = [int(parse_cell(v)) for row in grid for v in row if parse_cell(v).isdigit()]
    return sum(ws) / len(ws) if ws else 1


def make_heuristic(kind: str, end: Pos, wmin: int, wavg: float = 1.0):
    if kind == "zero":
        return lambda p: 0
    if kind == "manhattan":
        return lambda p: manhattan(p, end)
    if kind == "admissible":
        return lambda p: max(manhattan(p, end) - 1, 0) * wmin
    if kind == "avg":
        return lambda p: manhattan(p, end) * wavg
    raise ValueError(kind)


def solve_terrain(start: Pos, end: Pos, grid: List[List], heuristic: str = "admissible",
                  weight: float = 1.0, return_info: bool = False):
    rows, cols = len(grid), len(grid[0])
    h = make_heuristic(heuristic, end, min_weight(grid), avg_weight(grid))
    info = astar_grid(rows, cols, start, end, make_cost_fn(grid, end), h, weight)
    return info if return_info else info["cost"]


TEST_START = (0, 0)
TEST_END = (4, 4)
TEST_GRID = [
    ['S', 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, 'E'],
]

if __name__ == "__main__":
    info = solve_terrain(TEST_START, TEST_END, TEST_GRID, return_info=True)
    print(info["cost"], info["path"])
