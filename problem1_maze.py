"""题目一：迷宫寻路（A*，四方向，每步代价 1，曼哈顿距离启发）。"""
from typing import List, Tuple

from astar import astar_grid, manhattan

Pos = Tuple[int, int]


def solve_maze(start: Pos, end: Pos, grid: List[List[str]], use_heuristic: bool = True,
               weight: float = 1.0, return_info: bool = False):
    """返回从 start 到 end 的最短路径坐标列表；无解返回 []。

    use_heuristic=False 时 h≡0，A* 退化为一致代价搜索（此处等价于 BFS）。
    """
    rows, cols = len(grid), len(grid[0])
    cost = lambda p: None if grid[p[0]][p[1]] == 'X' else 1
    h = (lambda p: manhattan(p, end)) if use_heuristic else (lambda p: 0)
    info = astar_grid(rows, cols, start, end, cost, h, weight)
    return info if return_info else info["path"]


TEST_START = (0, 0)
TEST_END = (3, 4)
TEST_GRID = [
    ['S', '.', '.', 'X', '.'],
    ['.', 'X', '.', '.', '.'],
    ['.', 'X', '.', 'X', '.'],
    ['.', '.', '.', 'X', 'E'],
    ['X', 'X', '.', '.', '.'],
]

if __name__ == "__main__":
    print(solve_maze(TEST_START, TEST_END, TEST_GRID))
