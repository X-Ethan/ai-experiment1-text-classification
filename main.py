import json
import sys

import problem1_maze as p1
import problem2_terrain as p2
import problem3_alphabeta as p3
import problem4_treasure as p4


def check(name, got, ok):
    print(f"[{'通过' if ok else '失败'}] {name}: {got}")
    return ok


def run_tests() -> bool:
    all_ok = True
    print("=" * 20, "题目一：迷宫寻路（A*）", "=" * 20)
    path = p1.solve_maze(p1.TEST_START, p1.TEST_END, p1.TEST_GRID)
    valid = (len(path) == 8 and path[0] == p1.TEST_START and path[-1] == p1.TEST_END
             and all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:]))
             and all(p1.TEST_GRID[r][c] != 'X' for r, c in path))
    all_ok &= check("最短路径", path, valid)

    print("=" * 20, "题目二：加权地形（A*）", "=" * 20)
    info = p2.solve_terrain(p2.TEST_START, p2.TEST_END, p2.TEST_GRID, return_info=True)
    all_ok &= check("最小代价", info["cost"], info["cost"] == 7)
    print("       路径:", " -> ".join(map(str, info["path"])))

    print("=" * 20, "题目三：Alpha-Beta 剪枝", "=" * 20)
    v1 = p3.best_score(p3.build_example())
    v2 = p3.best_score(p3.from_json(p3.TEST_JSON))
    all_ok &= check("TreeNode 构建输入", v1, v1 == 5)
    all_ok &= check("JSON 输入", v2, v2 == 5)

    print("=" * 20, "题目四：宝藏分配（A* + 博弈搜索）", "=" * 20)
    D = p4.distance_matrix()
    print("A* 距离矩阵：")
    print("      " + "".join(f"{n:>5}" for n in p4.NAMES))
    for a in p4.NAMES:
        print(f"{a:>5} " + "".join(f"{D[a][b]:>5}" for b in p4.NAMES))
    expected = {2: {"best_target": "T2", "best_value": -1}, 4: {"best_target": "T3", "best_value": 3}}
    for H in (2, 4):
        for m in ("minimax", "alphabeta"):
            res, st, _ = p4.best_first_move(H, m, D=D)
            all_ok &= check(f"H={H} {m:<9} (访问状态数 {st.nodes:>3})",
                            json.dumps(res, ensure_ascii=False), res == expected[H])
        print("       最优对弈序列:", p4.principal_variation(H, D))
    return all_ok


def run_experiments():
    import experiments
    maze, terrain, tree, treasure = experiments.run_all()
    print("\n" + "=" * 20, "对照实验摘要", "=" * 20)
    print("题目一 扩展节点数  h=0: %d  曼哈顿: %d" % (maze["test"]["h0"]["expanded"], maze["test"]["manhattan"]["expanded"]))
    for n, v in maze["random"].items():
        print(f"  随机迷宫 n={n:<3} " + "  ".join(f"{k}:{x['expanded']:.0f}(路径比{x['len_ratio']:.3f})" for k, x in v.items()))
    print("题目二 随机地形(200 个 30×30)：")
    for k, v in terrain["random"].items():
        print(f"  {k:<20} 扩展 {v['expanded']:.1f}  非最优 {v['suboptimal']}/200  平均超额 {v['excess_pct']:.2f}%")
    print("题目三 测试树：", {k: (v["visited"], v["leaves"]) for k, v in tree["test"].items()})
    for k, v in tree["random"].items():
        print("  随机树", k, {a: round(b, 1) for a, b in v.items()})
    print("题目四 访问状态数：")
    for r in treasure["runs"]:
        print(f"  H={r['H']} {r['method']:<9} 顺序={r['order']:<7} TT={str(r['tt']):<5} 节点={r['nodes']:<3} "
              f"叶子={r['leaves']:<3} 剪枝={r['cutoffs']}  结果={r['best_target']},{r['best_value']}")
    print("  不同深度的决策：", [(x["H"], x["best_target"], x["best_value"]) for x in treasure["by_H"]])
    print("拓展（8 宝藏，平均访问状态数）：")
    for H, v in treasure["extension"].items():
        print(f"  H={H} " + "  ".join(f"{k}:{x['nodes']:.0f}" for k, x in v.items() if k != "values_agree")
              + f"  结果一致:{v['values_agree']}")
    print("\n详细结果：results/results.json，图表：figures/")


if __name__ == "__main__":
    ok = run_tests()
    print("\n全部测试用例通过！" if ok else "\n存在未通过的测试用例！")
    if "--quick" not in sys.argv:
        run_experiments()
    sys.exit(0 if ok else 1)
