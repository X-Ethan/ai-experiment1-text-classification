import json
import os
import random
import statistics
import time

import problem1_maze as p1
import problem2_terrain as p2
import problem3_alphabeta as p3
import problem4_treasure as p4
from astar import astar_grid, manhattan

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
SEED = 2026


def random_maze(n, density, rng):
    g = [['X' if rng.random() < density else '.' for _ in range(n)] for _ in range(n)]
    g[0][0], g[n - 1][n - 1] = 'S', 'E'
    return g


def exp_maze():
    out = {}
    info0 = p1.solve_maze(p1.TEST_START, p1.TEST_END, p1.TEST_GRID, False, return_info=True)
    infoh = p1.solve_maze(p1.TEST_START, p1.TEST_END, p1.TEST_GRID, True, return_info=True)
    out["test"] = {"h0": {"expanded": info0["expanded"], "generated": info0["generated"], "len": len(info0["path"])},
                   "manhattan": {"expanded": infoh["expanded"], "generated": infoh["generated"], "len": len(infoh["path"])}}
    rng = random.Random(SEED)
    configs = [("h=0", False, 1.0), ("曼哈顿", True, 1.0), ("加权A* w=2", True, 2.0), ("贪心 w=100", True, 100.0)]
    rand = {}
    for n in (20, 40, 60, 80, 100):
        stats = {c[0]: {"exp": [], "len": [], "ms": []} for c in configs}
        got = 0
        while got < 20:
            g = random_maze(n, 0.3, rng)
            base = p1.solve_maze((0, 0), (n - 1, n - 1), g, False, return_info=True)
            if not base["path"]:
                continue
            got += 1
            for name, use_h, w in configs:
                t = time.perf_counter()
                info = p1.solve_maze((0, 0), (n - 1, n - 1), g, use_h, w, return_info=True)
                stats[name]["ms"].append((time.perf_counter() - t) * 1000)
                stats[name]["exp"].append(info["expanded"])
                stats[name]["len"].append((len(info["path"]) - 1) / (len(base["path"]) - 1))
        rand[n] = {k: {"expanded": statistics.mean(v["exp"]), "ms": statistics.mean(v["ms"]),
                       "len_ratio": statistics.mean(v["len"]), "len_ratio_max": max(v["len"])}
                   for k, v in stats.items()}
    out["random"] = rand
    g = [row[:] for row in p1.TEST_GRID]
    g[2][4] = g[4][4] = 'X'; g[3][3] = 'X'
    out["no_path"] = {"h0": p1.solve_maze((0, 0), (3, 4), g, False, return_info=True)["expanded"],
                      "manhattan": p1.solve_maze((0, 0), (3, 4), g, True, return_info=True)["expanded"],
                      "path": p1.solve_maze((0, 0), (3, 4), g)}
    return out, info0, infoh


def exp_terrain():
    out = {"test": {}}
    variants = [("h=0", "zero", 1.0), ("(d-1)·w_min（可采纳）", "admissible", 1.0),
                ("曼哈顿 d", "manhattan", 1.0), ("d×平均权值", "avg", 1.0),
                ("加权A* w=2", "admissible", 2.0)]
    for name, h, w in variants:
        info = p2.solve_terrain(p2.TEST_START, p2.TEST_END, p2.TEST_GRID, h, w, return_info=True)
        out["test"][name] = {"cost": info["cost"], "expanded": info["expanded"]}
    info = p2.solve_terrain(p2.TEST_START, p2.TEST_END, p2.TEST_GRID, return_info=True)
    cost_fn = p2.make_cost_fn(p2.TEST_GRID, p2.TEST_END)
    acc, g = [], 0
    for p in info["path"][1:]:
        g += cost_fn(p); acc.append((p, cost_fn(p), g))
    out["accumulate"] = acc
    rng = random.Random(SEED)
    stats = {v[0]: {"exp": [], "sub": 0, "excess": []} for v in variants}
    for _ in range(200):
        n = 30
        grid = [[rng.choices([1, 3, 8], [0.6, 0.25, 0.15])[0] for _ in range(n)] for _ in range(n)]
        grid[0][0], grid[n - 1][n - 1] = 'S', 'E'
        opt = p2.solve_terrain((0, 0), (n - 1, n - 1), grid, "zero")
        for name, h, w in variants:
            info = p2.solve_terrain((0, 0), (n - 1, n - 1), grid, h, w, return_info=True)
            stats[name]["exp"].append(info["expanded"])
            stats[name]["sub"] += info["cost"] > opt
            stats[name]["excess"].append((info["cost"] - opt) / opt)
    out["random"] = {k: {"expanded": statistics.mean(v["exp"]), "suboptimal": v["sub"],
                         "excess_pct": 100 * statistics.mean(v["excess"]),
                         "excess_max_pct": 100 * max(v["excess"])} for k, v in stats.items()}
    return out


def random_tree(b, d, rng):
    if d == 0:
        return p3.TreeNode(value=rng.randint(0, 99))
    return p3.TreeNode(children=[random_tree(b, d - 1, rng) for _ in range(b)])


def exp_tree():
    out = {"test": {}}
    root = p3.annotate_players(p3.from_json(p3.TEST_JSON))
    c = p3.Counter(); v = p3.minimax(root, True, c)
    out["test"]["Minimax"] = {"value": v, "visited": c.visited, "leaves": c.leaves, "cutoffs": 0}
    for mode, name in (("left", "AB 从左到右"), ("right", "AB 从右到左"), ("best", "AB 最优序"), ("worst", "AB 最差序")):
        c = p3.Counter(); v = p3.alpha_beta(p3.reorder(root, mode), cnt=c)
        out["test"][name] = {"value": v, "visited": c.visited, "leaves": c.leaves, "cutoffs": c.cutoffs}
    c = p3.Counter(); p3.alpha_beta(p3.build_example(), cnt=c)
    out["test_shared"] = {"visited": c.visited, "leaves": c.leaves}
    rng = random.Random(SEED)
    rand = {}
    for b, d in ((3, 6), (4, 6), (5, 6)):
        acc = {k: [] for k in ("Minimax", "AB 随机序", "AB 最优序", "AB 最差序")}
        for _ in range(30):
            t = p3.annotate_players(random_tree(b, d, rng))
            c = p3.Counter(); p3.minimax(t, True, c); acc["Minimax"].append(c.leaves)
            c = p3.Counter(); p3.alpha_beta(t, cnt=c); acc["AB 随机序"].append(c.leaves)
            c = p3.Counter(); p3.alpha_beta(p3.reorder(t, "best"), cnt=c); acc["AB 最优序"].append(c.leaves)
            c = p3.Counter(); p3.alpha_beta(p3.reorder(t, "worst"), cnt=c); acc["AB 最差序"].append(c.leaves)
        theory = b ** ((d + 1) // 2) + b ** (d // 2) - 1
        rand[f"b={b},d={d}"] = {**{k: statistics.mean(v) for k, v in acc.items()}, "理论最优": theory}
    out["random"] = rand
    return out


def exp_treasure():
    D = p4.distance_matrix()
    out = {"matrix": D, "runs": [], "by_H": [], "pv": {}}
    for H in (2, 4):
        for method, order, tt in (("minimax", "default", False), ("alphabeta", "default", False),
                                  ("alphabeta", "reverse", False), ("alphabeta", "gain", False),
                                  ("alphabeta", "worst", False), ("alphabeta", "gain", True)):
            res, st, cv = p4.best_first_move(H, method, order, tt, D)
            out["runs"].append({"H": H, "method": method, "order": order, "tt": tt, **res,
                                "nodes": st.nodes, "leaves": st.leaves, "cutoffs": st.cutoffs,
                                "tt_hits": st.tt_hits, "child_vals": cv})
        out["pv"][H] = p4.principal_variation(H, D)
    for H in (1, 2, 3, 4):
        res, st, cv = p4.best_first_move(H, "minimax", D=D)
        out["by_H"].append({"H": H, **res, "child_vals": cv})
    out["extension"] = exp_treasure_scaled()
    return out


def exp_treasure_scaled():
    rng = random.Random(SEED)
    results = {}
    saved = (p4.VALUES, p4.NAMES)
    for seed_i in range(5):
        n = 15
        while True:
            grid = [[('X' if rng.random() < 0.2 else str(rng.choice([1, 1, 1, 2, 3, 5]))) for _ in range(n)] for _ in range(n)]
            cells = rng.sample([(r, c) for r in range(n) for c in range(n)], 10)
            names = ['A', 'B'] + [f"T{i}" for i in range(1, 9)]
            for nm, (r, c) in zip(names, cells):
                grid[r][c] = nm
            p4.NAMES = names
            p4.VALUES = {f"T{i}": rng.randint(5, 20) for i in range(1, 9)}
            D = p4.distance_matrix(grid)
            if all(D[a][b] >= 0 for a in names for b in names):
                break
        s0 = p4.GameState('A', 'B', tuple(f"T{i}" for i in range(1, 9)), 0, 0, True)
        for H in range(2, 9):
            for label, method, order, tt in (("Minimax", "minimax", "default", False),
                                             ("AB 默认序", "alphabeta", "default", False),
                                             ("AB 收益排序", "alphabeta", "gain", False),
                                             ("AB 收益排序+置换表", "alphabeta", "gain", True)):
                st = p4.Stats()
                t = time.perf_counter()
                if method == "minimax":
                    v = p4.minimax(s0, H, D, st, order)
                else:
                    v = p4.alphabeta(s0, H, float("-inf"), float("inf"), D, st, order, {} if tt else None)
                ms = (time.perf_counter() - t) * 1000
                results.setdefault(H, {}).setdefault(label, {"nodes": [], "ms": [], "value": []})
                results[H][label]["nodes"].append(st.nodes)
                results[H][label]["ms"].append(ms)
                results[H][label]["value"].append(v)
    p4.VALUES, p4.NAMES = saved
    summary = {}
    for H, d in results.items():
        vals = {tuple(x["value"]) for x in d.values()}
        summary[H] = {k: {"nodes": statistics.mean(x["nodes"]), "ms": statistics.mean(x["ms"])} for k, x in d.items()}
        summary[H]["values_agree"] = len(vals) == 1
    return summary


def setup_font():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "SimHei",
                                       "Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9
    return plt


def draw_maze(ax, grid, info, title):
    import matplotlib.patches as patches
    n, m = len(grid), len(grid[0])
    path = set(info["path"])
    for r in range(n):
        for c in range(m):
            if grid[r][c] == 'X':
                col = "#444444"
            elif (r, c) in path:
                col = "#f4a261"
            elif (r, c) in info["closed"]:
                col = "#a8dadc"
            else:
                col = "white"
            ax.add_patch(patches.Rectangle((c, n - 1 - r), 1, 1, facecolor=col, edgecolor="#999999"))
    for idx, (r, c) in enumerate(info["order"]):
        ax.text(c + 0.5, n - 0.5 - r, str(idx + 1), ha="center", va="center", fontsize=7)
    ax.set_xlim(0, m); ax.set_ylim(0, n); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=9)


def make_figures(maze, info0, infoh, terrain, tree, treasure):
    plt = setup_font()
    os.makedirs(FIG, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    draw_maze(axes[0], p1.TEST_GRID, info0, f"(a) h=0：扩展 {info0['expanded']} 个节点")
    draw_maze(axes[1], p1.TEST_GRID, infoh, f"(b) 曼哈顿：扩展 {infoh['expanded']} 个节点")
    sizes = sorted(maze["random"].keys())
    for name, mk in (("h=0", "o-"), ("曼哈顿", "s-"), ("加权A* w=2", "^-"), ("贪心 w=100", "d--")):
        axes[2].plot(sizes, [maze["random"][n][name]["expanded"] for n in sizes], mk, label=name, ms=4)
    axes[2].set_xlabel("迷宫边长 n（障碍密度 0.3）"); axes[2].set_ylabel("平均扩展节点数")
    axes[2].set_yscale("log"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    axes[2].set_title("(c) 随机迷宫：启发函数对扩展节点数的影响", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_astar.png"), dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    keys = list(tree["random"].keys())
    series = ["Minimax", "AB 最差序", "AB 随机序", "AB 最优序", "理论最优"]
    w = 0.16
    for i, s in enumerate(series):
        axes[0].bar([k + (i - 2) * w for k in range(len(keys))], [tree["random"][k][s] for k in keys], w, label=s)
    axes[0].set_xticks(range(len(keys))); axes[0].set_xticklabels(keys, fontsize=8)
    axes[0].set_yscale("log"); axes[0].set_ylabel("评估叶子数"); axes[0].legend(fontsize=7)
    axes[0].set_title("(a) 随机博弈树：访问顺序与剪枝", fontsize=9)
    labels, n2, n4 = [], [], []
    for r in treasure["runs"]:
        if r["tt"]:
            continue
        if r["H"] == 2:
            lab = "Minimax" if r["method"] == "minimax" else {"default": "AB默认", "reverse": "AB逆序", "gain": "AB收益序", "worst": "AB最差序"}[r["order"]] + ("+TT" if r["tt"] else "")
            labels.append(lab); n2.append(r["nodes"])
        else:
            n4.append(r["nodes"])
    x = range(len(labels))
    axes[1].bar([i - 0.2 for i in x], n2, 0.4, label="H=2")
    axes[1].bar([i + 0.2 for i in x], n4, 0.4, label="H=4")
    for i, v in enumerate(n4):
        axes[1].text(i + 0.2, v + 1, str(v), ha="center", fontsize=7)
    axes[1].set_xticks(list(x)); axes[1].set_xticklabels(labels, rotation=30, fontsize=7)
    axes[1].set_ylabel("访问状态数"); axes[1].legend(fontsize=8)
    axes[1].set_title("(b) 题目四：方法与访问顺序", fontsize=9)
    ext = treasure["extension"]
    Hs = sorted(ext.keys())
    for name, mk in (("Minimax", "o-"), ("AB 默认序", "s-"), ("AB 收益排序", "^-"), ("AB 收益排序+置换表", "d-")):
        axes[2].plot(Hs, [ext[h][name]["nodes"] for h in Hs], mk, label=name, ms=4)
    axes[2].set_yscale("log"); axes[2].set_xlabel("搜索深度 H"); axes[2].set_ylabel("访问状态数")
    axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)
    axes[2].set_title("(c) 拓展：8 宝藏 15×15 地图", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_pruning.png"), dpi=200); plt.close(fig)


def run_all(make_plots=True):
    maze, info0, infoh = exp_maze()
    terrain = exp_terrain()
    tree = exp_tree()
    treasure = exp_treasure()
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"maze": maze, "terrain": terrain, "tree": tree, "treasure": treasure},
                  f, ensure_ascii=False, indent=1, default=str)
    if make_plots:
        make_figures(maze, info0, infoh, terrain, tree, treasure)
    return maze, terrain, tree, treasure


if __name__ == "__main__":
    run_all()
    print("结果已写入 results/results.json，图表已保存至 figures/")
