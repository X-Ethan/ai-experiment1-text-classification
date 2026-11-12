import json
from typing import List, Optional


class TreeNode:
    def __init__(self, value=None, children=None, is_max: Optional[bool] = None):
        self.value = value
        self.children = children if children is not None else []
        self.is_max = is_max


class Counter:
    def __init__(self):
        self.visited = 0
        self.leaves = 0
        self.cutoffs = 0


def _player(node: TreeNode, default_max: bool) -> bool:
    return default_max if node.is_max is None else node.is_max


def minimax(node: TreeNode, is_max: bool = True, cnt: Optional[Counter] = None):
    if cnt is not None:
        cnt.visited += 1
    if not node.children:
        if cnt is not None:
            cnt.leaves += 1
        return node.value
    is_max = _player(node, is_max)
    vals = [minimax(ch, not is_max, cnt) for ch in node.children]
    return max(vals) if is_max else min(vals)


def alpha_beta(node: TreeNode, alpha=float("-inf"), beta=float("inf"),
               is_max: bool = True, cnt: Optional[Counter] = None):
    if cnt is not None:
        cnt.visited += 1
    if not node.children:
        if cnt is not None:
            cnt.leaves += 1
        return node.value
    is_max = _player(node, is_max)
    if is_max:
        best = float("-inf")
        for i, ch in enumerate(node.children):
            best = max(best, alpha_beta(ch, alpha, beta, False, cnt))
            alpha = max(alpha, best)
            if alpha >= beta:
                if cnt is not None and i < len(node.children) - 1:
                    cnt.cutoffs += 1
                break
        return best
    best = float("inf")
    for i, ch in enumerate(node.children):
        best = min(best, alpha_beta(ch, alpha, beta, True, cnt))
        beta = min(beta, best)
        if alpha >= beta:
            if cnt is not None and i < len(node.children) - 1:
                cnt.cutoffs += 1
            break
    return best


def best_score(root: TreeNode) -> int:
    return alpha_beta(root)


def from_dict(d: dict) -> TreeNode:
    if "children" not in d or not d["children"]:
        return TreeNode(value=d.get("value"))
    return TreeNode(children=[from_dict(c) for c in d["children"]],
                    is_max=d.get("is_maximizing_player"))


def from_json(text: str) -> TreeNode:
    data = json.loads(text)
    return from_dict(data.get("tree_structure", data))


def build_example() -> TreeNode:
    leaf1, leaf2, leaf3 = TreeNode(value=3), TreeNode(value=5), TreeNode(value=6)
    leaf4, leaf5, leaf6 = TreeNode(value=9), TreeNode(value=1), TreeNode(value=2)
    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])
    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])
    return TreeNode(children=[min1, min2])


TEST_JSON = """{"tree_structure": {"is_maximizing_player": true, "children": [
  {"is_maximizing_player": false, "children": [
     {"is_maximizing_player": true, "children": [{"value": 3}, {"value": 5}]},
     {"is_maximizing_player": true, "children": [{"value": 6}, {"value": 9}]}]},
  {"is_maximizing_player": false, "children": [
     {"is_maximizing_player": true, "children": [{"value": 6}, {"value": 9}]},
     {"is_maximizing_player": true, "children": [{"value": 1}, {"value": 2}]}]}]}}"""


def reorder(node: TreeNode, mode: str) -> TreeNode:
    if not node.children:
        return TreeNode(value=node.value)
    kids = [reorder(c, mode) for c in node.children]
    if mode == "right":
        kids = kids[::-1]
    elif mode in ("best", "worst"):
        is_max = node.is_max if node.is_max is not None else True
        keyed = [(minimax(k, not is_max), i, k) for i, k in enumerate(kids)]
        rev = is_max if mode == "best" else not is_max
        keyed.sort(key=lambda t: (t[0], t[1]), reverse=rev)
        kids = [k for _, _, k in keyed]
    return TreeNode(children=kids, is_max=node.is_max)


def annotate_players(node: TreeNode, is_max: bool = True) -> TreeNode:
    if node.children:
        node.is_max = is_max if node.is_max is None else node.is_max
        for c in node.children:
            annotate_players(c, not node.is_max)
    return node


if __name__ == "__main__":
    print(best_score(build_example()), best_score(from_json(TEST_JSON)))
