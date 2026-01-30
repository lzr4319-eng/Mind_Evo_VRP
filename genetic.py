#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSP solver using Held–Karp dynamic programming.
- 输入示例：
  "有路径：A(50, 92), B(63, 15), C(85, 50), D(102, 85), E(25, 0), F(15, 37), G(0, 37)"
- 默认解 *闭合回路*（回到起点）。若要“开口路径”，设置 return_to_start=False。
- 起点固定为输入的第一个点。
复杂度：O(n^2 * 2^n)，适合几十个以内的点。
"""

import re
from math import hypot
from typing import Dict, Tuple, List

Point = Tuple[float, float]

# 正则匹配坐标，例如 A(1,2) 或 A（1, 2）
LABEL_COORD_RE = re.compile(
    r'([A-Za-z0-9_\u4e00-\u9fff]+)\s*[\(（]\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*[\)）]'
)

def parse_points(text: str) -> Dict[str, Point]:
    """解析 'A(1,2), B(3,4)' 样式文本为 {label: (x, y)}"""
    matches = LABEL_COORD_RE.findall(text)
    if not matches:
        raise ValueError("未识别到任何坐标，请检查输入格式，如：A(1, 2), B(3, 4)")
    pts: Dict[str, Point] = {}
    for label, x, y in matches:
        if label in pts:
            raise ValueError(f"重复的标签: {label}")
        pts[label] = (float(x), float(y))
    return pts

def euclid(a: Point, b: Point) -> float:
    """欧氏距离"""
    return hypot(a[0] - b[0], a[1] - b[1])

def held_karp_closed(points: Dict[str, Point]) -> Tuple[List[str], float, List[Tuple[str, str, float]]]:
    """Held–Karp 动态规划算法：闭合回路"""
    labels = list(points.keys())
    n = len(labels)
    if n <= 1:
        return labels + (labels[:1] if n == 1 else []), 0.0, []

    start_label = labels[0]
    others = labels[1:]
    m = len(others)

    idx_to_label = {i: others[i] for i in range(m)}
    D = {(i, j): euclid(points[i], points[j]) for i in labels for j in labels if i != j}

    INF = float("inf")
    DP = {}
    parent = {}

    # 初始化
    for j in range(m):
        S = 1 << j
        lab_j = idx_to_label[j]
        DP[(S, j)] = D[(start_label, lab_j)]
        parent[(S, j)] = None

    # 动态规划
    for S in range(1, 1 << m):
        if S.bit_count() <= 1:
            continue
        for j in range(m):
            if not (S & (1 << j)):
                continue
            best = INF
            best_k = None
            Sj = S ^ (1 << j)
            lab_j = idx_to_label[j]
            for k in range(m):
                if not (Sj & (1 << k)):
                    continue
                lab_k = idx_to_label[k]
                cost = DP[(Sj, k)] + D[(lab_k, lab_j)]
                if cost < best:
                    best = cost
                    best_k = k
            DP[(S, j)] = best
            parent[(S, j)] = best_k

    # 闭合路径，返回起点
    ALL = (1 << m) - 1
    best_total = INF
    best_end = None
    for j in range(m):
        lab_j = idx_to_label[j]
        total = DP[(ALL, j)] + D[(lab_j, start_label)]
        if total < best_total:
            best_total = total
            best_end = j

    # 回溯路径
    order = []
    S = ALL
    j = best_end
    while j is not None:
        order.append(j)
        pj = parent[(S, j)]
        if pj is None:
            break
        S = S ^ (1 << j)
        j = pj

    order.reverse()
    middle_labels = [idx_to_label[i] for i in order]
    path = [start_label] + middle_labels + [start_label]

    # 分段距离
    legs = [(a, b, D[(a, b)]) for a, b in zip(path[:-1], path[1:])]
    return path, best_total, legs

def held_karp_open(points: Dict[str, Point]) -> Tuple[List[str], float, List[Tuple[str, str, float]]]:
    """Held–Karp 动态规划算法：开口路径"""
    labels = list(points.keys())
    n = len(labels)
    if n <= 1:
        return labels, 0.0, []

    start_label = labels[0]
    others = labels[1:]
    m = len(others)
    idx_to_label = {i: others[i] for i in range(m)}
    D = {(i, j): euclid(points[i], points[j]) for i in labels for j in labels if i != j}

    INF = float("inf")
    DP = {}
    parent = {}

    # 初始化
    for j in range(m):
        S = 1 << j
        lab_j = idx_to_label[j]
        DP[(S, j)] = D[(start_label, lab_j)]
        parent[(S, j)] = None

    # 动态规划
    for S in range(1, 1 << m):
        if S.bit_count() <= 1:
            continue
        for j in range(m):
            if not (S & (1 << j)):
                continue
            best = INF
            best_k = None
            Sj = S ^ (1 << j)
            lab_j = idx_to_label[j]
            for k in range(m):
                if not (Sj & (1 << k)):
                    continue
                lab_k = idx_to_label[k]
                cost = DP[(Sj, k)] + D[(lab_k, lab_j)]
                if cost < best:
                    best = cost
                    best_k = k
            DP[(S, j)] = best
            parent[(S, j)] = best_k

    ALL = (1 << m) - 1
    best_total = INF
    best_end = None
    for j in range(m):
        total = DP[(ALL, j)]
        if total < best_total:
            best_total = total
            best_end = j

    # 回溯路径
    order = []
    S = ALL
    j = best_end
    while j is not None:
        order.append(j)
        pj = parent[(S, j)]
        if pj is None:
            break
        S = S ^ (1 << j)
        j = pj

    order.reverse()
    path_nodes = [start_label] + [idx_to_label[i] for i in order]

    # 分段距离
    legs = [(a, b, D[(a, b)]) for a, b in zip(path_nodes[:-1], path_nodes[1:])]
    total = sum(d for _, _, d in legs)
    return path_nodes, total, legs

def solve_tsp(text: str, return_to_start: bool = True):
    """主入口：解析输入并求解"""
    pts = parse_points(text)
    if return_to_start:
        return held_karp_closed(pts)
    else:
        return held_karp_open(pts)

def format_report(path: List[str], total: float, legs: List[Tuple[str, str, float]], decimals: int = 3) -> str:
    """格式化输出"""
    if not path:
        return "没有路径。"
    arrow = " → "
    route_str = arrow.join(path)
    leg_lines = [f"- {a}→{b}: {d:.{decimals}f}" for a, b, d in legs]
    report = [
        f"最优路线：\n{route_str}",
        f"\n总长度：{total:.{decimals}f}",
        "\n分段距离：",
        *leg_lines
    ]
    return "\n".join(report)

if __name__ == "__main__":
    sample = "有路径：A(50, 92), B(63, 15), C(85, 50), D(12, 85), E(25, 0), F(15, 37), G(0, 37),H(60, 82), I(73, 25), J(85, 60)"
    path, total, legs = solve_tsp(sample, return_to_start=True)
    print(format_report(path, total, legs))
