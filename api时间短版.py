# api.py
"""
提供 run_mind_evolution(...) 生成器：
- 接受来自 Flask 的 prompt 与 GA 参数
- 可选接受 distance_matrix / time_windows / vehicle_capacity 等
- 迭代中实时 yield 结构化日志，供 /stream SSE 推送
- 本版：LLM 调用改为本地 ollama（默认模型可用环境变量 OLLAMA_MODEL 覆盖）
"""

import os
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Generator, Tuple

# =============== 新增：本地 LLM（Ollama）调用 ===============
import ollama

# 默认本地模型名，可用环境变量覆盖：export OLLAMA_MODEL="qwen2.5:14b"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")


def call_llm(messages: List[Dict[str, str]], *, json_mode: bool = True, temperature: float = 1.0) -> str:
    """
    使用本地 Ollama 聊天接口。失败则返回空串（触发无模型回退）。
    messages: 形如 [{'role':'system','content':'...'}, {'role':'user','content':'...'}]
    """
    try:
        # ollama 的 format='json' 会强制模型输出 JSON（对结构化初始方案很有帮助）
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=False,
            format="json" if json_mode else "",
            options={"temperature": temperature},
        )
        return resp["message"]["content"]
    except Exception as e:
        print("[LLM 调用失败，使用回退]:", e)
        return ""

# ===================== 默认 VRP 数据（可被调用方覆盖） =====================
DEFAULT_N = 25  # 节点数（含配送中心0）
DEFAULT_VEHICLE_CAPACITY = 20
VEHICLE_FIXED_COST = 50.0
TRANSPORT_COST_PER_KM = 2.2
AVG_SPEED = 30 / 60  # km/min

A_PENALTY, B_PENALTY, C_PENALTY, D_PENALTY = 0.1, 0.05, 0.2, 0.5

DEFAULT_TIME_WINDOWS: Dict[int, Tuple[float, float, float, float, float]] = {
    0: (0, 1e6, 0, 1e6, 0),
    1: (60, 90, 45, 105, 2.45), 2: (30, 60, 15, 75, 1.00), 3: (90, 120, 75, 135, 1.68),
    4: (90, 120, 75, 135, 1.54), 5: (0, 30, -15, 45, 2.37), 6: (60, 90, 45, 105, 1.70),
    7: (90, 120, 75, 135, 1.05), 8: (90, 120, 75, 135, 2.75), 9: (90, 120, 75, 135, 1.86),
    10: (30, 60, 15, 75, 1.50), 11: (90, 120, 75, 135, 2.55), 12: (60, 90, 45, 105, 1.78),
    13: (90, 120, 75, 135, 1.23), 14: (90, 120, 75, 135, 1.49), 15: (60, 90, 45, 105, 2.30),
    16: (30, 60, 15, 75, 1.90), 17: (90, 120, 75, 135, 2.05), 18: (90, 120, 75, 135, 1.72),
    19: (90, 120, 75, 135, 1.18), 20: (90, 120, 75, 135, 1.55), 21: (30, 60, 15, 75, 2.65),
    22: (90, 120, 75, 135, 2.40), 23: (90, 120, 75, 135, 1.13), 24: (90, 120, 75, 135, 1.78),
}

DEFAULT_DISTANCE_MATRIX: List[List[float]] = [
    [0.00, 0.46, 2.10, 2.40, 2.90, 0.88, 1.40, 2.60, 1.30, 1.70, 3.00, 2.00, 1.20, 2.60, 2.90, 2.10, 1.20, 2.00, 1.50, 1.60, 1.90, 1.70, 1.50, 1.50, 2.70],
    [0.46, 0.00, 2.10, 2.20, 2.90, 0.43, 1.60, 2.50, 1.20, 1.60, 3.20, 2.20, 1.40, 2.80, 3.00, 2.20, 0.70, 2.20, 1.70, 1.40, 1.80, 1.70, 1.40, 1.30, 2.50],
    [2.10, 2.10, 0.00, 2.70, 0.17, 1.50, 3.10, 1.90, 1.50, 2.40, 4.40, 2.50, 2.60, 2.20, 2.30, 1.80, 1.50, 3.20, 2.90, 2.50, 1.60, 1.10, 1.60, 1.10, 3.00],
    [2.40, 2.20, 2.70, 0.00, 3.50, 2.40, 3.00, 4.70, 2.30, 1.50, 5.10, 4.10, 3.30, 4.70, 5.10, 4.30, 3.00, 4.10, 3.60, 1.80, 1.80, 3.90, 1.80, 2.40, 0.84],
    [2.90, 2.90, 0.17, 3.50, 0.00, 1.90, 3.20, 3.50, 1.60, 2.50, 6.00, 4.20, 4.20, 3.60, 2.40, 1.80, 1.40, 4.70, 4.50, 3.30, 2.00, 2.60, 2.40, 1.20, 3.20],
    [0.88, 0.43, 1.50, 2.40, 1.90, 0.00, 2.00, 2.40, 0.82, 1.70, 3.60, 2.00, 1.80, 2.40, 2.90, 2.10, 0.56, 2.70, 2.10, 1.50, 1.40, 1.60, 1.40, 0.97, 2.60],
    [1.40, 1.60, 3.10, 3.00, 3.20, 2.00, 0.00, 3.60, 1.90, 2.30, 1.30, 3.00, 0.99, 3.60, 4.00, 3.20, 2.30, 3.00, 1.60, 1.20, 2.40, 2.70, 2.10, 2.10, 2.10],
    [2.60, 2.50, 1.90, 4.70, 3.50, 2.40, 3.60, 0.00, 3.90, 4.30, 3.80, 0.67, 2.60, 0.46, 2.70, 1.60, 2.50, 1.20, 2.20, 4.10, 3.40, 0.95, 4.00, 2.90, 5.20],
    [1.30, 1.20, 1.50, 2.30, 1.60, 0.82, 1.90, 3.90, 0.00, 1.50, 4.00, 3.00, 2.20, 2.90, 3.40, 2.60, 1.60, 3.10, 2.50, 1.40, 0.85, 1.90, 0.84, 0.42, 2.50],
    [1.70, 1.60, 2.40, 1.50, 2.50, 1.70, 2.30, 4.30, 1.50, 0.00, 4.40, 3.40, 2.60, 4.00, 4.40, 3.60, 2.40, 3.40, 2.90, 1.20, 0.77, 2.80, 1.10, 1.30, 1.60],
    [3.00, 3.20, 4.40, 5.10, 6.00, 3.60, 1.30, 3.80, 4.00, 4.40, 0.00, 2.90, 1.10, 3.40, 4.10, 3.30, 2.40, 2.90, 1.50, 1.80, 3.10, 2.80, 2.70, 2.70, 2.70],
    [2.00, 2.20, 2.50, 4.10, 4.20, 2.00, 3.00, 0.67, 3.00, 3.40, 2.90, 0.00, 2.00, 0.10, 3.30, 2.20, 2.00, 0.53, 1.60, 3.20, 3.50, 1.50, 3.10, 3.10, 4.30],
    [1.20, 1.40, 2.60, 3.30, 4.20, 1.80, 0.99, 2.60, 2.20, 2.60, 1.10, 2.00, 0.00, 2.50, 3.50, 2.70, 1.80, 2.10, 0.64, 1.70, 2.60, 2.20, 2.20, 2.10, 2.80],
    [2.60, 2.80, 2.20, 4.70, 3.60, 2.40, 3.60, 0.46, 2.90, 4.00, 3.40, 0.10, 2.50, 0.00, 2.90, 1.90, 2.30, 0.62, 2.00, 3.50, 3.80, 1.20, 3.50, 3.40, 4.60],
    [2.90, 3.00, 2.30, 5.10, 2.40, 2.90, 4.00, 2.70, 3.40, 4.40, 4.10, 3.30, 3.50, 2.90, 0.00, 0.98, 2.60, 3.60, 3.70, 3.90, 3.10, 1.90, 3.10, 2.50, 4.90],
    [2.10, 2.20, 1.80, 4.30, 1.80, 2.10, 3.20, 1.60, 2.60, 3.60, 3.30, 2.20, 2.70, 1.90, 0.98, 0.00, 2.10, 2.90, 3.00, 3.40, 2.70, 1.50, 2.70, 2.10, 4.50],
    [1.20, 0.70, 1.50, 3.00, 1.40, 0.56, 2.30, 2.50, 1.60, 2.40, 2.40, 2.00, 1.80, 2.30, 2.60, 2.10, 0.00, 2.90, 2.30, 2.10, 1.80, 1.20, 2.10, 1.30, 3.20],
    [2.00, 2.20, 3.20, 4.10, 4.70, 2.70, 3.00, 1.20, 3.10, 3.40, 2.90, 0.53, 2.10, 0.62, 3.60, 2.90, 2.90, 0.00, 1.80, 3.30, 3.70, 2.00, 3.30, 3.30, 4.40],
    [1.50, 1.70, 2.90, 3.60, 4.50, 2.10, 1.60, 2.20, 2.50, 2.90, 1.50, 1.60, 0.64, 2.00, 3.70, 3.00, 2.30, 1.80, 0.00, 2.00, 2.90, 2.60, 2.50, 2.50, 3.70],
    [1.60, 1.40, 2.50, 1.80, 3.30, 1.50, 1.20, 4.10, 1.40, 1.20, 1.80, 3.20, 1.70, 3.50, 3.90, 3.40, 2.10, 3.30, 2.00, 0.00, 1.30, 2.90, 0.92, 1.50, 1.10],
    [1.90, 1.80, 1.60, 1.80, 2.00, 1.40, 2.40, 3.40, 0.85, 0.77, 3.10, 3.50, 2.60, 3.80, 3.10, 2.70, 1.80, 3.70, 2.90, 1.30, 0.00, 2.10, 0.40, 0.59, 2.00],
    [1.70, 1.70, 1.10, 3.90, 2.60, 1.60, 2.70, 0.95, 1.90, 2.80, 2.80, 1.50, 2.20, 1.20, 1.90, 1.50, 1.20, 2.00, 2.60, 2.90, 2.10, 0.00, 3.60, 2.20, 4.80],
    [1.50, 1.40, 1.60, 1.80, 2.40, 1.40, 2.10, 4.00, 0.84, 1.10, 2.70, 3.10, 2.20, 3.50, 3.10, 2.70, 2.10, 3.30, 2.50, 0.92, 0.40, 3.60, 0.00, 0.61, 2.00],
    [1.50, 1.30, 1.10, 2.40, 1.20, 0.97, 2.10, 2.90, 0.42, 1.30, 2.70, 3.10, 2.10, 3.40, 2.50, 2.10, 1.30, 3.30, 2.50, 1.50, 0.59, 2.20, 0.61, 0.00, 2.60],
    [2.70, 2.50, 3.00, 0.84, 3.20, 2.60, 2.10, 5.20, 2.50, 1.60, 2.70, 4.30, 2.80, 4.60, 4.90, 4.50, 3.20, 4.40, 3.70, 1.10, 2.00, 4.80, 2.00, 2.60, 0.00],
]

# ===================== 基础结构 =====================
@dataclass
class Individual:
    plan: str  # JSON string: {"routes": [[0,...,0], ...]}
    fitness: float = float('inf')
    feedback: List[str] = field(default_factory=list)

class Evaluator:
    def __init__(self, dist: List[List[float]], tw: Dict[int, Tuple[float, float, float, float, float]],
                 vehicle_capacity: float):
        self.D = dist
        self.TW = tw
        self.cap = vehicle_capacity

    def _time_penalty(self, arrival: float, tw: Tuple[float, float, float, float, float]) -> float:
        start, end, acc_start, acc_end, _demand = tw
        if acc_start <= arrival <= acc_end:
            if start <= arrival <= end: return 0.0
            elif acc_start <= arrival < start: return B_PENALTY * (start - arrival)
            else: return C_PENALTY * (arrival - end)
        elif arrival < acc_start:
            return A_PENALTY * (acc_start - arrival) + B_PENALTY * (start - acc_start)
        else:
            return C_PENALTY * (acc_end - end) + D_PENALTY * (arrival - acc_end)

    def evaluate(self, plan: str) -> Tuple[float, List[str]]:
        feedback: List[str] = []
        try:
            routes = json.loads(plan)["routes"]
        except Exception:
            return float('inf'), ["方案 JSON 解析失败或缺少 routes"]

        # 硬约束：覆盖所有客户 1..N-1；每条路线 0 起 0 回；容量
        N = len(self.D)
        all_customers = set(range(1, N))
        visited = set()

        for idx, r in enumerate(routes, 1):
            if not r or r[0] != 0 or r[-1] != 0:
                return float('inf'), [f"路线 {idx} 必须从 0 出发并回到 0"]

            # 容量
            load = sum(self.TW.get(n, (0,0,0,0,0))[4] for n in r if n != 0)
            if load > self.cap:
                return float('inf'), [f"路线 {idx} 负载 {load:.2f} 超过容量 {self.cap}"]

            visited.update([n for n in r if n != 0])

        if visited != all_customers:
            missing = list(all_customers - visited)
            return float('inf'), [f"缺少客户: {missing}"]

        # 成本
        total_cost = len(routes) * VEHICLE_FIXED_COST
        for r in routes:
            dist = sum(self.D[r[i]][r[i+1]] for i in range(len(r)-1))
            total_cost += dist * TRANSPORT_COST_PER_KM

            t = 0.0
            for i in range(1, len(r)):
                t += self.D[r[i-1]][r[i]] / AVG_SPEED
                if r[i] != 0:
                    total_cost += self._time_penalty(t, self.TW[r[i]])

        return total_cost, ["方案有效"]

# ===================== 简单生成/操作（含无模型回退） =====================
def random_valid_partition(n_customers: int, cap: float, tw: Dict[int, Tuple[float,float,float,float,float]]) -> List[List[int]]:
    """把 1..n_customers 随机分桶，每桶需求不超 cap"""
    customers = list(range(1, n_customers + 1))
    random.shuffle(customers)
    buckets: List[List[int]] = []
    cur, cur_load = [], 0.0
    for c in customers:
        d = tw[c][4]
        if cur_load + d > cap and cur:
            buckets.append(cur)
            cur, cur_load = [], 0.0
        cur.append(c)
        cur_load += d
    if cur:
        buckets.append(cur)
    return buckets

def routeify(buckets: List[List[int]]) -> List[List[int]]:
    """每个桶前后加 0，桶内随机邻近交换几次"""
    routes = []
    for b in buckets:
        seq = b[:]
        for _ in range(random.randint(0, 2)):
            if len(seq) >= 2:
                i, j = random.sample(range(len(seq)), 2)
                seq[i], seq[j] = seq[j], seq[i]
        routes.append([0] + seq + [0])
    return routes

def mutate_routes(routes: List[List[int]], cap: float, tw: Dict[int, Tuple[float,float,float,float,float]]) -> List[List[int]]:
    """轻量变异：跨路交换或同路两点互换"""
    new = json.loads(json.dumps(routes))
    if not new:
        return new
    if random.random() < 0.5 and len(new) >= 2:
        a, b = random.sample(range(len(new)), 2)
        ra = [x for x in new[a] if x != 0]
        rb = [x for x in new[b] if x != 0]
        if ra and rb:
            ia, ib = random.randrange(len(ra)), random.randrange(len(rb))
            ra[ia], rb[ib] = rb[ib], ra[ia]
            new[a] = [0] + ra + [0]
            new[b] = [0] + rb + [0]
    else:
        ridx = random.randrange(len(new))
        r = [x for x in new[ridx] if x != 0]
        if len(r) >= 2:
            i, j = random.sample(range(len(r)), 2)
            r[i], r[j] = r[j], r[i]
            new[ridx] = [0] + r + [0]
    # 简单容量修正（如超容量则随机把一个客户搬到新车）
    for i, r in enumerate(new):
        load = sum(tw.get(n, (0,0,0,0,0))[4] for n in r if n != 0)
        if load > cap:
            customers = [n for n in r if n != 0]
            if customers:
                move = random.choice(customers)
                r2 = [0, move, 0]
                new[i] = [0] + [n for n in customers if n != move] + [0]
                new.append(r2)
    return new

# ===================== 主流程（生成器） =====================
def run_mind_evolution(
    prompt: str,
    population_size: int = 8,
    generations: int = 5,
    num_parents: int = 5,
    tournament_size: int = 5,
    max_retries: int = 3,
    *,
    distance_matrix: Optional[List[List[float]]] = None,
    time_windows: Optional[Dict[int, Tuple[float,float,float,float,float]]] = None,
    vehicle_capacity: Optional[float] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    生成器：按步骤产出实时日志/指标，供 Flask SSE 推送。
    返回的每个 step 都是字典，常见字段：
    - phase: "init" | "evolve" | "final"
    - gen: 当前代（从 0 开始）
    - best_fitness, best_plan
    - message: 文本日志
    """

    D = distance_matrix or DEFAULT_DISTANCE_MATRIX
    TW = time_windows or DEFAULT_TIME_WINDOWS
    N = len(D)
    CAP = float(vehicle_capacity if vehicle_capacity is not None else DEFAULT_VEHICLE_CAPACITY)

    evaluator = Evaluator(D, TW, CAP)

    # ===== 初始化种群 =====
    yield {"phase": "init", "message": "初始化种群中..."}
    population: List[Individual] = []

    # 优先调用本地 LLM 生成若干方案；若失败，使用“无模型回退”随机生成
    for attempt in range(max_retries):
        messages = [
            {"role": "system", "content": "你是物流优化助手。严格输出 JSON。"},
            {"role": "user", "content": f"""
请基于用户问题（可忽略不必要细节）给出 {population_size} 个 VRPTW 可行方案。
必须严格覆盖客户 1..{N-1}，每条路线从 0 出发并回到 0，且单车载荷不超过 {CAP}。
仅输出 JSON：
{{
  "solutions": [
    {{"routes": [[0, 1, 5, 0], [0, 2, 3, 4, 0]]}}
  ]
}}
用户问题：{prompt or "（无）"}
"""}
        ]
        content = call_llm(messages, json_mode=True, temperature=1.0)

        routes_list: List[List[List[int]]] = []
        if content:
            try:
                data = json.loads(content)
                for sol in data.get("solutions", []):
                    routes_list.append(sol["routes"])
            except Exception:
                routes_list = []

        if not routes_list:
            # 无模型回退：随机生成
            for _ in range(population_size):
                buckets = random_valid_partition(N-1, CAP, TW)
                routes_list.append(routeify(buckets))

        population = []
        for routes in routes_list[:population_size]:
            plan_str = json.dumps({"routes": routes}, ensure_ascii=False)
            fit, fb = evaluator.evaluate(plan_str)
            population.append(Individual(plan=plan_str, fitness=fit, feedback=fb))

        # 若存在有效个体则初始化成功
        if any(ind.fitness != float('inf') for ind in population):
            break

    best = min(population, key=lambda x: x.fitness if math.isfinite(x.fitness) else 1e30)
    yield {
        "phase": "init",
        "best_fitness": best.fitness,
        "best_plan": json.loads(best.plan),
        "message": f"初始化完成。当前最优 {best.fitness:.2f}"
    }

    # ===== 进化 =====
    def tournament_select(k: int) -> List[Individual]:
        valid = [p for p in population if math.isfinite(p.fitness)]
        src = valid if valid else population
        if not src:
            return []
        pool = src[:]
        parents: List[Individual] = []
        while pool and len(parents) < k:
            tsize = min(tournament_size, len(pool))
            cand = random.sample(pool, tsize)
            winner = min(cand, key=lambda x: x.fitness)
            parents.append(winner)
            pool.remove(winner)
        return parents

    for g in range(generations):
        yield {"phase": "evolve", "gen": g, "message": f"第 {g+1}/{generations} 代开始"}
        new_pop: List[Individual] = [best]  # 精英保留

        while len(new_pop) < population_size:
            parents = tournament_select(num_parents)
            if parents:
                base_routes = json.loads(min(parents, key=lambda x: x.fitness).plan)["routes"]
            else:
                base_routes = json.loads(best.plan)["routes"]

            child_routes = mutate_routes(base_routes, CAP, TW)
            child_plan = json.dumps({"routes": child_routes}, ensure_ascii=False)
            fit, fb = evaluator.evaluate(child_plan)
            new_pop.append(Individual(plan=child_plan, fitness=fit, feedback=fb))

            yield {
                "phase": "evolve",
                "gen": g,
                "message": f"  生成候选个体，成本 {fit:.2f}，状态：{';'.join(fb)}"
            }

        population = new_pop
        cur_best = min(population, key=lambda x: x.fitness)
        if cur_best.fitness < best.fitness:
            best = cur_best

        yield {
            "phase": "evolve",
            "gen": g,
            "best_fitness": best.fitness,
            "best_plan": json.loads(best.plan),
            "message": f"第 {g+1} 代结束；本代最佳 {cur_best.fitness:.2f}，全局最佳 {best.fitness:.2f}"
        }

    # ===== 结束 =====
    yield {
        "phase": "final",
        "best_fitness": best.fitness,
        "best_plan": json.loads(best.plan),
        "message": f"进化完成，最优成本 {best.fitness:.2f}"
    }
