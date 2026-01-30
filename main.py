# -*- coding: utf-8 -*-
import ollama
import json
import re
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ==============================================================================
# VRP问题数据
# ==============================================================================
N = 25  # 节点数（含配送中心0）
VEHICLE_CAPACITY = 20
VEHICLE_FIXED_COST = 50
TRANSPORT_COST_PER_KM = 2.2
AVG_SPEED = 30 / 60  # km/min

# 惩罚系数
A_PENALTY, B_PENALTY, C_PENALTY, D_PENALTY = 0.1, 0.05, 0.2, 0.5

# 时间窗 (start, end, accept_start, accept_end, demand)
# 0号点为配送中心
TIME_WINDOWS = {
    0: (0, 1e6, 0, 1e6, 0),
    1: (60, 90, 45, 105, 2.45), 2: (30, 60, 15, 75, 1.00), 3: (90, 120, 75, 135, 1.68),
    4: (90, 120, 75, 135, 1.54), 5: (0, 30, -15, 45, 2.37), 6: (60, 90, 45, 105, 1.70),
    7: (90, 120, 75, 135, 1.05), 8: (90, 120, 75, 135, 2.75), 9: (90, 120, 75, 135, 1.86),
    10: (30, 60, 15, 75, 1.50), 11: (90, 120, 75, 135, 2.55), 12: (60, 90, 45, 105, 1.78),
    13: (90, 120, 75, 135, 1.23), 14: (90, 120, 75, 135, 1.49), 15: (60, 90, 45, 105, 2.30),
    16: (30, 60, 15, 75, 1.90), 17: (90, 120, 75, 135, 2.05), 18: (90, 120, 75, 135, 1.72),
    19: (90, 120, 75, 135, 1.18), 20: (90, 120, 75, 135, 1.55), 21: (30, 60, 15, 75, 2.65),
    22: (90, 120, 75, 135, 2.40), 23: (90, 120, 75, 135, 1.13), 24: (90, 120, 75, 135, 1.78)
}

# 距离矩阵
DISTANCE_MATRIX = [
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
    [2.70, 2.50, 3.00, 0.84, 3.20, 2.60, 2.10, 5.20, 2.50, 1.60, 2.70, 4.30, 2.80, 4.60, 4.90, 4.50, 3.20, 4.40, 3.70, 1.10, 2.00, 4.80, 2.00, 2.60, 0.00]
]
# ==============================================================================
# Mind Evolution
# ==============================================================================
MODEL_NAME = 'qwen2.5:14b'
POPULATION_SIZE = 15
N_GENERATIONS = 8
N_PARENTS_FOR_RECOMBINATION = 5
TOURNAMENT_SIZE = 5
MAX_INIT_RETRIES = 3

@dataclass
class Individual:
    plan: str
    fitness: float = float('inf')
    feedback: List[str] = field(default_factory=list)

    def __str__(self):
        plan_str = self.plan.replace(" ", "").replace('"', "")
        return f"方案: {plan_str}, 总成本: {self.fitness:.2f}"

class Evaluator:
    def _time_penalty(self, arrival, tw):
        start, end, acc_start, acc_end, _ = tw
        if acc_start <= arrival <= acc_end:
            if start <= arrival <= end: return 0
            elif acc_start <= arrival < start: return B_PENALTY * (start - arrival)
            else: return C_PENALTY * (arrival - end)
        elif arrival < acc_start: return A_PENALTY * (acc_start - arrival) + B_PENALTY * (start - acc_start)
        else: return C_PENALTY * (acc_end - end) + D_PENALTY * (arrival - acc_end)

    def evaluate(self, plan: str) -> (float, List[str]): # type: ignore
        feedback = []
        try:
            plan_for_json = plan.replace("'", '"')
            routes_data = json.loads(plan_for_json)
            routes = routes_data['routes']
        except (json.JSONDecodeError, KeyError, TypeError):
            feedback.append("错误：方案不是有效的JSON格式，或缺少 'routes' 键。")
            return float('inf'), feedback

        # 硬约束检查
        customer_nodes = {i for i in range(1, N)}
        all_visited_customers = set()
        
        # 提取所有访问过的客户
        for i, route in enumerate(routes):
            # --- 新增的防御性代码 ---
            if not isinstance(route, list):
                feedback.append(f"错误：路线 {i+1} 的格式不是一个列表，而是一个 {type(route)}。")
                return float('inf'), feedback
            try:
                # 检查路线内部是否包含非整数或嵌套列表等无效类型
                if not all(isinstance(node, int) for node in route):
                    feedback.append(f"错误：路线 {i+1} 包含非整数节点。路线内容: {route}。")
                    return float('inf'), feedback
                route_customers = set(node for node in route if node != 0)
                all_visited_customers.update(route_customers)
            except TypeError:
                feedback.append(f"错误：路线 {i+1} 包含无效的数据类型（例如，嵌套列表），无法处理。路线内容: {route}")
                return float('inf'), feedback
            # --- 防御性代码结束 ---

        # 检查是否所有客户都被访问
        if all_visited_customers != customer_nodes:
            missing = customer_nodes - all_visited_customers
            feedback.append(f"错误：方案遗漏了客户: {missing}。")
            return float('inf'), feedback

        # 检查其他硬约束
        for i, route in enumerate(routes):
            if not route or route[0] != 0 or route[-1] != 0:
                feedback.append(f"错误：路线 {i+1} 必须从配送中心0出发并返回0。")
                return float('inf'), feedback
            
            route_demand = sum(TIME_WINDOWS.get(node, (0,0,0,0,0))[4] for node in route)
            if route_demand > VEHICLE_CAPACITY:
                feedback.append(f"错误：路线 {i+1} 的总需求 ({route_demand:.2f}) 超出车辆容量 ({VEHICLE_CAPACITY})。")
                return float('inf'), feedback
        
        # 如果所有硬约束都通过，再计算详细成本
        total_cost = len(routes) * VEHICLE_FIXED_COST
        for r in routes:
            dist = sum(DISTANCE_MATRIX[r[j]][r[j + 1]] for j in range(len(r) - 1))
            total_cost += dist * TRANSPORT_COST_PER_KM

            t = 0
            for i in range(1, len(r)):
                travel_time = DISTANCE_MATRIX[r[i - 1]][r[i]] / AVG_SPEED
                t += travel_time
                if r[i] != 0:
                    total_cost += self._time_penalty(t, TIME_WINDOWS[r[i]])
        
        feedback.append("方案有效。")
        return total_cost, feedback

class MindEvolution:
    def __init__(self):
        self.evaluator = Evaluator()
        self.population: List[Individual] = []
        self.best_solution_so_far: Individual = Individual("N/A")

    def _prompt_llm(self, messages: List[Dict[str, str]], json_mode: bool = True) -> str:
        try:
            options = {"temperature": 1.0} # 鼓励多样性
            response = ollama.chat(
                model=MODEL_NAME, messages=messages, stream=False, format='json' if json_mode else '', options=options)
            return response['message']['content']
        except Exception as e:
            print(f"\n[LLM 调用错误]: {e}")
            return ""

    def initialize_population(self):
        print(f"--- 1. 初始化种群 (正在调用 {MODEL_NAME} 生成 {POPULATION_SIZE} 个初始方案) ---")
        
        customer_info = f"共 {N-1} 个客户(1-{N-1})。车辆容量上限: {VEHICLE_CAPACITY}。"
        all_customers_list = list(range(1, N))

        prompt_content = f"""
        为VRPTW问题生成多个不同的、但必须【完全有效】的配送方案。

        问题定义:
        - 客户点: {customer_info}
        - 总客户列表: {all_customers_list}
        - 成本构成: 总成本 = (车辆数 * 固定成本) + (总公里数 * 运输成本) + (时间窗惩罚)。

        核心规则 (必须严格遵守):
        1.  所有路线从0出发返回0。
        2.  【容量约束】: 每条路线总需求不得超过{VEHICLE_CAPACITY}。
        3.  【完整性约束】: 所有客户 (1 到 {N-1}) 都必须被服务，且仅一次。

        思考与自检步骤 (强制执行):
        1.  分组: 创造一个客户分组方案，确保每组的总需求不超过{VEHICLE_CAPACITY}。
        2.  自检: 生成方案后，立即检查方案中的所有客户集合是否与【总客户列表】完全一致。如果不一致，必须重新生成。
        3.  输出: 只有通过自检的方案才能输出。

        任务: 请生成 {POPULATION_SIZE} 个不同的、完全有效的解决方案。
        
        输出格式 (严格遵守):
        {{
            "solutions": [
                {{"routes": [[0, 1, 5, 0], [0, 2, 3, 4, ... , 0]]}},
                {{"routes": [[0, 21, 10, 0], [0, ... , 0]]}}
            ]
        }}
        """

        messages = [{'role': 'system', 'content': '你是一个善于生成多样化、有效解决方案的物流规划AI，严格遵循JSON输出格式和自检要求。'}, {'role': 'user', 'content': prompt_content}]

        for attempt in range(MAX_INIT_RETRIES):
            response_str = self._prompt_llm(messages)
            try:
                data = json.loads(response_str)
                initial_routes_list = [sol['routes'] for sol in data['solutions']]
                self.population = []
                for routes in initial_routes_list:
                    plan_str = json.dumps({"routes": routes})
                    individual = Individual(plan=plan_str)
                    individual.fitness, individual.feedback = self.evaluator.evaluate(individual.plan)
                    self.population.append(individual)
                
                if any(ind.fitness != float('inf') for ind in self.population):
                    print(f"成功生成并评估了 {len(self.population)} 个初始方案 (尝试 {attempt+1}/{MAX_INIT_RETRIES})。")
                    return
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"初始化失败 (尝试 {attempt+1}/{MAX_INIT_RETRIES})：{e}")
        print(f"在 {MAX_INIT_RETRIES} 次尝试后，仍未能生成任何有效的初始方案。")

    def _selection(self) -> List[Individual]:
        parents = []
        valid_population = [p for p in self.population if p.fitness != float('inf')]
        source_population = valid_population if valid_population else self.population
        
        if not source_population: return []

        selectable_population = list(source_population)
        num_parents_to_select = min(N_PARENTS_FOR_RECOMBINATION, len(selectable_population))

        for _ in range(num_parents_to_select):
            current_tournament_size = min(TOURNAMENT_SIZE, len(selectable_population))
            if current_tournament_size == 0: break
            tournament_contenders = random.sample(selectable_population, current_tournament_size)
            winner = min(tournament_contenders, key=lambda ind: ind.fitness)
            parents.append(winner)
            selectable_population.remove(winner)
        return parents

    def _recombine_and_refine(self, parents: List[Individual]) -> Individual:
        all_customers_list = list(range(1, N))
        system_prompt = f"""
        你将扮演两个角色来优化VRPTW方案。
        总客户列表: {all_customers_list}

        1.  批判家 (Critic):
            - 分析父代方案的成本和结构。指出问题，例如：哪些路线时间惩罚高？客户分组是否合理？能否通过重组减少车辆数或总距离？

        2.  作者 (Author):
            - 基于批判分析，创造一个全新的、必须有效的解决方案。
            - 思考与自检步骤 (强制执行):
                1. 设计新分组: 提出全新的客户分组，确保每组总需求不超过{VEHICLE_CAPACITY}。
                2. 优化顺序: 思考每个组内的访问顺序以减少成本。
                3. 自检: 生成方案后，立即检查方案中的所有客户集合是否与【总客户列表】完全一致。如果不一致，必须重新设计。
                4. 输出: 只有通过自检的方案才能输出。

        输出格式 (严格遵守):
        {{
            "critic_analysis": "...",
            "author_new_plan": {{"routes": [...]}}
        }}
        """
        
        parent_info = ""
        for i, p in enumerate(parents):
            parent_info += f"父代方案 {i+1} (成本: {p.fitness:.2f}): {p.plan}\n"

        user_prompt = f"参考以下父代方案:\n---\n{parent_info}---\n现在，请开始你的工作。"
        
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response_str = self._prompt_llm(messages)
        
        try:
            data = json.loads(response_str)
            new_plan_str = json.dumps(data['author_new_plan'])
            print(f"  [批判家分析]: {data.get('critic_analysis', '无')}")
            print(f"  [作者新方案]: {new_plan_str}")
            return Individual(plan=new_plan_str)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"  [精炼失败]: {e}")
            return random.choice(parents) if parents else Individual('{"routes":[]}')

    def run(self):
        self.initialize_population()
        if not self.population or all(p.fitness == float('inf') for p in self.population):
            print("初始化失败，无法创建任何有效初始方案，程序终止。")
            return

        self.best_solution_so_far = min(self.population, key=lambda ind: ind.fitness)
        
        print(f"\n--- 初始最佳方案 ---\n{self.best_solution_so_far}\n" + "="*30)

        for gen in range(N_GENERATIONS):
            print(f"\n--- 第 {gen + 1}/{N_GENERATIONS} 代进化开始 ---")
            
            if not any(p.fitness != float('inf') for p in self.population):
                print("当前种群已无有效方案，进化终止。")
                break

            new_population = [self.best_solution_so_far] # 精英主义

            while len(new_population) < POPULATION_SIZE:
                parents = self._selection()
                if not parents: 
                    # 如果没有足够的多样性来选择父代, 则对最优解进行变异
                    print("\n种群多样性不足，对最优解进行变异...")
                    child = self.mutate_with_llm(self.best_solution_so_far)
                else:
                    child = self._recombine_and_refine(parents)
                
                child.fitness, child.feedback = self.evaluator.evaluate(child.plan)
                print(f"  评估新方案 -> 成本: {'%.2f' % child.fitness}, 反馈: {', '.join(child.feedback)}")
                new_population.append(child)

            self.population = new_population
            
            current_best_in_gen = min(self.population, key=lambda ind: ind.fitness)
            if current_best_in_gen.fitness < self.best_solution_so_far.fitness:
                self.best_solution_so_far = current_best_in_gen
            
            print(f"\n--- 第 {gen + 1} 代结束 ---")
            print(f"本代最佳: {current_best_in_gen}")
            print(f"全局最佳: {self.best_solution_so_far}")
            print("="*30)
            
        print("\n--- 进化结束 ---")
        print("最终找到的最优方案是:")
        print(self.best_solution_so_far)
        self.print_detailed_report()
    
    def mutate_with_llm(self, individual: Individual) -> Individual:
        """使用LLM对一个个体进行智能变异"""
        system_prompt = f"""
        你是一个VRPTW变异专家。你的任务是对一个现有方案进行小幅度的、智能的修改，以探索其邻近的解空间。
        核心规则: 1. 新方案必须保持有效（满足所有硬性约束）。 2. 变异应该是小范围的，例如交换一两个客户，或者将一个客户从一条路线移动到另一条路线。
        总客户列表: {list(range(1, N))}
        输出格式: 严格输出JSON: {{"mutated_plan": {{"routes": [...]}}}}
        """
        user_prompt = f"这是需要进行智能变异的方案 (成本: {individual.fitness:.2f}):\n{individual.plan}\n请生成一个经过小幅修改的新方案。"
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        
        response_str = self._prompt_llm(messages)
        try:
            data = json.loads(response_str)
            mutated_plan_str = json.dumps(data['mutated_plan'])
            print(f"  [智能变异新方案]: {mutated_plan_str}")
            return Individual(plan=mutated_plan_str)
        except (json.JSONDecodeError, KeyError, TypeError):
            print("  [智能变异失败]，返回原方案。")
            return individual


    def print_detailed_report(self):
        print("\n详细成本分析:")
        try:
            routes_data = json.loads(self.best_solution_so_far.plan)
            routes = routes_data['routes']
        except (json.JSONDecodeError, TypeError):
            print("最终方案格式错误，无法生成详细报告。")
            return

        total_transport = 0
        total_penalty = 0
        total_fixed = len(routes) * VEHICLE_FIXED_COST

        for i, route in enumerate(routes, 1):
            route_fixed = VEHICLE_FIXED_COST
            dist = sum(DISTANCE_MATRIX[route[j]][route[j + 1]] for j in range(len(route) - 1))
            transport_cost = dist * TRANSPORT_COST_PER_KM
            
            t = 0
            penalty_cost = 0
            for j in range(1, len(route)):
                travel_time = DISTANCE_MATRIX[route[j - 1]][route[j]] / AVG_SPEED
                t += travel_time
                if route[j] != 0:
                    penalty = self.evaluator._time_penalty(t, TIME_WINDOWS[route[j]])
                    penalty_cost += penalty

            route_cost = route_fixed + transport_cost + penalty_cost
            load = sum(TIME_WINDOWS[node][4] for node in route if node != 0)

            total_transport += transport_cost
            total_penalty += penalty_cost

            print(f"车辆{i}: 载重{load:5.2f}, 距离{dist:6.2f}km, "
                  f"运输成本{transport_cost:6.2f}元, 时间惩罚{penalty_cost:6.2f}元, "
                  f"总成本{route_cost:7.2f}元")
            print(f"      路径: {route}")

        print(f"\n汇总: 固定成本{total_fixed:6.2f}元, "
              f"运输成本{total_transport:6.2f}元, "
              f"时间惩罚{total_penalty:6.2f}元, "
              f"总计{self.best_solution_so_far.fitness:.2f}元")

def main():
    print("="*50)
    print("Mind Evolution for Customer VRPTW")
    print(f"模型: {MODEL_NAME}")
    print(f"配置: {N_GENERATIONS} 代, 每代 {POPULATION_SIZE} 个方案")
    print("="*50)

    evolution_engine = MindEvolution()
    evolution_engine.run()

if __name__ == "__main__":
    main()

