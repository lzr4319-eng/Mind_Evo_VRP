import re
import math
import itertools
from functools import lru_cache

class VrpSolverDP:
    def __init__(self, input_string):
        """
        解析输入字符串并初始化问题数据。
        """
        self.points = {}
        self.demands = {}
        self.capacity = 0
        self.depot_name = 'Depot'
        self.customer_names = []
        
        self._parse_input(input_string)
        self._calculate_distance_matrix()

    def _parse_input(self, input_string):
        """
        使用正则表达式解析输入，填充实例数据。
        """
        try:
            # 1. 解析仓库
            depot_match = re.search(r'Depot\((\d+),(\d+)\)', input_string)
            if not depot_match:
                raise ValueError("未找到或格式错误的 Depot 信息")
            self.points[self.depot_name] = (int(depot_match.group(1)), int(depot_match.group(2)))

            # 2. 解析客户
            cust_pattern = r'([A-Z])\((\d+),(\d+),\s*demand:(\d+)\)'
            customers_match = re.findall(cust_pattern, input_string)
            if not customers_match:
                raise ValueError("未找到或格式错误的 Customers 信息")
            
            for name, x, y, demand in customers_match:
                self.customer_names.append(name)
                self.points[name] = (int(x), int(y))
                self.demands[name] = int(demand)
            
            # 3. 解析容量
            cap_match = re.search(r'Capacity:(\d+)', input_string)
            if not cap_match:
                 raise ValueError("未找到或格式错误的 Capacity 信息")
            self.capacity = int(cap_match.group(1))

            # 客户名称排序，保证一致性
            self.customer_names.sort()

        except Exception as e:
            print(f"输入格式解析错误: {e}")
            raise

    def _calculate_distance_matrix(self):
        """
        计算所有点对之间的欧几里得距离。
        """
        self.locations = [self.depot_name] + self.customer_names
        self.dist_matrix = {}
        for p1_name in self.locations:
            self.dist_matrix[p1_name] = {}
            for p2_name in self.locations:
                p1_coords = self.points[p1_name]
                p2_coords = self.points[p2_name]
                dist = math.sqrt((p1_coords[0] - p2_coords[0])**2 + (p1_coords[1] - p2_coords[1])**2)
                self.dist_matrix[p1_name][p2_name] = dist
                
    def _solve_tsp_for_route(self, customer_subset):
        """
        使用动态规划(Held-Karp)为给定的客户子集找到从仓库出发的最优路径。
        返回 (最短距离, 路径列表)。
        """
        nodes = [self.depot_name] + customer_subset
        n = len(nodes)
        
        # Memoization cache
        memo = {}

        def tsp_dp(mask, last_node_idx):
            # 如果所有节点都已访问
            if mask == (1 << n) - 1:
                return self.dist_matrix[nodes[last_node_idx]][self.depot_name], [self.depot_name]

            state = (mask, last_node_idx)
            if state in memo:
                return memo[state]

            min_dist = float('inf')
            best_path = []

            for next_node_idx in range(1, n): # 不从仓库开始循环
                if not (mask & (1 << next_node_idx)): # 如果节点未访问
                    new_mask = mask | (1 << next_node_idx)
                    dist, path_segment = tsp_dp(new_mask, next_node_idx)
                    current_dist = self.dist_matrix[nodes[last_node_idx]][nodes[next_node_idx]] + dist
                    
                    if current_dist < min_dist:
                        min_dist = current_dist
                        best_path = [nodes[next_node_idx]] + path_segment
            
            memo[state] = (min_dist, best_path)
            return min_dist, best_path

        # 从仓库开始 (index 0, mask 1)
        dist, path = tsp_dp(1, 0)
        return dist, [self.depot_name] + path

    def solve(self):
        """
        主求解函数，执行两阶段动态规划。
        """
        print("第一阶段：生成所有可行的单车路线及其成本...")
        # 1. 生成所有满足容量约束的客户子集，并计算其最优TSP路径成本
        valid_routes = {} # key: frozenset of customers, value: (cost, path)
        num_customers = len(self.customer_names)
        
        for i in range(1, num_customers + 1):
            for subset in itertools.combinations(self.customer_names, i):
                demand = sum(self.demands[c] for c in subset)
                if demand <= self.capacity:
                    cost, path = self._solve_tsp_for_route(list(subset))
                    valid_routes[frozenset(subset)] = (cost, path)
        
        print(f"共生成 {len(valid_routes)} 条可行路线。\n")
        print("第二阶段：求解集合划分问题以找到最佳路线组合...")

        # 2. 使用DP解决集合划分问题
        # dp[mask] = (min_cost, plan)
        # mask是一个位掩码，表示已服务的客户集
        # plan是达到该状态所使用的路线列表
        
        # 将客户名称映射到位掩码索引
        customer_map = {name: i for i, name in enumerate(self.customer_names)}
        
        dp = {0: (0, [])}
        
        for mask in range(1, 1 << num_customers):
            dp[mask] = (float('inf'), None)
            for route_customers_fs, (route_cost, route_path) in valid_routes.items():
                
                # 创建此路线的位掩码
                route_mask = 0
                for customer in route_customers_fs:
                    route_mask |= (1 << customer_map[customer])
                
                # 如果此路线是当前状态mask的子集
                if (mask & route_mask) == route_mask:
                    prev_mask = mask ^ route_mask
                    if prev_mask in dp and dp[prev_mask][0] != float('inf'):
                        new_cost = dp[prev_mask][0] + route_cost
                        if new_cost < dp[mask][0]:
                            new_plan = dp[prev_mask][1] + [(route_cost, route_path, route_customers_fs)]
                            dp[mask] = (new_cost, new_plan)
                            
        # 从最终状态回溯结果
        final_mask = (1 << num_customers) - 1
        if final_mask not in dp or dp[final_mask][0] == float('inf'):
            print("未能找到解决方案。")
            return

        total_distance, final_plan = dp[final_mask]
        
        print("\n=============== 最佳VRP方案 ===============\n")
        print(f"总路径长度: {total_distance:.2f}")
        print(f"所需车辆数: {len(final_plan)}")
        
        for i, (route_dist, route_path, route_customers) in enumerate(final_plan):
            route_demand = sum(self.demands[c] for c in route_customers)
            print(f"\n--- 车辆 {i+1} ---")
            print(f"  路径: {' -> '.join(route_path)}")
            print(f"  路径长度: {route_dist:.2f}")
            print(f"  服务客户: {', '.join(sorted(list(route_customers)))}")
            print(f"  本车负载: {route_demand} (容量: {self.capacity})")
        print("\n===========================================")


# --- 使用示例 ---
# 严格按照您提供的格式
input_data = "Depot(50,50) Customers:A(5,80, demand:22), B(90,92, demand:18), C(20,15, demand:25), D(75,5, demand:16), E(40,95, demand:28), F(95,55, demand:19), G(15,50, demand:21) Capacity:55"

print("正在处理输入数据...")
print(f"输入: {input_data}\n")

try:
    solver = VrpSolverDP(input_data)
    solver.solve()
except ValueError as e:
    print(f"程序终止，原因: {e}")
    
# 🧩 第一步：理解她的“沉默”其实不是拒绝

# 沉默型的人吵架时不是“不在乎”，而是大脑真的被情绪“卡住”了，暂时无法理性表达。
# 他们往往需要时间和空间来恢复自控力，否则说出来的话只会更糟。
# 如果你理解这一点，就不会把“她不说话”理解成“不理我”“冷暴力”。

# 🕰 第二步：建立一个“冷静—沟通”机制

# 建议你们在不吵架的时候先商量好：

# 当矛盾出现时，可以先暂时冷静 30 分钟 / 1 小时 / 一晚上，
# 但要有一个明确约定：冷静后一定会回来沟通。

# 举个例子，你可以对她说：

# “我知道你需要时间冷静，我也尊重你。但我希望我们能约定，比如冷静一小时后我们能再聊一聊，这样我心里也不会悬着。”

# 这能让你有安全感，也让她有空间。