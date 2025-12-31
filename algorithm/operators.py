# -*- coding: utf-8 -*-
"""
ALNS破坏与修复算子 (Destroy & Repair Operators)
这是ALNS算法的核心组件
"""

from typing import List, Tuple, Callable, Optional, Dict
import random
import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.solution import Solution
from models.vehicle import Vehicle
from models.node import Node, Order
from algorithm.objective import ObjectiveFunction
import config


class DestroyOperators:
    """
    破坏算子集合
    
    破坏算子负责从当前解中移除一部分订单
    """
    
    def __init__(
        self, 
        destroy_rate_min: float = None,
        destroy_rate_max: float = None,
        random_seed: int = None
    ):
        self.destroy_rate_min = destroy_rate_min or config.DESTROY_RATE_MIN
        self.destroy_rate_max = destroy_rate_max or config.DESTROY_RATE_MAX
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # 注册所有破坏算子
        self.operators: List[Tuple[str, Callable]] = [
            ("random_removal", self.random_removal),
            ("worst_removal", self.worst_removal),
            ("shaw_removal", self.shaw_removal),
            ("route_removal", self.route_removal),
            ("spatial_proximity_removal", self.spatial_proximity_removal),  # h2: 空间邻近移除
            ("deadline_based_removal", self.deadline_based_removal),  # h7: 截止时间移除
        ]
        
        # 算子权重 (用于自适应选择)
        self.weights = {name: 1.0 for name, _ in self.operators}
        self.scores = {name: 0.0 for name, _ in self.operators}
        self.usage_counts = {name: 0 for name, _ in self.operators}
        
        # UCB参数
        self.use_ucb = True  # 是否使用UCB算法
        self.ucb_c = 2.0  # UCB探索系数
        self.total_iterations = 0  # 总迭代次数
        self.avg_rewards = {name: 0.0 for name, _ in self.operators}  # 平均奖励
    
    def get_destroy_count(self, solution: Solution) -> int:
        """计算要移除的订单数量"""
        assigned_count = len(solution.orders) - solution.num_unassigned
        if assigned_count == 0:
            return 0
        
        rate = random.uniform(self.destroy_rate_min, self.destroy_rate_max)
        count = max(1, int(assigned_count * rate))
        return min(count, assigned_count)
    
    def select_operator(self) -> Tuple[str, Callable]:
        """
        选择破坏算子
        使用UCB (Upper Confidence Bound) 算法或轮盘赌选择
        
        UCB公式: Score_i = X̄_i + C * sqrt(2 * ln(N) / n_i)
        - X̄_i: 算子i的平均奖励
        - N: 总迭代次数
        - n_i: 算子i被使用的次数
        - C: 探索系数
        """
        self.total_iterations += 1
        
        if self.use_ucb:
            # 使用UCB算法
            ucb_scores = {}
            for name, _ in self.operators:
                if self.usage_counts[name] == 0:
                    # 未使用过的算子给予最高优先级
                    ucb_scores[name] = float('inf')
                else:
                    # UCB Score = 平均奖励 + 探索项
                    exploitation = self.avg_rewards[name]
                    exploration = self.ucb_c * math.sqrt(
                        2 * math.log(self.total_iterations) / self.usage_counts[name]
                    )
                    ucb_scores[name] = exploitation + exploration
            
            # 选择UCB分数最高的算子
            best_name = max(ucb_scores, key=ucb_scores.get)
            idx = [name for name, _ in self.operators].index(best_name)
            return self.operators[idx]
        else:
            # 使用传统轮盘赌选择
            total_weight = sum(self.weights.values())
            probabilities = [self.weights[name] / total_weight for name, _ in self.operators]
            
            idx = np.random.choice(len(self.operators), p=probabilities)
            return self.operators[idx]
    
    def random_removal(self, solution: Solution) -> List[Order]:
        """
        随机移除算子
        
        随机选择N个订单移除
        """
        n = self.get_destroy_count(solution)
        if n == 0:
            return []
        
        # 获取所有已分配订单
        assigned_order_ids = solution.get_assigned_orders()
        if len(assigned_order_ids) == 0:
            return []
        
        # 随机选择要移除的订单
        remove_count = min(n, len(assigned_order_ids))
        selected_ids = random.sample(list(assigned_order_ids), remove_count)
        
        removed_orders = []
        for order_id in selected_ids:
            order = solution.get_order_by_id(order_id)
            if order and solution.unassign_order(order):
                removed_orders.append(order)
        
        return removed_orders
    
    def worst_removal(self, solution: Solution) -> List[Order]:
        """
        最差移除算子
        
        移除那些导致成本增加最多的订单 (贡献最差的订单)
        """
        n = self.get_destroy_count(solution)
        if n == 0:
            return []
        
        objective = ObjectiveFunction()
        order_costs = []
        
        # 计算每个订单的"移除后成本减少量"
        for order_id in solution.get_assigned_orders():
            order = solution.get_order_by_id(order_id)
            if order is None:
                continue
            
            vehicle = solution.find_order_vehicle(order)
            if vehicle is None:
                continue
            
            # 计算移除前后的成本差
            original_distance = vehicle.calculate_distance()
            original_violation = vehicle.calculate_time_violation()
            
            # 临时移除
            temp_route = vehicle.route.copy()
            try:
                vehicle.route.remove(order.pickup_node)
                vehicle.route.remove(order.delivery_node)
            except ValueError:
                vehicle.route = temp_route
                continue
            
            vehicle.invalidate_cache()
            new_distance = vehicle.calculate_distance()
            new_violation = vehicle.calculate_time_violation()
            
            # 恢复
            vehicle.route = temp_route
            vehicle.invalidate_cache()
            
            # 成本减少量 (正值意味着移除后成本降低)
            cost_reduction = (
                (original_distance - new_distance) * config.WEIGHT_DISTANCE +
                (original_violation - new_violation) * config.WEIGHT_TIME_PENALTY
            )
            
            order_costs.append((order, cost_reduction))
        
        # 按成本减少量排序 (移除后成本减少最多的排在前面)
        order_costs.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前N个移除 (带随机性)
        removed_orders = []
        remaining = list(order_costs)
        
        while len(removed_orders) < n and remaining:
            # 使用随机化的worst removal (防止陷入局部最优)
            p = random.random()
            idx = int((p ** 2) * len(remaining))  # 二次分布，偏向选择差的
            idx = min(idx, len(remaining) - 1)
            
            order, _ = remaining.pop(idx)
            if solution.unassign_order(order):
                removed_orders.append(order)
        
        return removed_orders
    
    def shaw_removal(self, solution: Solution) -> List[Order]:
        """
        Shaw移除算子 (相关性移除)
        
        移除相似度高的一组订单 (距离近、时间近)
        """
        n = self.get_destroy_count(solution)
        if n == 0:
            return []
        
        assigned_ids = list(solution.get_assigned_orders())
        if len(assigned_ids) == 0:
            return []
        
        # 随机选择一个种子订单
        seed_id = random.choice(assigned_ids)
        seed_order = solution.get_order_by_id(seed_id)
        
        # 计算所有订单与种子订单的相似度
        similarities = []
        for order_id in assigned_ids:
            if order_id == seed_id:
                continue
            order = solution.get_order_by_id(order_id)
            if order is None:
                continue
            
            similarity = self._calculate_relatedness(seed_order, order)
            similarities.append((order, similarity))
        
        # 按相似度排序 (相似度越高越靠前)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 移除种子订单和相似的订单
        removed_orders = []
        if solution.unassign_order(seed_order):
            removed_orders.append(seed_order)
        
        for order, _ in similarities:
            if len(removed_orders) >= n:
                break
            if solution.unassign_order(order):
                removed_orders.append(order)
        
        return removed_orders
    
    def _calculate_relatedness(self, order1: Order, order2: Order) -> float:
        """
        计算两个订单的相关性 (Shaw Removal)
        
        相关性 = 1 / (归一化距离 + 归一化时间差)
        
        对距离和时间进行归一化处理，避免某一因素主导相似性计算：
        - 距离归一化：除以 GRID_SIZE (最大可能距离约为 2*GRID_SIZE)
        - 时间归一化：除以 TIME_HORIZON (最大可能时间差)
        """
        # 取货点距离
        pickup_dist = order1.pickup_node.distance_to(order2.pickup_node)
        # 送货点距离
        delivery_dist = order1.delivery_node.distance_to(order2.delivery_node)
        # 时间窗差异 (取货点时间)
        pickup_time_diff = abs(order1.pickup_node.ready_time - order2.pickup_node.ready_time)
        # 时间窗差异 (送货点时间)
        delivery_time_diff = abs(order1.delivery_node.due_time - order2.delivery_node.due_time)
        
        # 归一化处理
        # 曼哈顿距离最大值约为 2 * GRID_SIZE
        max_distance = 2 * config.GRID_SIZE
        # 时间差最大值为 TIME_HORIZON
        max_time_diff = config.TIME_HORIZON
        
        # 归一化后的距离因子 (0-1 范围)
        normalized_dist = (pickup_dist + delivery_dist) / (2 * max_distance)
        # 归一化后的时间因子 (0-1 范围)
        normalized_time = (pickup_time_diff + delivery_time_diff) / (2 * max_time_diff)
        
        # 权重系数：距离和时间同等重要
        distance_weight = 0.5
        time_weight = 0.5
        
        # 相关性 = 1 / (加权归一化距离 + epsilon)
        # epsilon 防止除零
        relatedness = 1.0 / (
            distance_weight * normalized_dist + 
            time_weight * normalized_time + 
            0.001
        )
        
        return relatedness
    
    def route_removal(self, solution: Solution) -> List[Order]:
        """
        路径移除算子
        
        随机选择一条路径，移除其中的部分或全部订单
        """
        # 找到非空路径
        non_empty_vehicles = [v for v in solution.vehicles if len(v.route) > 0]
        if len(non_empty_vehicles) == 0:
            return []
        
        # 随机选择一条路径
        vehicle = random.choice(non_empty_vehicles)
        order_ids = list(vehicle.get_order_ids())
        
        if len(order_ids) == 0:
            return []
        
        # 决定移除数量 (50%-100%的订单)
        remove_ratio = random.uniform(0.5, 1.0)
        remove_count = max(1, int(len(order_ids) * remove_ratio))
        
        selected_ids = random.sample(order_ids, remove_count)
        
        removed_orders = []
        for order_id in selected_ids:
            order = solution.get_order_by_id(order_id)
            if order and solution.unassign_order(order):
                removed_orders.append(order)
        
        return removed_orders
    
    def spatial_proximity_removal(self, solution: Solution) -> List[Order]:
        """
        空间邻近移除算子 (h2: Spatial Proximity Removal)
        
        移除地理位置非常接近的一组已分配订单
        这有助于跳出局部最优，重新优化某个区域的订单分配
        """
        n = self.get_destroy_count(solution)
        if n == 0:
            return []
        
        assigned_ids = list(solution.get_assigned_orders())
        if len(assigned_ids) == 0:
            return []
        
        # 随机选择一个种子订单
        seed_id = random.choice(assigned_ids)
        seed_order = solution.get_order_by_id(seed_id)
        
        # 计算移除半径（基于网格大小）
        radius = config.GRID_SIZE * random.uniform(0.15, 0.35)
        
        # 找到半径内的所有订单
        nearby_orders = []
        for order_id in assigned_ids:
            if order_id == seed_id:
                continue
            order = solution.get_order_by_id(order_id)
            if order is None:
                continue
            
            # 计算订单与种子订单的平均距离
            # 考虑取货点和送货点的距离
            pickup_dist = seed_order.pickup_node.distance_to(order.pickup_node)
            delivery_dist = seed_order.delivery_node.distance_to(order.delivery_node)
            avg_dist = (pickup_dist + delivery_dist) / 2
            
            if avg_dist <= radius:
                nearby_orders.append(order)
        
        # 移除种子订单和附近订单
        removed_orders = []
        if solution.unassign_order(seed_order):
            removed_orders.append(seed_order)
        
        # 按距离排序，优先移除距离更近的
        nearby_orders.sort(
            key=lambda o: (
                seed_order.pickup_node.distance_to(o.pickup_node) +
                seed_order.delivery_node.distance_to(o.delivery_node)
            ) / 2
        )
        
        for order in nearby_orders:
            if len(removed_orders) >= n:
                break
            if solution.unassign_order(order):
                removed_orders.append(order)
        
        return removed_orders
    
    def deadline_based_removal(self, solution: Solution) -> List[Order]:
        """
        截止时间移除算子 (h7: Deadline-based Removal)
        
        移除截止时间最紧迫或最晚的订单
        这些订单往往是造成整个解不可行或成本过高的"钉子户"
        """
        n = self.get_destroy_count(solution)
        if n == 0:
            return []
        
        assigned_ids = list(solution.get_assigned_orders())
        if len(assigned_ids) == 0:
            return []
        
        # 收集所有已分配订单及其deadline信息
        order_deadlines = []
        for order_id in assigned_ids:
            order = solution.get_order_by_id(order_id)
            if order is None:
                continue
            
            # 使用送货点的截止时间作为关键指标
            deadline = order.delivery_node.due_time
            # 计算时间窗宽度（越窄越紧迫）
            time_window = order.delivery_node.due_time - order.delivery_node.ready_time
            
            order_deadlines.append((order, deadline, time_window))
        
        if len(order_deadlines) == 0:
            return []
        
        # 随机选择策略：移除最紧迫的或最晚的
        strategy = random.choice(['earliest', 'latest', 'tightest'])
        
        if strategy == 'earliest':
            # 移除截止时间最早的订单（最紧迫）
            order_deadlines.sort(key=lambda x: x[1])
        elif strategy == 'latest':
            # 移除截止时间最晚的订单
            order_deadlines.sort(key=lambda x: x[1], reverse=True)
        else:  # tightest
            # 移除时间窗最窄的订单（最难安排）
            order_deadlines.sort(key=lambda x: x[2])
        
        # 移除前N个订单（带随机性）
        removed_orders = []
        for i in range(min(n, len(order_deadlines))):
            # 使用随机化选择，偏向前面的订单
            p = random.random()
            idx = int((p ** 2) * min(len(order_deadlines) - i, 5))
            idx = min(idx, len(order_deadlines) - i - 1)
            
            order, _, _ = order_deadlines.pop(idx)
            if solution.unassign_order(order):
                removed_orders.append(order)
        
        return removed_orders
    
    def update_weights(self, operator_name: str, score: float, decay: float = None):
        """
        更新算子权重和统计信息
        
        参考ALNS标准方法：
        w_i = w_i * (1 - r) + r * (score_i / count_i)
        
        Args:
            operator_name: 算子名称
            score: 本次得分（sigma_1, sigma_2, 或 sigma_3）
            decay: 权重衰减系数（None时使用config.DECAY_RATE）
        """
        if decay is None:
            decay = config.DECAY_RATE
        
        self.scores[operator_name] += score
        self.usage_counts[operator_name] += 1
        
        # 更新平均奖励（用于UCB）
        self.avg_rewards[operator_name] = self.scores[operator_name] / self.usage_counts[operator_name]
        
        # 分段更新权重（用于轮盘赌）
        if sum(self.usage_counts.values()) % config.SEGMENT_SIZE == 0:
            for name in self.weights:
                if self.usage_counts[name] > 0:
                    avg_score = self.scores[name] / self.usage_counts[name]
                    # ALNS标准公式：w_new = w_old * (1-decay) + decay * avg_score
                    self.weights[name] = self.weights[name] * (1 - decay) + decay * avg_score
                    # 防止权重过低
                    self.weights[name] = max(0.1, self.weights[name])
            
            # 重置分数和计数
            self.scores = {name: 0.0 for name in self.scores}
            self.usage_counts = {name: 0 for name in self.usage_counts}


class RepairOperators:
    """
    修复算子集合
    
    修复算子负责将移除的订单重新插入解中
    """
    
    def __init__(self, random_seed: int = None):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.objective = ObjectiveFunction()
        
        # 注册所有修复算子
        self.operators: List[Tuple[str, Callable]] = [
            ("greedy_insertion", self.greedy_insertion),
            ("regret_2_insertion", self.regret_2_insertion),
            ("regret_3_insertion", self.regret_3_insertion),
            ("random_insertion", self.random_insertion),
        ]
        
        # 算子权重
        self.weights = {name: 1.0 for name, _ in self.operators}
        self.scores = {name: 0.0 for name, _ in self.operators}
        self.usage_counts = {name: 0 for name, _ in self.operators}
        
        # UCB参数
        self.use_ucb = True  # 是否使用UCB算法
        self.ucb_c = 2.0  # UCB探索系数
        self.total_iterations = 0  # 总迭代次数
        self.avg_rewards = {name: 0.0 for name, _ in self.operators}  # 平均奖励
    
    def select_operator(self) -> Tuple[str, Callable]:
        """
        选择修复算子
        使用UCB (Upper Confidence Bound) 算法或轮盘赌选择
        """
        self.total_iterations += 1
        
        if self.use_ucb:
            # 使用UCB算法
            ucb_scores = {}
            for name, _ in self.operators:
                if self.usage_counts[name] == 0:
                    ucb_scores[name] = float('inf')
                else:
                    exploitation = self.avg_rewards[name]
                    exploration = self.ucb_c * math.sqrt(
                        2 * math.log(self.total_iterations) / self.usage_counts[name]
                    )
                    ucb_scores[name] = exploitation + exploration
            
            best_name = max(ucb_scores, key=ucb_scores.get)
            idx = [name for name, _ in self.operators].index(best_name)
            return self.operators[idx]
        else:
            # 使用传统轮盘赌选择
            total_weight = sum(self.weights.values())
            probabilities = [self.weights[name] / total_weight for name, _ in self.operators]
            
            idx = np.random.choice(len(self.operators), p=probabilities)
            return self.operators[idx]
    
    def greedy_insertion(self, solution: Solution, orders: List[Order]) -> int:
        """
        贪婪插入算子
        
        对每个订单，插入到成本增加最小的位置
        """
        inserted_count = 0
        
        # 按时间窗紧迫程度排序
        sorted_orders = sorted(orders, key=lambda o: o.pickup_node.due_time)
        
        for order in sorted_orders:
            best_insertion = self._find_best_insertion(solution, order)
            
            if best_insertion is not None:
                vehicle_id, p_pos, d_pos = best_insertion
                vehicle = solution.vehicles[vehicle_id]
                if solution.assign_order(order, vehicle, p_pos, d_pos):
                    inserted_count += 1
        
        return inserted_count
    
    def _find_best_insertion(
        self, 
        solution: Solution, 
        order: Order
    ) -> Optional[Tuple[int, int, int]]:
        """找到最佳插入位置"""
        best_cost = float('inf')
        best_insertion = None
        
        for v_idx, vehicle in enumerate(solution.vehicles):
            route_len = len(vehicle.route)
            
            for p_pos in range(route_len + 1):
                for d_pos in range(p_pos, route_len + 1):
                    cost, feasible = self.objective.calculate_insertion_cost(
                        vehicle,
                        order.pickup_node,
                        order.delivery_node,
                        p_pos,
                        d_pos
                    )
                    
                    if feasible and cost < best_cost:
                        best_cost = cost
                        best_insertion = (v_idx, p_pos, d_pos)
        
        return best_insertion
    
    def regret_2_insertion(self, solution: Solution, orders: List[Order]) -> int:
        """Regret-2 插入算子"""
        return self._regret_insertion(solution, orders, k=2)
    
    def regret_3_insertion(self, solution: Solution, orders: List[Order]) -> int:
        """Regret-3 插入算子"""
        return self._regret_insertion(solution, orders, k=3)
    
    def _regret_insertion(
        self, 
        solution: Solution, 
        orders: List[Order], 
        k: int
    ) -> int:
        """
        Regret-k 插入算子
        
        优先插入后悔值大的订单
        """
        inserted_count = 0
        remaining_orders = list(orders)
        
        while remaining_orders:
            # 计算每个订单的regret值
            regrets = []
            for order in remaining_orders:
                regret, best_insertion = self._calculate_regret(solution, order, k)
                regrets.append((regret, order, best_insertion))
            
            # 按regret值降序排序
            regrets.sort(key=lambda x: x[0], reverse=True)
            
            best_regret, best_order, best_insertion = regrets[0]
            
            if best_insertion is not None:
                vehicle_id, p_pos, d_pos = best_insertion
                vehicle = solution.vehicles[vehicle_id]
                
                if solution.assign_order(best_order, vehicle, p_pos, d_pos):
                    inserted_count += 1
            
            remaining_orders.remove(best_order)
        
        return inserted_count
    
    def _calculate_regret(
        self, 
        solution: Solution, 
        order: Order, 
        k: int
    ) -> Tuple[float, Optional[Tuple[int, int, int]]]:
        """计算订单的regret-k值"""
        insertion_costs = []
        
        for v_idx, vehicle in enumerate(solution.vehicles):
            route_len = len(vehicle.route)
            
            for p_pos in range(route_len + 1):
                for d_pos in range(p_pos, route_len + 1):
                    cost, feasible = self.objective.calculate_insertion_cost(
                        vehicle,
                        order.pickup_node,
                        order.delivery_node,
                        p_pos,
                        d_pos
                    )
                    
                    if feasible:
                        insertion_costs.append((cost, v_idx, p_pos, d_pos))
        
        if len(insertion_costs) == 0:
            return float('inf'), None
        
        insertion_costs.sort(key=lambda x: x[0])
        
        best_cost = insertion_costs[0][0]
        best_insertion = (insertion_costs[0][1], insertion_costs[0][2], insertion_costs[0][3])
        
        if len(insertion_costs) >= k:
            kth_cost = insertion_costs[k - 1][0]
        else:
            kth_cost = insertion_costs[-1][0]
        
        regret = kth_cost - best_cost
        
        return regret, best_insertion
    
    def random_insertion(self, solution: Solution, orders: List[Order]) -> int:
        """
        随机插入算子
        
        随机选择可行的插入位置
        """
        inserted_count = 0
        random.shuffle(orders)
        
        for order in orders:
            feasible_insertions = []
            
            for v_idx, vehicle in enumerate(solution.vehicles):
                route_len = len(vehicle.route)
                
                for p_pos in range(route_len + 1):
                    for d_pos in range(p_pos, route_len + 1):
                        _, feasible = self.objective.calculate_insertion_cost(
                            vehicle,
                            order.pickup_node,
                            order.delivery_node,
                            p_pos,
                            d_pos
                        )
                        
                        if feasible:
                            feasible_insertions.append((v_idx, p_pos, d_pos))
            
            if feasible_insertions:
                vehicle_id, p_pos, d_pos = random.choice(feasible_insertions)
                vehicle = solution.vehicles[vehicle_id]
                
                if solution.assign_order(order, vehicle, p_pos, d_pos):
                    inserted_count += 1
        
        return inserted_count
    
    def update_weights(self, operator_name: str, score: float, decay: float = None):
        """
        更新算子权重和统计信息
        
        参考ALNS标准方法：
        w_i = w_i * (1 - r) + r * (score_i / count_i)
        
        Args:
            operator_name: 算子名称
            score: 本次得分
            decay: 权重衰减系数（None时使用config.DECAY_RATE）
        """
        if decay is None:
            decay = config.DECAY_RATE
        
        self.scores[operator_name] += score
        self.usage_counts[operator_name] += 1
        
        # 更新平均奖励（用于UCB）
        self.avg_rewards[operator_name] = self.scores[operator_name] / self.usage_counts[operator_name]
        
        # 分段更新权重（用于轮盘赌）
        if sum(self.usage_counts.values()) % config.SEGMENT_SIZE == 0:
            for name in self.weights:
                if self.usage_counts[name] > 0:
                    avg_score = self.scores[name] / self.usage_counts[name]
                    # ALNS标准公式：w_new = w_old * (1-decay) + decay * avg_score
                    self.weights[name] = self.weights[name] * (1 - decay) + decay * avg_score
                    # 防止权重过低
                    self.weights[name] = max(0.1, self.weights[name])
            
            # 重置分数和计数
            self.scores = {name: 0.0 for name in self.scores}
            self.usage_counts = {name: 0 for name in self.usage_counts}
