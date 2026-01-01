# -*- coding: utf-8 -*-
"""
目标函数与约束检查 (Objective Function & Validity Check)
这是PDPTW的"裁判"，决定解的好坏和合法性
"""

from typing import List, Dict, Tuple, Optional
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.solution import Solution
from models.vehicle import Vehicle
from models.node import Node, NodeType
import config


class ObjectiveFunction:
    """
    目标函数类
    
    目标函数公式:
    Cost = w1 * Distance + w2 * TimePenalty + w3 * UnassignedPenalty + w4 * VehicleUsage
    """
    
    def __init__(
        self,
        w_distance: float = None,
        w_time_penalty: float = None,
        w_unassigned: float = None,
        w_vehicle: float = None
    ):
        self.w_distance = w_distance or config.WEIGHT_DISTANCE
        self.w_time_penalty = w_time_penalty or config.WEIGHT_TIME_PENALTY
        self.w_unassigned = w_unassigned or config.WEIGHT_UNASSIGNED
        self.w_vehicle = w_vehicle or config.WEIGHT_VEHICLE_USAGE
    
    def calculate(self, solution: Solution) -> float:
        """
        计算解的总成本
        """
        return solution.calculate_cost(
            w_distance=self.w_distance,
            w_time_penalty=self.w_time_penalty,
            w_unassigned=self.w_unassigned,
            w_vehicle=self.w_vehicle
        )
    
    def calculate_detailed(self, solution: Solution) -> Dict[str, float]:
        """
        计算解的详细成本分解
        """
        distance_cost = self.w_distance * solution.total_distance
        time_penalty = self.w_time_penalty * solution.total_time_violation
        unassigned_penalty = self.w_unassigned * solution.num_unassigned
        vehicle_cost = self.w_vehicle * solution.num_used_vehicles
        
        return {
            'total_cost': distance_cost + time_penalty + unassigned_penalty + vehicle_cost,
            'distance_cost': distance_cost,
            'time_penalty': time_penalty,
            'unassigned_penalty': unassigned_penalty,
            'vehicle_cost': vehicle_cost,
            'raw_distance': solution.total_distance,
            'raw_time_violation': solution.total_time_violation,
            'raw_unassigned': solution.num_unassigned,
            'raw_vehicles': solution.num_used_vehicles
        }
    
    def calculate_insertion_cost(
        self, 
        vehicle: Vehicle,
        pickup_node: Node,
        delivery_node: Node,
        pickup_pos: int,
        delivery_pos: int,
        use_matching_score: bool = True,
        alpha: float = 0.7,
        beta: float = 0.3
    ) -> Tuple[float, bool]:
        """
        计算在指定位置插入订单后的成本增加量
        
        引入Matching Degree Score机制：
        Score = α * Cost + β * Risk
        - Cost: 传统的成本增加量
        - Risk: 风险评分（基于时间缓冲）
        
        Args:
            vehicle: 目标骑手
            pickup_node: 取货点
            delivery_node: 送货点
            pickup_pos: 取货点插入位置
            delivery_pos: 送货点插入位置
            use_matching_score: 是否使用风险决策评分（默认True）
            alpha: 成本权重（默认0.7）
            beta: 风险权重（默认0.3）
        
        Returns:
            (score, is_feasible): 匹配分数和是否可行
        """
        if pickup_pos > delivery_pos:
            return float('inf'), False
        
        # 计算原始成本
        original_distance = vehicle.calculate_distance()
        original_violation = vehicle.calculate_time_violation()
        
        # 临时插入
        old_route = vehicle.route.copy()
        vehicle.route.insert(pickup_pos, pickup_node)
        vehicle.route.insert(delivery_pos + 1, delivery_node)
        vehicle.invalidate_cache()
        
        # 检查可行性
        is_feasible = vehicle.is_feasible()
        
        # 计算新成本
        new_distance = vehicle.calculate_distance()
        new_violation = vehicle.calculate_time_violation()
        
        # 计算成本增加
        cost_increase = (
            self.w_distance * (new_distance - original_distance) +
            self.w_time_penalty * (new_violation - original_violation)
        )
        
        # 计算风险（如果启用Matching Score）
        risk = 0.0
        if use_matching_score and is_feasible:
            risk = self._calculate_insertion_risk(vehicle, pickup_pos, delivery_pos + 1)
        
        # 恢复原始路径
        vehicle.route = old_route
        vehicle.invalidate_cache()
        
        if not is_feasible:
            return float('inf'), False
        
        # 返回Matching Score或传统Cost
        if use_matching_score:
            score = alpha * cost_increase + beta * risk
        else:
            score = cost_increase
        
        return score, True
    
    def _calculate_insertion_risk(self, vehicle: Vehicle, pickup_idx: int, delivery_idx: int) -> float:
        """
        计算插入风险（基于时间缓冲）
        
        Risk = 时间紧张度的加权和
        - 如果一个节点插入后，骑手剩余缓冲时间很少，风险就高
        - 缓冲时间 = due_time - arrival_time
        
        Args:
            vehicle: 车辆（已经插入了新订单）
            pickup_idx: 取货点在route中的索引
            delivery_idx: 送货点在route中的索引
        
        Returns:
            风险值（越大表示风险越高）
        """
        arrival_times = vehicle.get_arrival_times()
        full_route = vehicle.full_route[1:]  # 排除起始depot
        
        risk = 0.0
        max_risk_per_node = 1000.0  # 最大风险值
        
        # 只考虑受影响的节点（插入位置及之后的节点）
        affected_start = min(pickup_idx, delivery_idx)
        
        for i in range(affected_start, len(full_route)):
            node = full_route[i]
            arrival = arrival_times[i]
            
            # 计算时间缓冲（slack time）
            slack = node.due_time - arrival
            
            if slack < 0:
                # 违反时间窗，高风险
                risk += max_risk_per_node
            elif slack < 10:
                # 缓冲时间很少（<10分钟），高风险
                risk += max_risk_per_node * (1 - slack / 10)
            elif slack < 30:
                # 缓冲时间较少（10-30分钟），中等风险
                risk += max_risk_per_node * 0.3 * (1 - (slack - 10) / 20)
            # else: 缓冲时间充足，风险低
        
        # 归一化风险（除以受影响节点数）
        if len(full_route) > affected_start:
            risk /= (len(full_route) - affected_start)
        
        return risk
    
    def calculate_insertion_cost_fast(
        self,
        vehicle: Vehicle,
        pickup_node: Node,
        delivery_node: Node,
        pickup_pos: int,
        delivery_pos: int
    ) -> Tuple[float, bool]:
        """
        快速增量评估插入成本（无深拷贝优化）
        
        关键优化：
        1. 只计算距离增量（Delta Distance），不修改route
        2. 使用O(1)的纯数学计算
        3. 快速容量检查
        4. 两阶段评估：先快速筛选，再精确检查
        
        Args:
            vehicle: 目标骑手
            pickup_node: 取货点
            delivery_node: 送货点
            pickup_pos: 取货点插入位置
            delivery_pos: 送货点插入位置（基于原route）
        
        Returns:
            (cost_delta, is_feasible): 成本增量和是否可行
        """
        if pickup_pos > delivery_pos:
            return float('inf'), False
        
        route = vehicle.route
        route_len = len(route)
        
        # ========== 阶段1: 快速距离增量计算（O(1)） ==========
        delta_dist = 0.0
        
        # 1.1 计算取货点插入带来的距离变化
        if pickup_pos == 0:
            prev_p = vehicle.current_location if vehicle.current_location else vehicle.depot
        else:
            prev_p = route[pickup_pos - 1]
        
        if pickup_pos < route_len:
            next_p = route[pickup_pos]
            # 减去旧边 prev_p -> next_p
            delta_dist -= prev_p.distance_to(next_p)
            # 加上新边 prev_p -> pickup_node -> next_p
            delta_dist += prev_p.distance_to(pickup_node)
            delta_dist += pickup_node.distance_to(next_p)
        else:
            # 插入到末尾
            delta_dist += prev_p.distance_to(pickup_node)
        
        # 1.2 计算送货点插入带来的距离变化
        # 注意：delivery_pos是基于原route，需要考虑pickup已插入的偏移
        actual_d_pos = delivery_pos + 1  # pickup插入后，后续索引+1
        
        if actual_d_pos == 0:
            prev_d = vehicle.current_location if vehicle.current_location else vehicle.depot
        elif actual_d_pos - 1 == pickup_pos:
            # 送货点紧跟在取货点后面
            prev_d = pickup_node
        elif actual_d_pos - 1 < route_len:
            prev_d = route[actual_d_pos - 1]
        else:
            prev_d = route[-1] if route_len > 0 else (vehicle.current_location or vehicle.depot)
        
        if actual_d_pos <= route_len:
            next_d = route[actual_d_pos - 1] if actual_d_pos > 0 and actual_d_pos - 1 < route_len else None
            if next_d and next_d != pickup_node:
                # 减去旧边 prev_d -> next_d
                delta_dist -= prev_d.distance_to(next_d)
                # 加上新边 prev_d -> delivery_node -> next_d
                delta_dist += prev_d.distance_to(delivery_node)
                delta_dist += delivery_node.distance_to(next_d)
            else:
                delta_dist += prev_d.distance_to(delivery_node)
        else:
            # 插入到末尾
            delta_dist += prev_d.distance_to(delivery_node)
        
        # 距离成本增量
        cost_delta = self.w_distance * delta_dist
        
        # ========== 阶段2: 快速容量检查（O(1)） ==========
        # 检查最大载重是否超限
        max_load_in_route = 0
        current_load = 0
        for node in route:
            current_load += node.demand
            max_load_in_route = max(max_load_in_route, current_load)
        
        # 新订单增加的载重
        new_max_load = max_load_in_route + pickup_node.demand
        if new_max_load > vehicle.capacity:
            return float('inf'), False
        
        # ========== 阶段3: 快速时间窗检查 ==========
        # 估算时间窗可行性（不完整模拟，但足够快速筛选）
        # 计算取货点到达时间的下界
        if pickup_pos == 0:
            start_time = 0  # 骑手从depot出发
            start_node = vehicle.current_location if vehicle.current_location else vehicle.depot
        else:
            # 估算前一节点的完成时间（简化：使用ready_time作为下界）
            prev_node = route[pickup_pos - 1]
            start_time = prev_node.ready_time + prev_node.service_time
            start_node = prev_node
        
        # 到达取货点的时间
        travel_time_to_pickup = start_node.distance_to(pickup_node) / config.VEHICLE_SPEED
        arrival_pickup = start_time + travel_time_to_pickup
        
        # 快速检查取货点时间窗
        if arrival_pickup > pickup_node.due_time + 30:  # 允许30分钟软违反
            return float('inf'), False
        
        # 到达送货点的时间
        service_pickup = max(arrival_pickup, pickup_node.ready_time) + pickup_node.service_time
        travel_time_to_delivery = pickup_node.distance_to(delivery_node) / config.VEHICLE_SPEED
        arrival_delivery = service_pickup + travel_time_to_delivery
        
        # 快速检查送货点时间窗
        if arrival_delivery > delivery_node.due_time + 30:  # 允许30分钟软违反
            return float('inf'), False
        
        # ========== 返回快速评估结果 ==========
        # 如果距离增量很大，可行但成本高
        if cost_delta > 500.0:  # 阈值可调
            return cost_delta, True  # 仍然返回可行，让精确评估决定
        
        # 对于距离增量较小的情况，返回可行性
        return cost_delta, True


def check_validity(solution: Solution) -> Tuple[bool, List[str]]:
    """
    检查解的合法性 (硬约束)
    
    Returns:
        (is_valid, violations): 是否合法及违反的约束列表
    """
    violations = []
    
    # 1. 检查配对约束: 取送点必须在同一辆车
    for order_id, order in solution.orders.items():
        if order in solution.unassigned_orders:
            continue
        
        pickup_vehicle = None
        delivery_vehicle = None
        
        for vehicle in solution.vehicles:
            for node in vehicle.route:
                if node.id == order.pickup_node.id:
                    pickup_vehicle = vehicle.id
                if node.id == order.delivery_node.id:
                    delivery_vehicle = vehicle.id
        
        if pickup_vehicle is None and delivery_vehicle is None:
            if order not in solution.unassigned_orders:
                violations.append(f"Order {order_id}: 未分配但不在未分配列表中")
        elif pickup_vehicle != delivery_vehicle:
            violations.append(
                f"Order {order_id}: 配对约束违反 - 取货在Vehicle{pickup_vehicle}, 送货在Vehicle{delivery_vehicle}"
            )
    
    # 2. 检查每个骑手的路径
    for vehicle in solution.vehicles:
        route = vehicle.route
        if len(route) == 0:
            continue
        
        # 2.1 检查顺序约束: 取货必须在送货之前
        visited_pickups = set()
        for i, node in enumerate(route):
            if node.is_pickup():
                visited_pickups.add(node.id)
            elif node.is_delivery():
                if node.pair_id not in visited_pickups:
                    violations.append(
                        f"Vehicle {vehicle.id}: 顺序约束违反 - D{node.id}在P{node.pair_id}之前"
                    )
        
        # 2.2 检查容量约束
        current_load = 0
        for node in route:
            current_load += node.demand
            if current_load > vehicle.capacity:
                violations.append(
                    f"Vehicle {vehicle.id}: 容量约束违反 - 载重{current_load}超过容量{vehicle.capacity}"
                )
                break
            if current_load < 0:
                violations.append(
                    f"Vehicle {vehicle.id}: 容量约束违反 - 载重为负{current_load}"
                )
                break
    
    return len(violations) == 0, violations


def check_time_window_hard(solution: Solution) -> Tuple[bool, List[str]]:
    """
    检查硬时间窗约束 (如果启用)
    
    Returns:
        (is_valid, violations): 是否合法及违反的约束列表
    """
    violations = []
    
    for vehicle in solution.vehicles:
        if len(vehicle.route) == 0:
            continue
        
        arrival_times = vehicle.get_arrival_times()
        full_route = vehicle.full_route[1:]  # 排除起始depot
        
        for i, (node, arrival) in enumerate(zip(full_route, arrival_times)):
            if arrival > node.due_time:
                violations.append(
                    f"Vehicle {vehicle.id}: 时间窗违反 - 到达{node}时间{arrival:.2f}超过截止时间{node.due_time}"
                )
    
    return len(violations) == 0, violations


def evaluate_route(vehicle: Vehicle, objective: ObjectiveFunction = None) -> Dict:
    """
    评估单个骑手路径
    """
    if objective is None:
        objective = ObjectiveFunction()
    
    return {
        'vehicle_id': vehicle.id,
        'route_length': len(vehicle.route),
        'distance': vehicle.calculate_distance(),
        'time_violation': vehicle.calculate_time_violation(),
        'is_feasible': vehicle.is_feasible(),
        'capacity_ok': vehicle.check_capacity_feasibility(),
        'precedence_ok': vehicle.check_precedence_feasibility(),
        'order_ids': list(vehicle.get_order_ids())
    }


def calculate_regret(
    order,
    solution: Solution,
    objective: ObjectiveFunction,
    k: int = 2
) -> Tuple[float, Optional[Tuple]]:
    """
    计算订单的k-regret值
    
    Regret = (第k好的插入成本 - 最好的插入成本)
    
    用于regret insertion启发式
    
    Returns:
        (regret_value, best_insertion): regret值和最佳插入位置
    """
    insertion_costs = []
    
    for vehicle in solution.vehicles:
        route_len = len(vehicle.route)
        
        # 遍历所有可能的取货点位置
        for p_pos in range(route_len + 1):
            # 遍历所有可能的送货点位置 (必须在取货点之后)
            for d_pos in range(p_pos, route_len + 1):
                cost, feasible = objective.calculate_insertion_cost(
                    vehicle,
                    order.pickup_node,
                    order.delivery_node,
                    p_pos,
                    d_pos
                )
                
                if feasible:
                    insertion_costs.append((cost, vehicle.id, p_pos, d_pos))
    
    if len(insertion_costs) == 0:
        return float('inf'), None
    
    # 按成本排序
    insertion_costs.sort(key=lambda x: x[0])
    
    best_cost = insertion_costs[0][0]
    best_insertion = insertion_costs[0]
    
    # 计算regret
    if len(insertion_costs) >= k:
        kth_cost = insertion_costs[k - 1][0]
    else:
        kth_cost = insertion_costs[-1][0]
    
    regret = kth_cost - best_cost
    
    return regret, best_insertion
