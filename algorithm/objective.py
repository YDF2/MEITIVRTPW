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
        delivery_pos: int
    ) -> Tuple[float, bool]:
        """
        计算在指定位置插入订单后的成本增加量
        
        Args:
            vehicle: 目标骑手
            pickup_node: 取货点
            delivery_node: 送货点
            pickup_pos: 取货点插入位置
            delivery_pos: 送货点插入位置
        
        Returns:
            (cost_increase, is_feasible): 成本增加量和是否可行
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
        
        # 恢复原始路径
        vehicle.route = old_route
        vehicle.invalidate_cache()
        
        if not is_feasible:
            return float('inf'), False
        
        # 计算成本增加
        cost_increase = (
            self.w_distance * (new_distance - original_distance) +
            self.w_time_penalty * (new_violation - original_violation)
        )
        
        return cost_increase, True


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
