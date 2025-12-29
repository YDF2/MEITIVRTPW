# -*- coding: utf-8 -*-
"""
贪婪插入算法 (Greedy Insertion)
用于生成初始解
"""

from typing import List, Tuple, Optional, TYPE_CHECKING
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.solution import Solution
from models.vehicle import Vehicle
from models.node import Node, Order
from algorithm.objective import ObjectiveFunction

if TYPE_CHECKING:
    pass


class GreedyInsertion:
    """
    贪婪插入算法
    
    策略: 对于每个未分配订单, 找到成本增加最小的可行插入位置
    """
    
    def __init__(self, objective: ObjectiveFunction = None):
        self.objective = objective or ObjectiveFunction()
    
    def generate_initial_solution(self, solution: Solution) -> Solution:
        """
        使用贪婪插入生成初始解
        
        Args:
            solution: 包含未分配订单的初始解
        
        Returns:
            带有初始路径的解
        """
        # 复制解以避免修改原始数据
        current_solution = solution.copy()
        
        # 按照某种顺序处理订单 (可以按时间窗紧迫程度排序)
        orders_to_insert = list(current_solution.unassigned_orders)
        orders_to_insert.sort(key=lambda o: o.pickup_node.due_time)
        
        for order in orders_to_insert:
            best_insertion = self._find_best_insertion(current_solution, order)
            
            if best_insertion is not None:
                vehicle_id, p_pos, d_pos = best_insertion
                vehicle = current_solution.vehicles[vehicle_id]
                current_solution.assign_order(order, vehicle, p_pos, d_pos)
        
        return current_solution
    
    def _find_best_insertion(
        self, 
        solution: Solution, 
        order: Order
    ) -> Optional[Tuple[int, int, int]]:
        """
        为订单找到最佳插入位置
        
        Returns:
            (vehicle_id, pickup_pos, delivery_pos) 或 None
        """
        best_cost = float('inf')
        best_insertion = None
        
        for v_idx, vehicle in enumerate(solution.vehicles):
            route_len = len(vehicle.route)
            
            # 遍历所有可能的取货点位置
            for p_pos in range(route_len + 1):
                # 遍历所有可能的送货点位置
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
    
    def insert_order(
        self, 
        solution: Solution, 
        order: Order
    ) -> Tuple[bool, Optional[Tuple[int, int, int]]]:
        """
        将单个订单插入解中
        
        Returns:
            (success, insertion_info)
        """
        best_insertion = self._find_best_insertion(solution, order)
        
        if best_insertion is not None:
            vehicle_id, p_pos, d_pos = best_insertion
            vehicle = solution.vehicles[vehicle_id]
            success = solution.assign_order(order, vehicle, p_pos, d_pos)
            return success, best_insertion
        
        return False, None
    
    def insert_orders(
        self, 
        solution: Solution, 
        orders: List[Order]
    ) -> int:
        """
        插入多个订单
        
        Returns:
            成功插入的订单数量
        """
        inserted_count = 0
        
        # 按时间窗紧迫程度排序
        orders_sorted = sorted(orders, key=lambda o: o.pickup_node.due_time)
        
        for order in orders_sorted:
            success, _ = self.insert_order(solution, order)
            if success:
                inserted_count += 1
        
        return inserted_count


class RegretInsertion:
    """
    Regret-k 插入算法
    
    策略: 优先插入"后悔值"大的订单
    后悔值 = 次优插入成本 - 最优插入成本
    
    高后悔值意味着: 如果现在不插入，以后插入会更贵
    """
    
    def __init__(self, k: int = 2, objective: ObjectiveFunction = None):
        self.k = k  # regret-k 中的k
        self.objective = objective or ObjectiveFunction()
    
    def insert_orders(
        self, 
        solution: Solution, 
        orders: List[Order]
    ) -> int:
        """
        使用regret启发式插入多个订单
        
        Returns:
            成功插入的订单数量
        """
        inserted_count = 0
        remaining_orders = list(orders)
        
        while remaining_orders:
            # 计算每个订单的regret值
            regrets = []
            for order in remaining_orders:
                regret, best_insertion = self._calculate_regret(solution, order)
                regrets.append((regret, order, best_insertion))
            
            # 按regret值降序排序, 选择regret最大的订单
            regrets.sort(key=lambda x: x[0], reverse=True)
            
            # 尝试插入regret最大的订单
            best_regret, best_order, best_insertion = regrets[0]
            
            if best_insertion is not None:
                vehicle_id, p_pos, d_pos = best_insertion
                vehicle = solution.vehicles[vehicle_id]
                success = solution.assign_order(best_order, vehicle, p_pos, d_pos)
                
                if success:
                    inserted_count += 1
                    remaining_orders.remove(best_order)
                else:
                    # 插入失败，跳过此订单
                    remaining_orders.remove(best_order)
            else:
                # 无法插入，跳过此订单
                remaining_orders.remove(best_order)
        
        return inserted_count
    
    def _calculate_regret(
        self, 
        solution: Solution, 
        order: Order
    ) -> Tuple[float, Optional[Tuple[int, int, int]]]:
        """
        计算订单的regret-k值
        """
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
        
        # 按成本排序
        insertion_costs.sort(key=lambda x: x[0])
        
        best_cost = insertion_costs[0][0]
        best_insertion = (insertion_costs[0][1], insertion_costs[0][2], insertion_costs[0][3])
        
        # 计算regret
        if len(insertion_costs) >= self.k:
            kth_cost = insertion_costs[self.k - 1][0]
        else:
            kth_cost = insertion_costs[-1][0]
        
        regret = kth_cost - best_cost
        
        return regret, best_insertion


def generate_initial_solution(solution: Solution) -> Solution:
    """
    便捷函数: 使用贪婪插入生成初始解
    """
    greedy = GreedyInsertion()
    return greedy.generate_initial_solution(solution)
