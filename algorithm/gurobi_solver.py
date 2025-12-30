# -*- coding: utf-8 -*-
"""
Gurobi 优化求解器适配器
用于求解 PDPTW 的 MIP 模型（适合小规模子问题）
"""

from typing import List, Dict, Tuple, Optional
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

from models.solution import Solution
from models.node import Order, Node
from models.vehicle import Vehicle
import config


class GurobiPDPTWSolver:
    """
    使用 Gurobi 求解 PDPTW 的 MIP 模型
    适合小规模子问题（<50订单）
    """
    
    def __init__(self, time_limit: int = 60, verbose: bool = False):
        """
        Args:
            time_limit: 求解时间限制（秒）
            verbose: 是否输出详细信息
        """
        if not GUROBI_AVAILABLE:
            raise ImportError("Gurobi 未安装或许可证无效")
        
        self.time_limit = time_limit
        self.verbose = verbose
    
    def solve(self, solution: Solution) -> Solution:
        """
        使用 Gurobi 求解 PDPTW
        
        Args:
            solution: 初始解（包含订单和骑手）
            
        Returns:
            优化后的解
        """
        if self.verbose:
            print(f"  使用 Gurobi 求解子问题...")
        
        try:
            # 创建模型
            model = gp.Model("PDPTW")
            
            if not self.verbose:
                model.setParam('OutputFlag', 0)
            
            model.setParam('TimeLimit', self.time_limit)
            model.setParam('MIPGap', 0.05)  # 5% gap
            
            # 准备数据
            depot = solution.depot
            orders = list(solution.orders.values())
            vehicles = solution.vehicles
            
            num_orders = len(orders)
            num_vehicles = len(vehicles)
            
            # 节点列表: 0=depot, 1..n=pickups, n+1..2n=deliveries
            nodes = [depot]
            for order in orders:
                nodes.append(order.pickup_node)
            for order in orders:
                nodes.append(order.delivery_node)
            
            num_nodes = len(nodes)
            
            # 距离矩阵
            dist = {}
            for i in range(num_nodes):
                for j in range(num_nodes):
                    dist[i, j] = nodes[i].distance_to(nodes[j])
            
            # 决策变量
            # x[i,j,k] = 1 if vehicle k goes from node i to node j
            x = {}
            for k in range(num_vehicles):
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:
                            x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}_{k}')
            
            # 到达时间变量
            t = {}
            for k in range(num_vehicles):
                for i in range(num_nodes):
                    t[i, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f't_{i}_{k}')
            
            # 目标函数：最小化总距离
            obj = gp.quicksum(dist[i, j] * x[i, j, k] 
                            for k in range(num_vehicles)
                            for i in range(num_nodes)
                            for j in range(num_nodes)
                            if i != j)
            
            model.setObjective(obj, GRB.MINIMIZE)
            
            # 约束1: 每个取货点恰好访问一次
            for o_idx, order in enumerate(orders):
                pickup_idx = 1 + o_idx
                model.addConstr(
                    gp.quicksum(x[i, pickup_idx, k] 
                              for k in range(num_vehicles)
                              for i in range(num_nodes)
                              if i != pickup_idx) == 1,
                    name=f'visit_pickup_{o_idx}'
                )
            
            # 约束2: 每个送货点恰好访问一次
            for o_idx, order in enumerate(orders):
                delivery_idx = 1 + num_orders + o_idx
                model.addConstr(
                    gp.quicksum(x[i, delivery_idx, k] 
                              for k in range(num_vehicles)
                              for i in range(num_nodes)
                              if i != delivery_idx) == 1,
                    name=f'visit_delivery_{o_idx}'
                )
            
            # 约束3: 流守恒
            for k in range(num_vehicles):
                for j in range(num_nodes):
                    model.addConstr(
                        gp.quicksum(x[i, j, k] for i in range(num_nodes) if i != j) ==
                        gp.quicksum(x[j, i, k] for i in range(num_nodes) if i != j),
                        name=f'flow_{j}_{k}'
                    )
            
            # 约束4: 每个骑手从depot出发
            for k in range(num_vehicles):
                model.addConstr(
                    gp.quicksum(x[0, j, k] for j in range(1, num_nodes)) <= 1,
                    name=f'start_{k}'
                )
            
            # 约束5: 每个骑手回到depot
            for k in range(num_vehicles):
                model.addConstr(
                    gp.quicksum(x[i, 0, k] for i in range(1, num_nodes)) <= 1,
                    name=f'end_{k}'
                )
            
            # 约束6: 取货必须在送货之前（同一订单，同一骑手）
            for k in range(num_vehicles):
                for o_idx in range(num_orders):
                    pickup_idx = 1 + o_idx
                    delivery_idx = 1 + num_orders + o_idx
                    
                    # 如果骑手k服务了这个订单
                    model.addConstr(
                        gp.quicksum(x[i, pickup_idx, k] for i in range(num_nodes) if i != pickup_idx) ==
                        gp.quicksum(x[i, delivery_idx, k] for i in range(num_nodes) if i != delivery_idx),
                        name=f'pd_pair_{o_idx}_{k}'
                    )
            
            # 约束7: 时间窗约束（简化版）
            M = 10000  # 大M
            for k in range(num_vehicles):
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:
                            # t[j,k] >= t[i,k] + service_time + travel_time - M(1-x[i,j,k])
                            travel_time = dist[i, j] / config.VEHICLE_SPEED
                            service_time = nodes[i].service_time if i > 0 else 0
                            
                            model.addConstr(
                                t[j, k] >= t[i, k] + service_time + travel_time - M * (1 - x[i, j, k]),
                                name=f'time_{i}_{j}_{k}'
                            )
            
            # 约束8: 取货在送货之前（时间）
            for k in range(num_vehicles):
                for o_idx in range(num_orders):
                    pickup_idx = 1 + o_idx
                    delivery_idx = 1 + num_orders + o_idx
                    
                    model.addConstr(
                        t[delivery_idx, k] >= t[pickup_idx, k] + nodes[pickup_idx].service_time,
                        name=f'pd_time_{o_idx}_{k}'
                    )
            
            # 求解
            model.optimize()
            
            # 解析结果
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                return self._build_solution(solution, model, x, nodes, orders, vehicles)
            else:
                if self.verbose:
                    print(f"  Gurobi 未找到可行解，使用原始解")
                return solution
                
        except Exception as e:
            if self.verbose:
                print(f"  Gurobi 求解失败: {str(e)}, 使用原始解")
            return solution
    
    def _build_solution(self, original_solution, model, x, nodes, orders, vehicles):
        """从 Gurobi 解构建 Solution 对象"""
        try:
            # 直接修改original_solution中的vehicles，而不是传入的深拷贝
            for vehicle in original_solution.vehicles:
                vehicle.route = []
            
            # 提取路径
            num_vehicles = len(vehicles)
            num_nodes = len(nodes)
            
            for k in range(num_vehicles):
                route = []
                current = 0  # 从depot开始
                visited = set([0])
                
                while True:
                    next_node = None
                    for j in range(num_nodes):
                        if j not in visited and current != j:
                            try:
                                if x[current, j, k].X > 0.5:
                                    next_node = j
                                    break
                            except:
                                pass
                    
                    if next_node is None or next_node == 0:
                        break
                    
                    route.append(nodes[next_node])
                    visited.add(next_node)
                    current = next_node
                
                if route:
                    original_solution.vehicles[k].route = route
            
            # 更新未分配订单列表
            assigned_order_ids = set()
            for vehicle in original_solution.vehicles:
                for node in vehicle.route:
                    if node.order_id is not None:
                        assigned_order_ids.add(node.order_id)
            
            original_solution.unassigned_orders = [
                order for order in original_solution.orders.values()
                if order.id not in assigned_order_ids
            ]
            
            return original_solution
            
        except Exception as e:
            return original_solution


def solve_with_gurobi(solution: Solution, time_limit: int = 60, verbose: bool = False) -> Solution:
    """
    使用 Gurobi 求解的便捷函数
    
    Args:
        solution: 初始解
        time_limit: 时间限制（秒）
        verbose: 是否输出详细信息
        
    Returns:
        优化后的解
    """
    if not GUROBI_AVAILABLE:
        # 如果Gurobi不可用，回退到ALNS
        from algorithm.alns import solve_pdptw
        return solve_pdptw(solution, max_iterations=300, verbose=verbose)
    
    solver = GurobiPDPTWSolver(time_limit=time_limit, verbose=verbose)
    return solver.solve(solution)
