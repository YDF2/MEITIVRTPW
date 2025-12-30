# -*- coding: utf-8 -*-
"""
分治策略 (Divide and Conquer) 求解器 - 支持多进程并行和Gurobi优化
用于大规模问题的高效求解

核心思路：
1. 使用 K-Means 将订单聚类成多个簇
2. 多进程并行求解每个簇
3. 合并解
4. 全局优化处理边界效应
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
import copy
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from models.solution import Solution
from models.node import Order, Node
from models.vehicle import Vehicle
from algorithm.base_solver import BaseSolver
from algorithm.alns import ALNS, solve_pdptw
from algorithm.objective import ObjectiveFunction
import config

# 尝试导入Gurobi求解器
try:
    from algorithm.gurobi_solver import solve_with_gurobi, GUROBI_AVAILABLE
except ImportError:
    GUROBI_AVAILABLE = False


def _solve_sub_problem_worker(args):
    """
    子问题求解工作函数（用于多进程）
    
    注意：这个函数必须在模块级别定义才能被pickle序列化
    
    Args:
        args: (cluster_id, orders_data, vehicles_data, depot_data, config_params)
        
    Returns:
        (cluster_id, solved_solution)
    """
    cluster_id, orders_data, vehicles_data, depot_data, solver_params = args
    
    try:
        # 重建对象
        depot = Node(
            node_id=depot_data['id'],
            node_type=depot_data['type'],
            x=depot_data['x'],
            y=depot_data['y']
        )
        
        # 重建订单
        orders = []
        for o_data in orders_data:
            pickup = Node(
                node_id=o_data['pickup']['id'],
                node_type=o_data['pickup']['type'],
                x=o_data['pickup']['x'],
                y=o_data['pickup']['y'],
                ready_time=o_data['pickup']['tw_start'],
                due_time=o_data['pickup']['tw_end'],
                service_time=o_data['pickup']['service_time']
            )
            delivery = Node(
                node_id=o_data['delivery']['id'],
                node_type=o_data['delivery']['type'],
                x=o_data['delivery']['x'],
                y=o_data['delivery']['y'],
                ready_time=o_data['delivery']['tw_start'],
                due_time=o_data['delivery']['tw_end'],
                service_time=o_data['delivery']['service_time']
            )
            order = Order(
                order_id=o_data['id'],
                pickup_node=pickup,
                delivery_node=delivery
            )
            orders.append(order)
        
        # 重建骑手
        vehicles = []
        for v_data in vehicles_data:
            vehicle = Vehicle(
                vehicle_id=v_data['id'],
                depot=depot,
                capacity=v_data['capacity']
            )
            vehicles.append(vehicle)
        
        # 创建子问题Solution
        sub_solution = Solution(vehicles=vehicles, orders=orders, depot=depot)
        
        # 根据配置选择求解器
        use_gurobi = solver_params.get('use_gurobi', False)
        sub_iterations = solver_params.get('sub_iterations', 300)
        random_seed = solver_params.get('random_seed', 42)
        
        if use_gurobi and GUROBI_AVAILABLE:
            # 使用Gurobi求解
            solved = solve_with_gurobi(sub_solution, time_limit=60, verbose=False)
            
            # 调试：检查Gurobi返回的解
            print(f"    [DEBUG Worker Gurobi] 簇{cluster_id} Gurobi求解完成")
            for v_idx, vehicle in enumerate(solved.vehicles):
                if len(vehicle.route) > 0:
                    print(f"      车辆{vehicle.id}: {len(vehicle.route)}个节点")
                    if len(vehicle.route) > 0:
                        print(f"        第1个节点: order_id={vehicle.route[0].order_id}")
        else:
            # 使用ALNS求解
            solved = solve_pdptw(
                sub_solution,
                max_iterations=sub_iterations,
                random_seed=random_seed + cluster_id,
                verbose=False
            )
        
        # 序列化结果 - 保存订单ID和节点类型信息
        # 计算已分配订单数（通过检查路径中的订单）
        assigned_order_ids = set()
        for vehicle in solved.vehicles:
            for node in vehicle.route:
                if node.order_id is not None:
                    assigned_order_ids.add(node.order_id)
        
        num_assigned = len(assigned_order_ids)
        
        result_data = {
            'cluster_id': cluster_id,
            'cost': solved.calculate_cost(),
            'vehicles': [],
            'num_assigned': num_assigned
        }
        
        for vehicle in solved.vehicles:
            if len(vehicle.route) > 0:
                # 打印原始路径信息
                print(f"    [DEBUG Worker] 簇{cluster_id} 车辆{vehicle.id} 原始路径长度: {len(vehicle.route)}")
                for i, node in enumerate(vehicle.route[:3]):  # 只打印前3个
                    print(f"      节点{i}: order_id={node.order_id}, type={node.node_type}")
                
                # 保存路径中每个节点的订单ID和类型（跳过depot）
                route_nodes = []
                for node in vehicle.route:
                    # 只保存订单节点，跳过depot
                    if node.order_id is not None:
                        node_info = {
                            'order_id': node.order_id,  # 订单ID
                            'is_pickup': node.is_pickup()  # 是否为取货点（调用方法）
                        }
                        route_nodes.append(node_info)
                
                route_data = {
                    'id': vehicle.id,
                    'route_nodes': route_nodes,  # 改用节点信息而非节点ID
                    'distance': vehicle.calculate_distance(),
                    'time_violation': vehicle.calculate_time_violation()
                }
                
                # 调试输出
                if len(route_nodes) > 0:
                    print(f"    [DEBUG] 簇{cluster_id} 车辆{vehicle.id}: {len(route_nodes)}个节点")
                
                result_data['vehicles'].append(route_data)
        
        return result_data
        
    except Exception as e:
        # 返回错误信息
        return {
            'cluster_id': cluster_id,
            'error': str(e),
            'cost': float('inf'),
            'vehicles': []
        }


class DivideAndConquerSolver(BaseSolver):
    """
    分治求解器 (使用Gurobi) - 支持多进程并行
    适用于大规模问题（>100订单）
    """
    
    def __init__(
        self,
        num_clusters: int = None,
        sub_iterations: int = 300,
        global_iterations: int = 50,
        random_seed: int = 42,
        verbose: bool = True,
        use_gurobi: bool = None,
        use_parallel: bool = True,
        max_workers: int = None,
        skip_global_optimization: bool = False
    ):
        """
        Args:
            num_clusters: 聚类数量（None时自动确定）
            sub_iterations: 子问题ALNS迭代次数
            global_iterations: 全局优化迭代次数
            random_seed: 随机种子
            verbose: 是否输出详细信息
            use_gurobi: 是否使用Gurobi（None时自动判断）
            use_parallel: 是否使用多进程并行
            max_workers: 最大工作进程数（None时使用CPU核心数）
            skip_global_optimization: 是否跳过全局优化（Gurobi解质量高时可跳过）
        """
        super().__init__(random_seed=random_seed, verbose=verbose)
        
        self.num_clusters = num_clusters
        self.sub_iterations = sub_iterations
        self.global_iterations = global_iterations
        self.skip_global_optimization = skip_global_optimization
        self.use_parallel = use_parallel
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)  # 留一个核心
        
        # 自动判断是否使用Gurobi
        if use_gurobi is None:
            self.use_gurobi = GUROBI_AVAILABLE
        else:
            self.use_gurobi = use_gurobi and GUROBI_AVAILABLE
        
        if self.verbose:
            if self.use_gurobi:
                print("  ✓ Gurobi 可用，将用于子问题求解")
            if self.use_parallel:
                print(f"  ✓ 多进程并行已启用，最大工作进程: {self.max_workers}")
        
    def solve(self, initial_solution: Solution) -> Solution:
        """
        分治求解主流程
        
        Args:
            initial_solution: 初始解（包含所有订单和骑手）
            
        Returns:
            优化后的解
        """
        num_orders = len(initial_solution.orders)
        num_vehicles = len(initial_solution.vehicles)
        
        if self.verbose:
            print("\n" + "="*70)
            print("  分治策略求解器 (Divide and Conquer)")
            print("="*70)
            print(f"订单数量: {num_orders}")
            print(f"骑手数量: {num_vehicles}")
        
        # 自动确定聚类数量
        if self.num_clusters is None:
            self.num_clusters = self._auto_determine_clusters(num_orders, num_vehicles)
        
        # 如果聚类数为1，直接使用标准ALNS
        if self.num_clusters == 1:
            if self.verbose:
                print(f"订单数量较少，使用标准ALNS算法...")
                print("-"*70)
            
            alns = ALNS(
                max_iterations=self.sub_iterations + self.global_iterations,
                random_seed=self.random_seed,
                verbose=self.verbose
            )
            return alns.solve(initial_solution)
        
        if self.verbose:
            print(f"聚类数量: {self.num_clusters}")
            print("-"*70)
        
        # 步骤1: 聚类
        if self.verbose:
            print("\n[步骤1] 对订单进行 K-Means 聚类...")
        
        time_start = time.time()
        clusters = self._cluster_orders(initial_solution)
        time_cluster = time.time() - time_start
        
        if self.verbose:
            print(f"  ✓ 聚类完成，耗时: {time_cluster:.2f}秒")
            for i, orders in clusters.items():
                print(f"    簇 {i}: {len(orders)} 订单")
        
        # 步骤2: 分配骑手到各个簇
        if self.verbose:
            print("\n[步骤2] 分配骑手到各簇...")
        
        time_start = time.time()
        cluster_vehicles = self._assign_vehicles_to_clusters(
            initial_solution, clusters
        )
        time_assign = time.time() - time_start
        
        if self.verbose:
            print(f"  ✓ 分配完成，耗时: {time_assign:.2f}秒")
            for i, vehicles in cluster_vehicles.items():
                print(f"    簇 {i}: {len(vehicles)} 骑手")
        
        # 步骤3: 分组求解（并行或串行）
        if self.verbose:
            print(f"\n[步骤3] {'并行' if self.use_parallel else '串行'}求解各簇...")
        
        time_start = time.time()
        if self.use_parallel:
            cluster_solutions = self._solve_clusters_parallel(
                initial_solution, clusters, cluster_vehicles
            )
        else:
            cluster_solutions = self._solve_clusters_sequential(
                initial_solution, clusters, cluster_vehicles
            )
        time_solve = time.time() - time_start
        
        if self.verbose:
            print(f"  ✓ 所有簇求解完成，耗时: {time_solve:.2f}秒")
        
        # 步骤4: 合并解
        if self.verbose:
            print("\n[步骤4] 合并各簇的解...")
        
        time_start = time.time()
        merged_solution = self._merge_solutions(
            initial_solution, cluster_solutions
        )
        time_merge = time.time() - time_start
        
        cost_before = merged_solution.calculate_cost()
        
        if self.verbose:
            print(f"  ✓ 合并完成，耗时: {time_merge:.2f}秒")
            print(f"  合并后成本: {cost_before:.2f}")
        
        # 步骤5: 全局优化（可选）
        if self.skip_global_optimization:
            if self.verbose:
                print("\n[步骤5] 跳过全局优化（Gurobi子解质量已足够高）")
            return merged_solution
        
        if self.verbose:
            print("\n[步骤5] 全局优化（处理边界效应）...")
        
        time_start = time.time()
        final_solution = self._global_optimization(merged_solution)
        time_global = time.time() - time_start
        
        cost_after = final_solution.calculate_cost()
        improvement = cost_before - cost_after
        
        if self.verbose:
            print(f"  ✓ 全局优化完成，耗时: {time_global:.2f}秒")
            print(f"  优化后成本: {cost_after:.2f}")
            print(f"  成本改进: {improvement:.2f} ({improvement/cost_before*100:.2f}%)")
            
            print("\n" + "="*70)
            print("  分治求解完成")
            print("="*70)
            total_time = time_cluster + time_assign + time_solve + time_merge + time_global
            print(f"总耗时: {total_time:.2f}秒")
            print(f"  - 聚类: {time_cluster:.2f}秒")
            print(f"  - 分配骑手: {time_assign:.2f}秒")
            print(f"  - 分组求解: {time_solve:.2f}秒")
            print(f"  - 合并: {time_merge:.2f}秒")
            print(f"  - 全局优化: {time_global:.2f}秒")
            print("="*70)
        
        return final_solution
    
    def _auto_determine_clusters(self, num_orders: int, num_vehicles: int) -> int:
        """
        自动确定聚类数量
        
        原则：
        - 每个簇最多50个订单
        - 每个簇至少10个骑手
        """
        # 基于订单数量（每簇最多50订单）
        clusters_by_orders = max(1, (num_orders + 49) // 50)  # 向上取整
        
        # 基于骑手数量（每簇至少10个骑手）
        clusters_by_vehicles = max(1, num_vehicles // 10)
        
        # 取较小值，确保每个簇有足够资源
        num_clusters = min(clusters_by_orders, clusters_by_vehicles)
        
        # 限制范围 [2, 20]，至少2个簇才有分治意义
        if num_orders < 50:
            num_clusters = 1  # 太小的问题不分治
        else:
            num_clusters = max(2, min(20, num_clusters))
        
        return num_clusters
    
    def _cluster_orders(
        self,
        solution: Solution
    ) -> Dict[int, List[Order]]:
        """
        使用 K-Means 对订单聚类
        
        Args:
            solution: 解对象
            
        Returns:
            {cluster_id: [orders]}
        """
        orders = list(solution.orders.values())
        
        if len(orders) <= self.num_clusters:
            # 订单数不足，每个订单一个簇
            return {i: [order] for i, order in enumerate(orders)}
        
        # 提取取货点坐标
        X = np.array([[o.pickup_node.x, o.pickup_node.y] for o in orders])
        
        # K-Means 聚类
        np.random.seed(self.random_seed)
        kmeans = KMeans(
            n_clusters=self.num_clusters,
            random_state=self.random_seed,
            n_init=10
        ).fit(X)
        
        # 分组
        clusters = {i: [] for i in range(self.num_clusters)}
        for idx, label in enumerate(kmeans.labels_):
            clusters[label].append(orders[idx])
        
        return clusters
    
    def _assign_vehicles_to_clusters(
        self,
        solution: Solution,
        clusters: Dict[int, List[Order]]
    ) -> Dict[int, List[Vehicle]]:
        """
        将骑手分配给各簇
        
        策略：基于簇的订单数量按比例分配，每簇至少1个骑手
        
        Args:
            solution: 解对象
            clusters: 订单簇
            
        Returns:
            {cluster_id: [vehicles]}
        """
        total_orders = sum(len(orders) for orders in clusters.values())
        total_vehicles = len(solution.vehicles)
        
        cluster_vehicles = {}
        assigned_count = 0
        
        # 按比例分配
        for i, orders in clusters.items():
            if i == self.num_clusters - 1:
                # 最后一个簇分配剩余所有骑手
                num_vehicles = total_vehicles - assigned_count
            else:
                # 按订单比例分配，至少1个
                ratio = len(orders) / total_orders
                num_vehicles = max(1, int(total_vehicles * ratio))
                num_vehicles = min(num_vehicles, total_vehicles - assigned_count - (self.num_clusters - i - 1))
            
            # 复制骑手对象
            vehicles = []
            for j in range(num_vehicles):
                idx = assigned_count + j
                if idx < total_vehicles:
                    original_vehicle = solution.vehicles[idx]
                    # 创建新的Vehicle对象，清空路径
                    new_vehicle = Vehicle(
                        vehicle_id=original_vehicle.id,
                        depot=solution.depot,
                        capacity=original_vehicle.capacity
                    )
                    vehicles.append(new_vehicle)
            
            cluster_vehicles[i] = vehicles
            assigned_count += len(vehicles)
        
        return cluster_vehicles
    
    def _serialize_for_parallel(
        self,
        cluster_id: int,
        orders: List[Order],
        vehicles: List[Vehicle],
        depot: Node
    ) -> Tuple:
        """
        将子问题数据序列化为可pickle的格式
        """
        # 序列化depot
        depot_data = {
            'id': depot.id,
            'type': depot.node_type,
            'x': depot.x,
            'y': depot.y
        }
        
        # 序列化orders
        orders_data = []
        for order in orders:
            o_data = {
                'id': order.id,
                'pickup': {
                    'id': order.pickup_node.id,
                    'type': order.pickup_node.node_type,
                    'x': order.pickup_node.x,
                    'y': order.pickup_node.y,
                    'tw_start': order.pickup_node.ready_time,
                    'tw_end': order.pickup_node.due_time,
                    'service_time': order.pickup_node.service_time
                },
                'delivery': {
                    'id': order.delivery_node.id,
                    'type': order.delivery_node.node_type,
                    'x': order.delivery_node.x,
                    'y': order.delivery_node.y,
                    'tw_start': order.delivery_node.ready_time,
                    'tw_end': order.delivery_node.due_time,
                    'service_time': order.delivery_node.service_time
                }
            }
            orders_data.append(o_data)
        
        # 序列化vehicles
        vehicles_data = []
        for vehicle in vehicles:
            v_data = {
                'id': vehicle.id,
                'capacity': vehicle.capacity
            }
            vehicles_data.append(v_data)
        
        # 求解器参数
        solver_params = {
            'use_gurobi': self.use_gurobi,
            'sub_iterations': self.sub_iterations,
            'random_seed': self.random_seed
        }
        
        return (cluster_id, orders_data, vehicles_data, depot_data, solver_params)
    
    def _solve_clusters_parallel(
        self,
        original_solution: Solution,
        clusters: Dict[int, List[Order]],
        cluster_vehicles: Dict[int, List[Vehicle]]
    ) -> Dict[int, Dict]:
        """
        并行求解所有簇
        """
        # 准备所有子问题的参数
        tasks = []
        for cluster_id in range(self.num_clusters):
            orders = clusters.get(cluster_id, [])
            vehicles = cluster_vehicles.get(cluster_id, [])
            
            if not orders or not vehicles:
                continue
            
            task_args = self._serialize_for_parallel(
                cluster_id, orders, vehicles, original_solution.depot
            )
            tasks.append(task_args)
        
        # 并行执行
        results = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_cluster = {
                executor.submit(_solve_sub_problem_worker, task): task[0] 
                for task in tasks
            }
            
            # 收集结果
            for future in as_completed(future_to_cluster):
                cluster_id = future_to_cluster[future]
                try:
                    result = future.result()
                    results[result['cluster_id']] = result
                    
                    if self.verbose and 'error' not in result:
                        print(f"    ✓ 簇 {result['cluster_id']} 完成, 成本: {result['cost']:.2f}")
                    elif 'error' in result:
                        print(f"    ✗ 簇 {result['cluster_id']} 失败: {result['error']}")
                        
                except Exception as e:
                    print(f"    ✗ 簇 {cluster_id} 发生异常: {str(e)}")
        
        return results
    
    def _solve_clusters_sequential(
        self,
        original_solution: Solution,
        clusters: Dict[int, List[Order]],
        cluster_vehicles: Dict[int, List[Vehicle]]
    ) -> Dict[int, Dict]:
        """
        串行求解所有簇（用于调试或无法并行的情况）
        """
        cluster_solutions = {}
        
        for cluster_id in range(self.num_clusters):
            orders = clusters.get(cluster_id, [])
            vehicles = cluster_vehicles.get(cluster_id, [])
            
            if not orders or not vehicles:
                continue
            
            if self.verbose:
                print(f"\n  求解簇 {cluster_id}: {len(orders)}订单, {len(vehicles)}骑手")
            
            # 创建子问题
            sub_solution = Solution(
                vehicles=vehicles,
                orders=orders,
                depot=original_solution.depot
            )
            
            # 使用 Gurobi 或 ALNS 求解
            time_start = time.time()
            if self.use_gurobi and GUROBI_AVAILABLE:
                solved_solution = solve_with_gurobi(sub_solution, time_limit=60, verbose=False)
            else:
                alns = ALNS(
                    max_iterations=self.sub_iterations,
                    random_seed=self.random_seed + cluster_id,
                    verbose=False
                )
                solved_solution = alns.solve(sub_solution)
            time_solve = time.time() - time_start
            
            cost = solved_solution.calculate_cost()
            stats = solved_solution.get_statistics()
            
            if self.verbose:
                print(f"    ✓ 簇 {cluster_id} 求解完成")
                print(f"      成本: {cost:.2f}, 耗时: {time_solve:.2f}秒")
                print(f"      已分配: {len(orders) - stats['num_unassigned']}/{len(orders)}")
            
            # 存储结果 - 使用相同的序列化格式
            # 计算已分配订单数
            assigned_order_ids = set()
            for vehicle in solved_solution.vehicles:
                for node in vehicle.route:
                    if node.order_id is not None:
                        assigned_order_ids.add(node.order_id)
            
            num_assigned = len(assigned_order_ids)
            
            result_data = {
                'cluster_id': cluster_id,
                'cost': cost,
                'vehicles': [],
                'num_assigned': num_assigned
            }
            
            for vehicle in solved_solution.vehicles:
                if len(vehicle.route) > 0:
                    # 保存路径中每个节点的订单ID和类型（跳过depot）
                    route_nodes = []
                    for node in vehicle.route:
                        # 只保存订单节点，跳过depot
                        if node.order_id is not None:
                            node_info = {
                                'order_id': node.order_id,
                                'is_pickup': node.is_pickup()  # 调用方法
                            }
                            route_nodes.append(node_info)
                    
                    route_data = {
                        'id': vehicle.id,
                        'route_nodes': route_nodes,
                        'distance': vehicle.calculate_distance(),
                        'time_violation': vehicle.calculate_time_violation()
                    }
                    result_data['vehicles'].append(route_data)
            
            cluster_solutions[cluster_id] = result_data
        
        return cluster_solutions
    
    def _merge_solutions(
        self,
        original_solution: Solution,
        cluster_results: Dict[int, Dict]
    ) -> Solution:
        """
        合并各簇的解
        
        Args:
            original_solution: 原始解
            cluster_results: 各簇的求解结果
            
        Returns:
            合并后的解
        """
        # 创建新解
        merged_solution = Solution(
            vehicles=copy.deepcopy(original_solution.vehicles),
            orders=list(original_solution.orders.values()),
            depot=original_solution.depot
        )
        
        # 清空所有骑手的路径
        for vehicle in merged_solution.vehicles:
            vehicle.route = []
        
        # 创建订单ID到节点的映射
        order_nodes = {}  # {order_id: {'pickup': node, 'delivery': node}}
        for order in merged_solution.orders.values():
            order_nodes[order.id] = {
                'pickup': order.pickup_node,
                'delivery': order.delivery_node
            }
        
        # 合并各簇的路径
        total_assigned = 0
        for cluster_id, result in cluster_results.items():
            if 'error' in result:
                if self.verbose:
                    print(f"  ! 簇 {cluster_id} 求解失败: {result.get('error')}")
                continue
            
            cluster_assigned = result.get('num_assigned', 0)
            total_assigned += cluster_assigned
            
            if self.verbose:
                print(f"    ✓ 簇 {cluster_id} 完成, 成本: {result['cost']:.2f}, 已分配: {cluster_assigned}订单")
            
            for v_data in result['vehicles']:
                vehicle_id = v_data['id']
                route_nodes_info = v_data.get('route_nodes', [])
                
                if self.verbose:
                    print(f"    [DEBUG] 处理车辆{vehicle_id}: {len(route_nodes_info)}个节点")
                
                # 找到对应的骑手
                vehicle = next((v for v in merged_solution.vehicles if v.id == vehicle_id), None)
                if vehicle is None:
                    if self.verbose:
                        print(f"  ! 警告: 找不到骑手 {vehicle_id}")
                    continue
                
                # 根据订单ID和节点类型恢复路径
                route = []
                for node_info in route_nodes_info:
                    order_id = node_info['order_id']
                    is_pickup = node_info['is_pickup']
                    
                    if order_id in order_nodes:
                        if is_pickup:
                            route.append(order_nodes[order_id]['pickup'])
                        else:
                            route.append(order_nodes[order_id]['delivery'])
                    else:
                        if self.verbose:
                            print(f"  ! 警告: 找不到订单 {order_id}")
                
                vehicle.route = route
        
        if self.verbose and total_assigned > 0:
            print(f"  总共分配: {total_assigned} 个订单")
        
        # 更新未分配订单列表
        assigned_order_ids = set()
        for vehicle in merged_solution.vehicles:
            for node in vehicle.route:
                if node.order_id is not None:
                    assigned_order_ids.add(node.order_id)
        
        merged_solution.unassigned_orders = [
            order for order in merged_solution.orders.values()
            if order.id not in assigned_order_ids
        ]
        
        return merged_solution
    
    def _global_optimization(self, solution: Solution) -> Solution:
        """
        全局优化，处理簇边界效应
        
        使用较少迭代的 ALNS 进行全局优化
        添加早停机制：如果连续20次迭代没有改进则提前结束
        
        Args:
            solution: 合并后的解
            
        Returns:
            优化后的解
        """
        # 使用更少的迭代次数，因为Gurobi子解质量已经很高
        effective_iterations = min(self.global_iterations, 50)
        
        alns = ALNS(
            max_iterations=effective_iterations,
            random_seed=self.random_seed + 999,
            verbose=False
        )
        
        optimized_solution = alns.solve(solution)
        
        return optimized_solution


def solve_large_scale(
    initial_solution: Solution,
    num_clusters: int = None,
    sub_iterations: int = 300,
    global_iterations: int = 100,
    random_seed: int = 42,
    verbose: bool = True,
    use_gurobi: bool = None,
    use_parallel: bool = True,
    max_workers: int = None
) -> Solution:
    """
    大规模问题求解的便捷函数
    
    Args:
        initial_solution: 初始解
        num_clusters: 聚类数量（None时自动确定）
        sub_iterations: 子问题迭代次数
        global_iterations: 全局优化迭代次数
        random_seed: 随机种子
        verbose: 是否输出详细信息
        use_gurobi: 是否使用Gurobi
        use_parallel: 是否使用多进程并行
        max_workers: 最大工作进程数
        
    Returns:
        优化后的解
    """
    solver = DivideAndConquerSolver(
        num_clusters=num_clusters,
        sub_iterations=sub_iterations,
        global_iterations=global_iterations,
        random_seed=random_seed,
        verbose=verbose,
        use_gurobi=use_gurobi,
        use_parallel=use_parallel,
        max_workers=max_workers
    )
    
    return solver.solve(initial_solution)
