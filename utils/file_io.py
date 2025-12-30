# -*- coding: utf-8 -*-
"""
文件读写工具 (File I/O)
支持JSON格式的数据保存和加载
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.solution import Solution
from models.vehicle import Vehicle
from models.node import Node, NodeType, Order
import config


def save_solution_to_json(
    solution: Solution,
    filename: str,
    output_dir: str = None,
    include_statistics: bool = True,
    solver_name: str = None,
    solve_time: float = None,
    generation_time: float = None,
    additional_info: dict = None
) -> str:
    """
    将解保存为JSON文件
    
    Args:
        solution: 解对象
        filename: 文件名 (不含路径)
        output_dir: 输出目录
        include_statistics: 是否包含统计信息
        solver_name: 求解器名称 (ALNS)
        solve_time: 求解耗时(秒)
        generation_time: 问题生成耗时(秒)
        additional_info: 其他额外信息
    
    Returns:
        保存的文件路径
    """
    output_dir = output_dir or config.OUTPUT_DIR
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建JSON数据
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_orders": len(solution.orders),
            "num_vehicles": len(solution.vehicles),
            "solver": solver_name or "Unknown",
            "solve_time_seconds": solve_time,
            "generation_time_seconds": generation_time,
            "total_time_seconds": (solve_time or 0) + (generation_time or 0)
        },
        "depot": _node_to_dict(solution.depot),
        "orders": [
            {
                "id": order.id,
                "demand": order.demand,
                "pickup": _node_to_dict(order.pickup_node),
                "delivery": _node_to_dict(order.delivery_node)
            }
            for order in solution.orders.values()
        ],
        "vehicles": [
            {
                "id": v.id,
                "capacity": v.capacity,
                "speed": v.speed,
                "detour_factor": v.detour_factor,
                "route": [_node_to_dict(n) for n in v.route],
                "route_distance": v.calculate_distance(),
                "route_time_violation": v.calculate_time_violation(),
                "num_orders": len(v.get_order_ids())
            }
            for v in solution.vehicles
        ],
        "unassigned_orders": [o.id for o in solution.unassigned_orders]
    }
    
    if include_statistics:
        data["statistics"] = solution.get_statistics()
    
    # 添加额外信息
    if additional_info:
        data["additional_info"] = additional_info
    
    # 保存文件
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"解已保存至: {filepath}")
    return filepath


def _node_to_dict(node: Node) -> Dict:
    """将节点转换为字典"""
    return {
        "id": node.id,
        "x": node.x,
        "y": node.y,
        "type": node.node_type.name,
        "demand": node.demand,
        "ready_time": node.ready_time,
        "due_time": node.due_time,
        "service_time": node.service_time,
        "pair_id": node.pair_id,
        "order_id": node.order_id
    }


def load_solution_from_json(filepath: str) -> Solution:
    """
    从JSON文件加载解
    
    Args:
        filepath: JSON文件路径
    
    Returns:
        Solution对象
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 重建depot
    depot = _dict_to_node(data["depot"])
    
    # 重建订单
    orders = []
    for order_data in data["orders"]:
        pickup_node = _dict_to_node(order_data["pickup"])
        delivery_node = _dict_to_node(order_data["delivery"])
        order = Order(
            order_id=order_data["id"],
            pickup_node=pickup_node,
            delivery_node=delivery_node,
            demand=order_data["demand"]
        )
        orders.append(order)
    
    # 重建骑手
    vehicles = []
    for v_data in data["vehicles"]:
        vehicle = Vehicle(
            vehicle_id=v_data["id"],
            capacity=v_data["capacity"],
            speed=v_data["speed"],
            detour_factor=v_data.get("detour_factor", config.DETOUR_FACTOR),
            depot=depot
        )
        # 重建路径
        vehicle.route = [_dict_to_node(n) for n in v_data["route"]]
        vehicles.append(vehicle)
    
    # 创建解
    solution = Solution(vehicles, orders, depot)
    
    # 更新未分配订单
    unassigned_ids = set(data.get("unassigned_orders", []))
    solution.unassigned_orders = [
        o for o in orders if o.id in unassigned_ids
    ]
    
    print(f"解已从 {filepath} 加载")
    return solution


def _dict_to_node(data: Dict) -> Node:
    """将字典转换为节点"""
    node_type = NodeType[data["type"]]
    return Node(
        node_id=data["id"],
        x=data["x"],
        y=data["y"],
        node_type=node_type,
        demand=data["demand"],
        ready_time=data["ready_time"],
        due_time=data["due_time"],
        service_time=data["service_time"],
        pair_id=data.get("pair_id"),
        order_id=data.get("order_id")
    )


def save_experiment_results(
    experiment_name: str,
    initial_solution: Solution,
    final_solution: Solution,
    alns_stats: Dict,
    cost_history: List[float],
    output_dir: str = None
) -> str:
    """
    保存完整实验结果
    
    Args:
        experiment_name: 实验名称
        initial_solution: 初始解
        final_solution: 最终解
        alns_stats: ALNS统计信息
        cost_history: 成本历史
        output_dir: 输出目录
    
    Returns:
        实验结果目录路径
    """
    output_dir = output_dir or config.OUTPUT_DIR
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 保存初始解
    save_solution_to_json(
        initial_solution,
        "initial_solution.json",
        experiment_dir
    )
    
    # 保存最终解
    save_solution_to_json(
        final_solution,
        "final_solution.json",
        experiment_dir
    )
    
    # 保存ALNS统计信息
    stats_data = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "alns_statistics": alns_stats,
        "cost_history": cost_history
    }
    
    stats_path = os.path.join(experiment_dir, "experiment_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)
    
    print(f"实验结果已保存至: {experiment_dir}")
    return experiment_dir


def load_problem_from_json(filepath: str) -> Solution:
    """
    从JSON文件加载问题实例 (不含路径规划)
    
    用于加载标准测试问题
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    depot = _dict_to_node(data["depot"])
    
    orders = []
    for order_data in data["orders"]:
        pickup_node = _dict_to_node(order_data["pickup"])
        delivery_node = _dict_to_node(order_data["delivery"])
        order = Order(
            order_id=order_data["id"],
            pickup_node=pickup_node,
            delivery_node=delivery_node,
            demand=order_data["demand"]
        )
        orders.append(order)
    
    num_vehicles = data.get("num_vehicles", config.NUM_VEHICLES)
    vehicles = [
        Vehicle(
            vehicle_id=i,
            capacity=config.VEHICLE_CAPACITY,
            speed=config.VEHICLE_SPEED,
            detour_factor=config.DETOUR_FACTOR,
            depot=depot
        )
        for i in range(num_vehicles)
    ]
    
    return Solution(vehicles, orders, depot)


def save_problem_to_json(
    solution: Solution,
    filename: str,
    output_dir: str = None
) -> str:
    """
    保存问题实例 (不含路径规划)
    
    用于保存可重复使用的测试问题
    """
    output_dir = output_dir or config.DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    data = {
        "num_orders": len(solution.orders),
        "num_vehicles": len(solution.vehicles),
        "depot": _node_to_dict(solution.depot),
        "orders": [
            {
                "id": order.id,
                "demand": order.demand,
                "pickup": _node_to_dict(order.pickup_node),
                "delivery": _node_to_dict(order.delivery_node)
            }
            for order in solution.orders.values()
        ]
    }
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"问题实例已保存至: {filepath}")
    return filepath
