# -*- coding: utf-8 -*-
"""
随机数据生成器 (Data Generator)
生成PDPTW问题的测试数据
"""

from typing import List, Tuple, Dict, Optional
import random
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.node import Node, NodeType, Order
from models.vehicle import Vehicle
from models.solution import Solution
import config


class DataGenerator:
    """
    随机数据生成器
    
    生成包含配送站、商家、顾客、骑手的测试数据
    """
    
    def __init__(
        self,
        grid_size: float = None,
        time_horizon: float = None,
        service_time_pickup: float = None,
        service_time_delivery: float = None,
        time_window_width: float = None,
        vehicle_capacity: float = None,
        vehicle_speed: float = None,
        random_seed: int = None
    ):
        self.grid_size = grid_size or config.GRID_SIZE
        self.time_horizon = time_horizon or config.TIME_HORIZON
        self.service_time_pickup = service_time_pickup or config.SERVICE_TIME_PICKUP
        self.service_time_delivery = service_time_delivery or config.SERVICE_TIME_DELIVERY
        self.time_window_width = time_window_width or config.TIME_WINDOW_WIDTH
        self.vehicle_capacity = vehicle_capacity or config.VEHICLE_CAPACITY
        self.vehicle_speed = vehicle_speed or config.VEHICLE_SPEED
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def generate_depot(self) -> Node:
        """生成配送站节点 (通常位于中心位置)"""
        return Node(
            node_id=0,
            x=self.grid_size / 2,
            y=self.grid_size / 2,
            node_type=NodeType.DEPOT,
            demand=0,
            ready_time=0,
            due_time=self.time_horizon,
            service_time=0
        )
    
    def generate_order(self, order_id: int) -> Order:
        """
        生成一个订单 (包含取货点和送货点)
        
        关键约束:
        1. 取货点时间窗必须早于送货点时间窗
        2. 需要考虑行驶时间
        """
        # 生成取货点 (商家) 坐标
        pickup_x = random.uniform(0, self.grid_size)
        pickup_y = random.uniform(0, self.grid_size)
        
        # 生成送货点 (顾客) 坐标
        # 通常顾客距离商家不会太远
        max_delivery_dist = self.grid_size * 0.5
        delivery_x = pickup_x + random.uniform(-max_delivery_dist, max_delivery_dist)
        delivery_y = pickup_y + random.uniform(-max_delivery_dist, max_delivery_dist)
        
        # 确保在边界内
        delivery_x = max(0, min(self.grid_size, delivery_x))
        delivery_y = max(0, min(self.grid_size, delivery_y))
        
        # 计算两点之间的行驶时间
        travel_dist = ((pickup_x - delivery_x) ** 2 + (pickup_y - delivery_y) ** 2) ** 0.5
        travel_time = travel_dist / self.vehicle_speed
        
        # 生成取货点时间窗
        # 出餐时间在时间跨度的前80%内
        pickup_ready = random.uniform(0, self.time_horizon * 0.6)
        pickup_due = pickup_ready + self.time_window_width
        
        # 生成送货点时间窗
        # 必须在取货完成 + 行驶时间之后
        min_delivery_ready = pickup_ready + self.service_time_pickup + travel_time
        delivery_ready = max(min_delivery_ready, pickup_ready + random.uniform(10, 30))
        delivery_due = delivery_ready + self.time_window_width
        
        # 确保不超过时间跨度
        delivery_due = min(delivery_due, self.time_horizon)
        
        # 生成货物数量 (1-3单位)
        demand = random.randint(1, 3)
        
        # 创建取货点节点
        pickup_node_id = order_id * 2 + 1
        pickup_node = Node(
            node_id=pickup_node_id,
            x=pickup_x,
            y=pickup_y,
            node_type=NodeType.PICKUP,
            demand=demand,
            ready_time=pickup_ready,
            due_time=pickup_due,
            service_time=self.service_time_pickup,
            order_id=order_id
        )
        
        # 创建送货点节点
        delivery_node_id = order_id * 2 + 2
        delivery_node = Node(
            node_id=delivery_node_id,
            x=delivery_x,
            y=delivery_y,
            node_type=NodeType.DELIVERY,
            demand=-demand,
            ready_time=delivery_ready,
            due_time=delivery_due,
            service_time=self.service_time_delivery,
            order_id=order_id
        )
        
        # 创建订单
        order = Order(order_id, pickup_node, delivery_node, demand)
        
        return order
    
    def generate_orders(self, num_orders: int) -> List[Order]:
        """生成多个订单"""
        return [self.generate_order(i) for i in range(num_orders)]
    
    def generate_vehicle(self, vehicle_id: int, depot: Node) -> Vehicle:
        """生成一个骑手"""
        return Vehicle(
            vehicle_id=vehicle_id,
            capacity=self.vehicle_capacity,
            speed=self.vehicle_speed,
            depot=depot
        )
    
    def generate_vehicles(self, num_vehicles: int, depot: Node) -> List[Vehicle]:
        """生成多个骑手"""
        return [self.generate_vehicle(i, depot) for i in range(num_vehicles)]
    
    def generate_instance(
        self, 
        num_orders: int = None, 
        num_vehicles: int = None
    ) -> Tuple[Node, List[Order], List[Vehicle]]:
        """
        生成完整的问题实例
        
        Returns:
            (depot, orders, vehicles)
        """
        num_orders = num_orders or config.NUM_ORDERS
        num_vehicles = num_vehicles or config.NUM_VEHICLES
        
        depot = self.generate_depot()
        orders = self.generate_orders(num_orders)
        vehicles = self.generate_vehicles(num_vehicles, depot)
        
        return depot, orders, vehicles
    
    def generate_solution(
        self, 
        num_orders: int = None, 
        num_vehicles: int = None
    ) -> Solution:
        """
        生成包含问题实例的空解
        
        Returns:
            初始化的Solution对象 (所有订单未分配)
        """
        depot, orders, vehicles = self.generate_instance(num_orders, num_vehicles)
        return Solution(vehicles, orders, depot)
    
    def generate_clustered_instance(
        self,
        num_orders: int = None,
        num_vehicles: int = None,
        num_clusters: int = 4
    ) -> Solution:
        """
        生成聚类分布的问题实例
        
        商家和顾客分布在几个聚类中心附近
        """
        num_orders = num_orders or config.NUM_ORDERS
        num_vehicles = num_vehicles or config.NUM_VEHICLES
        
        depot = self.generate_depot()
        
        # 生成聚类中心
        cluster_centers = [
            (random.uniform(self.grid_size * 0.2, self.grid_size * 0.8),
             random.uniform(self.grid_size * 0.2, self.grid_size * 0.8))
            for _ in range(num_clusters)
        ]
        
        orders = []
        for i in range(num_orders):
            # 选择一个聚类中心
            center_x, center_y = random.choice(cluster_centers)
            
            # 在聚类中心附近生成点
            std = self.grid_size * 0.1
            pickup_x = np.clip(np.random.normal(center_x, std), 0, self.grid_size)
            pickup_y = np.clip(np.random.normal(center_y, std), 0, self.grid_size)
            
            # 送货点也在附近
            delivery_x = np.clip(np.random.normal(pickup_x, std * 0.5), 0, self.grid_size)
            delivery_y = np.clip(np.random.normal(pickup_y, std * 0.5), 0, self.grid_size)
            
            # 时间窗
            travel_dist = ((pickup_x - delivery_x) ** 2 + (pickup_y - delivery_y) ** 2) ** 0.5
            travel_time = travel_dist / self.vehicle_speed
            
            pickup_ready = random.uniform(0, self.time_horizon * 0.6)
            pickup_due = pickup_ready + self.time_window_width
            
            min_delivery_ready = pickup_ready + self.service_time_pickup + travel_time
            delivery_ready = max(min_delivery_ready, pickup_ready + random.uniform(10, 30))
            delivery_due = min(delivery_ready + self.time_window_width, self.time_horizon)
            
            demand = random.randint(1, 3)
            
            pickup_node = Node(
                node_id=i * 2 + 1,
                x=pickup_x,
                y=pickup_y,
                node_type=NodeType.PICKUP,
                demand=demand,
                ready_time=pickup_ready,
                due_time=pickup_due,
                service_time=self.service_time_pickup,
                order_id=i
            )
            
            delivery_node = Node(
                node_id=i * 2 + 2,
                x=delivery_x,
                y=delivery_y,
                node_type=NodeType.DELIVERY,
                demand=-demand,
                ready_time=delivery_ready,
                due_time=delivery_due,
                service_time=self.service_time_delivery,
                order_id=i
            )
            
            orders.append(Order(i, pickup_node, delivery_node, demand))
        
        vehicles = self.generate_vehicles(num_vehicles, depot)
        
        return Solution(vehicles, orders, depot)


def generate_problem_instance(
    num_orders: int = None,
    num_vehicles: int = None,
    random_seed: int = None,
    clustered: bool = False
) -> Solution:
    """
    便捷函数: 生成问题实例
    
    Args:
        num_orders: 订单数量
        num_vehicles: 骑手数量
        random_seed: 随机种子
        clustered: 是否使用聚类分布
    
    Returns:
        初始化的Solution对象
    """
    generator = DataGenerator(random_seed=random_seed)
    
    if clustered:
        return generator.generate_clustered_instance(num_orders, num_vehicles)
    else:
        return generator.generate_solution(num_orders, num_vehicles)
