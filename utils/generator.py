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
    
    def generate_depot(self, depot_id: int = 0, x: float = None, y: float = None) -> Node:
        """
        生成配送站节点
        
        Args:
            depot_id: 站点ID
            x: x坐标（如果为None则使用中心位置）
            y: y坐标（如果为None则使用中心位置）
        """
        if x is None:
            x = self.grid_size / 2
        if y is None:
            y = self.grid_size / 2
            
        return Node(
            node_id=depot_id,
            x=x,
            y=y,
            node_type=NodeType.DEPOT,
            demand=0,
            ready_time=0,
            due_time=self.time_horizon,
            service_time=0
        )
    
    def generate_depots(self, depot_locations: List[Tuple[float, float]] = None) -> List[Node]:
        """
        生成多个配送站节点
        
        Args:
            depot_locations: 站点位置列表 [(x, y), ...]，如果为None则使用config中的配置
        
        Returns:
            站点节点列表
        """
        if depot_locations is None:
            depot_locations = config.DEPOT_LOCATIONS
        
        depots = []
        for i, (x, y) in enumerate(depot_locations):
            depot = self.generate_depot(depot_id=i, x=x, y=y)
            depots.append(depot)
        
        return depots
    
    def find_nearest_depot(self, x: float, y: float, depots: List[Node]) -> Node:
        """
        找到距离指定坐标最近的配送站
        
        Args:
            x: x坐标
            y: y坐标
            depots: 配送站列表
        
        Returns:
            最近的配送站
        """
        min_dist = float('inf')
        nearest_depot = depots[0]
        
        for depot in depots:
            dist = abs(x - depot.x) + abs(y - depot.y)  # 曼哈顿距离
            if dist < min_dist:
                min_dist = dist
                nearest_depot = depot
        
        return nearest_depot
    
    def generate_pickup_locations(self, num_locations: int) -> List[Tuple[float, float, float, float]]:
        """
        生成取货点位置及时间窗
        返回: [(x, y, ready_time, due_time), ...]
        """
        locations = []
        for _ in range(num_locations):
            x = random.uniform(0, self.grid_size)
            y = random.uniform(0, self.grid_size)
            ready_time = random.uniform(0, self.time_horizon * 0.6)
            due_time = ready_time + self.time_window_width
            locations.append((x, y, ready_time, due_time))
        return locations
    
    def generate_order(self, order_id: int) -> Order:
        """
        生成一个订单 (包含取货点和送货点)
        
        关键约束:
        1. 取货点时间窗必须早于送货点时间窗
        2. 需要考虑行驶时间
        3. 顾客距离商家不超过5km (实际业务约束)
        """
        # 生成取货点 (商家) 坐标
        pickup_x = random.uniform(0, self.grid_size)
        pickup_y = random.uniform(0, self.grid_size)
        
        # 生成送货点 (顾客) 坐标
        # 顾客只能选择5km以内的商家
        max_delivery_dist = 5.0  # 5km限制
        
        # 生成送货点，确保不超过5km
        attempts = 0
        while attempts < 100:  # 最多尝试100次
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0.5, max_delivery_dist)  # 至少0.5km，最多5km
            
            delivery_x = pickup_x + distance * np.cos(angle)
            delivery_y = pickup_y + distance * np.sin(angle)
            
            # 确保在边界内
            if 0 <= delivery_x <= self.grid_size and 0 <= delivery_y <= self.grid_size:
                break
            attempts += 1
        else:
            # 如果100次都没成功，使用保守策略
            delivery_x = np.clip(pickup_x + random.uniform(-2, 2), 0, self.grid_size)
            delivery_y = np.clip(pickup_y + random.uniform(-2, 2), 0, self.grid_size)
        
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
        """生成多个订单（每个订单有独立的取货点）"""
        return [self.generate_order(i) for i in range(num_orders)]
    
    def generate_orders_with_shared_pickups(self, num_orders: int) -> List[Order]:
        """
        生成多个订单（共享取货点）
        取货点数量不超过订单数量的1/3，每个取货点可能有多个订单
        """
        # 取货点数量：不超过订单数的1/3
        num_pickups = max(1, num_orders // 3)
        pickup_locations = self.generate_pickup_locations(num_pickups)
        
        orders = []
        for i in range(num_orders):
            # 随机选择一个取货点
            pickup_x, pickup_y, pickup_ready, pickup_due = random.choice(pickup_locations)
            
            # 生成送货点 (顾客) 坐标 - 限制在5km以内
            max_delivery_dist = 5.0  # 5km限制
            
            # 使用极坐标生成，确保距离控制
            attempts = 0
            while attempts < 100:
                angle = random.uniform(0, 2 * np.pi)
                distance = random.uniform(0.5, max_delivery_dist)  # 0.5-5km
                
                delivery_x = pickup_x + distance * np.cos(angle)
                delivery_y = pickup_y + distance * np.sin(angle)
                
                # 确保在边界内
                if 0 <= delivery_x <= self.grid_size and 0 <= delivery_y <= self.grid_size:
                    break
                attempts += 1
            else:
                # 保守策略
                delivery_x = np.clip(pickup_x + random.uniform(-2, 2), 0, self.grid_size)
                delivery_y = np.clip(pickup_y + random.uniform(-2, 2), 0, self.grid_size)
            
            # 计算两点之间的行驶时间
            travel_dist = ((pickup_x - delivery_x) ** 2 + (pickup_y - delivery_y) ** 2) ** 0.5
            travel_time = travel_dist / self.vehicle_speed
            
            # 生成送货点时间窗（必须在取货完成 + 行驶时间之后）
            min_delivery_ready = pickup_ready + self.service_time_pickup + travel_time
            delivery_ready = max(min_delivery_ready, pickup_ready + random.uniform(10, 30))
            delivery_due = delivery_ready + self.time_window_width
            
            # 确保不超过时间跨度
            delivery_due = min(delivery_due, self.time_horizon)
            
            # 生成货物数量 (1-3单位)
            demand = random.randint(1, 3)
            
            # 创建取货点节点
            pickup_node_id = i * 2 + 1
            pickup_node = Node(
                node_id=pickup_node_id,
                x=pickup_x,
                y=pickup_y,
                node_type=NodeType.PICKUP,
                demand=demand,
                ready_time=pickup_ready,
                due_time=pickup_due,
                service_time=self.service_time_pickup,
                order_id=i
            )
            
            # 创建送货点节点
            delivery_node_id = i * 2 + 2
            delivery_node = Node(
                node_id=delivery_node_id,
                x=delivery_x,
                y=delivery_y,
                node_type=NodeType.DELIVERY,
                demand=-demand,
                ready_time=delivery_ready,
                due_time=delivery_due,
                service_time=self.service_time_delivery,
                order_id=i
            )
            
            # 创建订单
            order = Order(i, pickup_node, delivery_node, demand)
            orders.append(order)
        
        return orders
    
    def generate_vehicle(self, vehicle_id: int, depot: Node) -> Vehicle:
        """生成一个骑手"""
        vehicle = Vehicle(
            vehicle_id=vehicle_id,
            capacity=self.vehicle_capacity,
            speed=self.vehicle_speed,
            detour_factor=config.DETOUR_FACTOR,
            depot=depot
        )
        # 开放式VRP：骑手初始位置在配送站
        vehicle.current_location = depot
        return vehicle
    
    def generate_vehicles(self, num_vehicles: int, depot: Node) -> List[Vehicle]:
        """生成属于单个站点的多个骑手"""
        return [self.generate_vehicle(i, depot) for i in range(num_vehicles)]
    
    def generate_vehicles_multi_depot(self, depots: List[Node], vehicles_per_depot: int = None) -> List[Vehicle]:
        """
        生成多站点骑手
        
        Args:
            depots: 配送站列表
            vehicles_per_depot: 每个站点的骑手数量（如果为None则使用config配置）
        
        Returns:
            所有骑手列表
        """
        if vehicles_per_depot is None:
            vehicles_per_depot = config.NUM_VEHICLES  # 使用NUM_VEHICLES作为每站点骑手数
        
        all_vehicles = []
        vehicle_id_counter = 0
        
        for depot in depots:
            for _ in range(vehicles_per_depot):
                vehicle = self.generate_vehicle(vehicle_id_counter, depot)
                all_vehicles.append(vehicle)
                vehicle_id_counter += 1
        
        return all_vehicles
    
    def generate_instance(
        self, 
        num_orders: int = None, 
        num_vehicles: int = None,
        shared_pickups: bool = True,
        multi_depot: bool = True
    ) -> Tuple[List[Node], List[Order], List[Vehicle]]:
        """
        生成完整的问题实例
        
        Args:
            num_orders: 订单数量
            num_vehicles: 每个站点的骑手数量
            shared_pickups: 是否使用共享取货点（默认True）
            multi_depot: 是否使用多站点（默认True，使用5个站点）
        
        Returns:
            (depots, orders, vehicles)
        """
        num_orders = num_orders or config.NUM_ORDERS
        num_vehicles = num_vehicles or config.NUM_VEHICLES
        
        if multi_depot:
            # 多站点模式
            depots = self.generate_depots()
            vehicles = self.generate_vehicles_multi_depot(depots, num_vehicles)
        else:
            # 单站点模式（向后兼容）
            depots = [self.generate_depot()]
            vehicles = self.generate_vehicles(num_vehicles, depots[0])
        
        # 生成订单
        if shared_pickups:
            orders = self.generate_orders_with_shared_pickups(num_orders)
        else:
            orders = self.generate_orders(num_orders)
        
        return depots, orders, vehicles
    
    def generate_solution(
        self, 
        num_orders: int = None, 
        num_vehicles: int = None,
        shared_pickups: bool = True,
        multi_depot: bool = True
    ) -> Solution:
        """
        生成包含问题实例的空解
        
        Args:
            num_orders: 订单数量
            num_vehicles: 每个站点的骑手数量
            shared_pickups: 是否使用共享取货点
            multi_depot: 是否使用多站点（默认True）
        
        Returns:
            初始化的Solution对象（所有订单未分配）
        """
        depots, orders, vehicles = self.generate_instance(
            num_orders, num_vehicles, shared_pickups, multi_depot
        )
        
        # 注意：Solution构造函数需要修改以支持多站点
        # 这里使用第一个depot作为主depot（向后兼容）
        solution = Solution(vehicles, orders, depots[0])
        
        # 存储所有站点信息
        solution.depots = depots
        
        return solution
    
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
    clustered: bool = False,
    multi_depot: bool = True
) -> Solution:
    """
    便捷函数: 生成问题实例
    
    Args:
        num_orders: 订单数量
        num_vehicles: 骑手数量（如果multi_depot=True，则为每站点骑手数）
        random_seed: 随机种子
        clustered: 是否使用聚类分布（默认False）
        multi_depot: 是否使用多站点模式（默认True，5个站点）
    
    Returns:
        初始化的Solution对象
    """
    generator = DataGenerator(random_seed=random_seed)
    
    if clustered:
        return generator.generate_clustered_instance(num_orders, num_vehicles)
    else:
        return generator.generate_solution(num_orders, num_vehicles, multi_depot=multi_depot)
