# -*- coding: utf-8 -*-
"""
骑手模型 (Vehicle)
表示配送骑手及其路径
"""

from typing import List, Optional, TYPE_CHECKING
import copy as copy_module

if TYPE_CHECKING:
    from .node import Node


class Vehicle:
    """
    骑手类
    
    Attributes:
        id: 骑手唯一标识
        capacity: 最大载重能力
        speed: 行驶速度
        depot: 起始配送站节点
        route: 访问节点列表 (不包含起始和结束的depot)
    """
    
    def __init__(
        self,
        vehicle_id: int,
        capacity: float,
        speed: float = 1.0,
        depot: Optional['Node'] = None
    ):
        self.id = vehicle_id
        self.capacity = capacity
        self.speed = speed
        self.depot = depot
        self.route: List['Node'] = []  # 路径中的节点列表
        
        # 缓存信息 (需要在路径变化时更新)
        self._distance: Optional[float] = None
        self._time_violation: Optional[float] = None
        self._is_valid: Optional[bool] = None
    
    def invalidate_cache(self):
        """使缓存失效"""
        self._distance = None
        self._time_violation = None
        self._is_valid = None
    
    @property
    def full_route(self) -> List['Node']:
        """完整路径 (包含起始和结束的depot)"""
        if self.depot is None:
            return self.route
        return [self.depot] + self.route + [self.depot]
    
    def calculate_distance(self) -> float:
        """计算路径总距离"""
        if self._distance is not None:
            return self._distance
            
        if len(self.route) == 0:
            self._distance = 0.0
            return 0.0
        
        total_distance = 0.0
        full_route = self.full_route
        
        for i in range(len(full_route) - 1):
            total_distance += full_route[i].distance_to(full_route[i + 1])
        
        self._distance = total_distance
        return total_distance
    
    def calculate_time_violation(self) -> float:
        """
        计算时间窗违反量 (软约束)
        返回: 总超时时间
        """
        if self._time_violation is not None:
            return self._time_violation
            
        if len(self.route) == 0:
            self._time_violation = 0.0
            return 0.0
        
        total_violation = 0.0
        current_time = 0.0
        full_route = self.full_route
        
        for i in range(1, len(full_route)):
            prev_node = full_route[i - 1]
            curr_node = full_route[i]
            
            # 行驶时间
            travel_time = prev_node.travel_time_to(curr_node, self.speed)
            current_time += travel_time
            
            # 等待时间 (如果太早到达)
            if current_time < curr_node.ready_time:
                current_time = curr_node.ready_time
            
            # 计算违反量 (如果迟到)
            if current_time > curr_node.due_time:
                total_violation += current_time - curr_node.due_time
            
            # 加上服务时间
            current_time += curr_node.service_time
        
        self._time_violation = total_violation
        return total_violation
    
    def get_arrival_times(self) -> List[float]:
        """获取到达每个节点的时间列表"""
        if len(self.route) == 0:
            return []
        
        arrival_times = []
        current_time = 0.0
        full_route = self.full_route
        
        for i in range(1, len(full_route)):
            prev_node = full_route[i - 1]
            curr_node = full_route[i]
            
            travel_time = prev_node.travel_time_to(curr_node, self.speed)
            current_time += travel_time
            
            # 等待时间
            if current_time < curr_node.ready_time:
                current_time = curr_node.ready_time
            
            arrival_times.append(current_time)
            current_time += curr_node.service_time
        
        return arrival_times
    
    def get_load_at_each_node(self) -> List[float]:
        """获取到达每个节点时的载重量"""
        loads = []
        current_load = 0.0
        
        for node in self.route:
            current_load += node.demand
            loads.append(current_load)
        
        return loads
    
    def check_capacity_feasibility(self) -> bool:
        """检查容量约束是否满足"""
        current_load = 0.0
        
        for node in self.route:
            current_load += node.demand
            if current_load > self.capacity or current_load < 0:
                return False
        
        return True
    
    def check_precedence_feasibility(self) -> bool:
        """检查配对和顺序约束"""
        # 记录已访问的取货点
        visited_pickups = set()
        
        for node in self.route:
            if node.is_pickup():
                visited_pickups.add(node.id)
            elif node.is_delivery():
                # 检查对应的取货点是否已访问
                if node.pair_id not in visited_pickups:
                    return False
        
        return True
    
    def is_feasible(self) -> bool:
        """检查路径是否可行"""
        if self._is_valid is not None:
            return self._is_valid
        
        self._is_valid = (
            self.check_capacity_feasibility() and
            self.check_precedence_feasibility()
        )
        return self._is_valid
    
    def insert_order(self, pickup_node: 'Node', delivery_node: 'Node', 
                     pickup_pos: int, delivery_pos: int) -> bool:
        """
        在指定位置插入一个订单 (取货点和送货点)
        
        Args:
            pickup_node: 取货点
            delivery_node: 送货点
            pickup_pos: 取货点插入位置
            delivery_pos: 送货点插入位置 (在原始路径的基础上)
        
        Returns:
            是否插入成功
        """
        if pickup_pos > delivery_pos:
            return False
        
        # 插入取货点
        self.route.insert(pickup_pos, pickup_node)
        # 插入送货点 (注意: pickup已经插入, 所以位置+1)
        self.route.insert(delivery_pos + 1, delivery_node)
        
        self.invalidate_cache()
        return True
    
    def remove_order(self, pickup_node: 'Node', delivery_node: 'Node') -> bool:
        """
        移除一个订单 (取货点和送货点)
        
        Returns:
            是否移除成功
        """
        try:
            self.route.remove(pickup_node)
            self.route.remove(delivery_node)
            self.invalidate_cache()
            return True
        except ValueError:
            return False
    
    def get_order_ids(self) -> set:
        """获取路径中所有订单ID"""
        return {node.order_id for node in self.route if node.order_id is not None}
    
    def copy(self) -> 'Vehicle':
        """创建骑手的深拷贝"""
        new_vehicle = Vehicle(
            vehicle_id=self.id,
            capacity=self.capacity,
            speed=self.speed,
            depot=self.depot  # depot是共享的
        )
        new_vehicle.route = [node.copy() for node in self.route]
        return new_vehicle
    
    def __len__(self) -> int:
        """返回路径长度"""
        return len(self.route)
    
    def __repr__(self) -> str:
        route_str = " -> ".join([str(n) for n in self.full_route])
        return f"Vehicle{self.id}: {route_str}"
