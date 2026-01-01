# -*- coding: utf-8 -*-
"""
解模型 (Solution)
封装整个调度系统的状态，便于算法迭代中复制和回滚
"""

from typing import List, Dict, Set, Optional, TYPE_CHECKING
import copy as copy_module
import numpy as np

if TYPE_CHECKING:
    from .node import Node, Order
    from .vehicle import Vehicle


class Solution:
    """
    解类
    封装完整的调度方案
    
    Attributes:
        vehicles: 所有骑手列表
        unassigned_orders: 未分配的订单列表
        nodes: 所有节点字典 {node_id: Node}
        orders: 所有订单字典 {order_id: Order}
        depot: 配送站节点
    """
    
    def __init__(
        self,
        vehicles: List['Vehicle'],
        orders: List['Order'],
        depot: 'Node'
    ):
        self.vehicles = vehicles
        self.orders = {order.id: order for order in orders}
        self.depot = depot
        self.unassigned_orders: List['Order'] = list(orders)  # 初始时所有订单未分配
        
        # 多站点支持：存储所有站点
        self.depots: List['Node'] = [depot]  # 默认单站点，可以通过外部设置为多站点
        
        # 构建节点字典
        self.nodes: Dict[int, 'Node'] = {depot.id: depot}
        for order in orders:
            self.nodes[order.pickup_node.id] = order.pickup_node
            self.nodes[order.delivery_node.id] = order.delivery_node
        
        # 缓存
        self._total_cost: Optional[float] = None
        
        # ========== 静态近邻缓存（优化核心） ==========
        self._distance_matrix: Optional[np.ndarray] = None
        self._node_id_to_index: Optional[Dict[int, int]] = None
        self._nearest_neighbors: Optional[Dict[int, List[int]]] = None
        self._build_distance_cache()
    
    def _build_distance_cache(self, k_neighbors: int = 10):
        """
        构建距离矩阵和最近邻居缓存
        
        Args:
            k_neighbors: 为每个节点保存的最近邻居数量
        """
        node_list = list(self.nodes.values())
        n = len(node_list)
        
        # 构建节点ID到索引的映射
        self._node_id_to_index = {node.id: i for i, node in enumerate(node_list)}
        
        # 构建距离矩阵
        self._distance_matrix = np.zeros((n, n))
        for i, node1 in enumerate(node_list):
            for j, node2 in enumerate(node_list):
                if i != j:
                    self._distance_matrix[i, j] = node1.distance_to(node2)
        
        # 为每个节点计算最近的k个邻居
        self._nearest_neighbors = {}
        for i, node in enumerate(node_list):
            # 获取距离并排序
            distances = self._distance_matrix[i].copy()
            distances[i] = float('inf')  # 排除自己
            
            # 获取最近的k个邻居的索引
            k = min(k_neighbors, n - 1)
            nearest_indices = np.argpartition(distances, k)[:k]
            nearest_indices = nearest_indices[np.argsort(distances[nearest_indices])]
            
            # 转换为节点ID
            nearest_node_ids = [node_list[idx].id for idx in nearest_indices]
            self._nearest_neighbors[node.id] = nearest_node_ids
    
    def get_distance(self, node1_id: int, node2_id: int) -> float:
        """
        快速获取两个节点之间的距离（使用缓存）
        
        Args:
            node1_id: 第一个节点ID
            node2_id: 第二个节点ID
        
        Returns:
            曼哈顿距离
        """
        if self._distance_matrix is None or self._node_id_to_index is None:
            # 缓存未构建，直接计算
            return self.nodes[node1_id].distance_to(self.nodes[node2_id])
        
        i = self._node_id_to_index[node1_id]
        j = self._node_id_to_index[node2_id]
        return self._distance_matrix[i, j]
    
    def get_nearest_neighbors(self, node_id: int, k: int = None) -> List[int]:
        """
        获取节点的最近邻居节点ID列表
        
        Args:
            node_id: 节点ID
            k: 返回的邻居数量（如果为None则返回所有缓存的邻居）
        
        Returns:
            最近邻居节点ID列表（按距离排序）
        """
        if self._nearest_neighbors is None:
            return []
        
        neighbors = self._nearest_neighbors.get(node_id, [])
        if k is not None:
            neighbors = neighbors[:k]
        
        return neighbors
    
    def invalidate_cache(self):
        """使缓存失效"""
        self._total_cost = None
        for vehicle in self.vehicles:
            vehicle.invalidate_cache()
    
    @property
    def total_distance(self) -> float:
        """总行驶距离"""
        return sum(v.calculate_distance() for v in self.vehicles)
    
    @property
    def total_time_violation(self) -> float:
        """总时间窗违反量"""
        return sum(v.calculate_time_violation() for v in self.vehicles)
    
    @property
    def num_used_vehicles(self) -> int:
        """使用的骑手数量"""
        return sum(1 for v in self.vehicles if len(v.route) > 0)
    
    @property
    def num_unassigned(self) -> int:
        """未分配订单数量"""
        return len(self.unassigned_orders)
    
    def calculate_cost(
        self,
        w_distance: float = 1.0,
        w_time_penalty: float = 100.0,
        w_unassigned: float = 1000.0,
        w_vehicle: float = 50.0
    ) -> float:
        """
        计算解的总成本
        
        Cost = w1 * Distance + w2 * TimePenalty + w3 * UnassignedPenalty + w4 * VehicleUsage
        """
        if self._total_cost is not None:
            return self._total_cost
        
        distance_cost = w_distance * self.total_distance
        time_penalty = w_time_penalty * self.total_time_violation
        unassigned_penalty = w_unassigned * self.num_unassigned
        vehicle_cost = w_vehicle * self.num_used_vehicles
        
        self._total_cost = distance_cost + time_penalty + unassigned_penalty + vehicle_cost
        return self._total_cost
    
    def is_feasible(self) -> bool:
        """检查解是否可行 (硬约束)"""
        # 检查所有骑手路径
        for vehicle in self.vehicles:
            if not vehicle.is_feasible():
                return False
        
        # 检查配对约束: 同一订单的取送点必须在同一辆车上
        for order in self.orders.values():
            if order in self.unassigned_orders:
                continue
            
            pickup_vehicle = None
            delivery_vehicle = None
            
            for vehicle in self.vehicles:
                for node in vehicle.route:
                    if node.id == order.pickup_node.id:
                        pickup_vehicle = vehicle.id
                    if node.id == order.delivery_node.id:
                        delivery_vehicle = vehicle.id
            
            if pickup_vehicle != delivery_vehicle:
                return False
        
        return True
    
    def get_assigned_orders(self) -> Set[int]:
        """获取所有已分配订单的ID"""
        assigned = set()
        for vehicle in self.vehicles:
            assigned.update(vehicle.get_order_ids())
        return assigned
    
    def assign_order(self, order: 'Order', vehicle: 'Vehicle', 
                     pickup_pos: int, delivery_pos: int) -> bool:
        """
        将订单分配给骑手
        
        Args:
            order: 订单对象
            vehicle: 目标骑手
            pickup_pos: 取货点插入位置
            delivery_pos: 送货点插入位置
        
        Returns:
            是否分配成功
        """
        # 插入节点
        success = vehicle.insert_order(
            order.pickup_node, 
            order.delivery_node,
            pickup_pos, 
            delivery_pos
        )
        
        if success:
            # 从未分配列表中移除
            if order in self.unassigned_orders:
                self.unassigned_orders.remove(order)
            self.invalidate_cache()
        
        return success
    
    def unassign_order(self, order: 'Order') -> bool:
        """
        取消订单分配
        
        Returns:
            是否成功
        """
        # 找到并移除订单
        for vehicle in self.vehicles:
            found = False
            for node in vehicle.route:
                if node.order_id == order.id:
                    found = True
                    break
            
            if found:
                vehicle.remove_order(order.pickup_node, order.delivery_node)
                if order not in self.unassigned_orders:
                    self.unassigned_orders.append(order)
                self.invalidate_cache()
                return True
        
        return False
    
    def get_order_by_id(self, order_id: int) -> Optional['Order']:
        """根据ID获取订单"""
        return self.orders.get(order_id)
    
    def get_node_by_id(self, node_id: int) -> Optional['Node']:
        """根据ID获取节点"""
        return self.nodes.get(node_id)
    
    def find_order_vehicle(self, order: 'Order') -> Optional['Vehicle']:
        """找到订单所在的骑手"""
        for vehicle in self.vehicles:
            if order.id in vehicle.get_order_ids():
                return vehicle
        return None
    
    def copy(self) -> 'Solution':
        """
        创建解的深拷贝
        这是ALNS算法中最关键的操作之一
        """
        # 深拷贝骑手 (包含路径)
        new_vehicles = [v.copy() for v in self.vehicles]
        
        # 创建新的订单列表引用 (订单对象本身可以共享)
        from .node import Order
        new_orders = list(self.orders.values())
        
        new_solution = Solution.__new__(Solution)
        new_solution.vehicles = new_vehicles
        new_solution.orders = self.orders.copy()
        new_solution.depot = self.depot
        new_solution.nodes = self.nodes.copy()
        new_solution.unassigned_orders = list(self.unassigned_orders)
        new_solution._total_cost = None
        
        # 【修复】复制多站点信息
        if hasattr(self, 'depots'):
            new_solution.depots = self.depots  # depots列表可以共享（节点对象不可变）
        else:
            new_solution.depots = [self.depot]  # 向后兼容
        
        return new_solution
    
    def get_statistics(self) -> Dict:
        """获取解的统计信息"""
        return {
            'total_cost': self.calculate_cost(),
            'total_distance': self.total_distance,
            'total_time_violation': self.total_time_violation,
            'num_vehicles_used': self.num_used_vehicles,
            'num_unassigned': self.num_unassigned,
            'is_feasible': self.is_feasible()
        }
    
    def __repr__(self) -> str:
        lines = [f"Solution (Cost: {self.calculate_cost():.2f})"]
        lines.append(f"  Used vehicles: {self.num_used_vehicles}/{len(self.vehicles)}")
        lines.append(f"  Unassigned orders: {self.num_unassigned}")
        lines.append(f"  Total distance: {self.total_distance:.2f}")
        lines.append(f"  Time violation: {self.total_time_violation:.2f}")
        lines.append("  Routes:")
        for v in self.vehicles:
            if len(v.route) > 0:
                lines.append(f"    {v}")
        return "\n".join(lines)
