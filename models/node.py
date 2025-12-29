# -*- coding: utf-8 -*-
"""
节点模型 (Node)
表示配送网络中的各类节点：配送站(起点)、商家(取货点)、顾客(送货点)
"""

from enum import Enum
from typing import Optional


class NodeType(Enum):
    """节点类型枚举"""
    DEPOT = 0       # 配送站/起点
    PICKUP = 1      # 取货点 (商家)
    DELIVERY = 2    # 送货点 (顾客)


class Node:
    """
    节点类
    
    Attributes:
        id: 节点唯一标识
        x: x坐标
        y: y坐标
        node_type: 节点类型 (DEPOT/PICKUP/DELIVERY)
        demand: 需求量 (取货为正, 送货为负, 配送站为0)
        ready_time: 时间窗开始时间 (最早服务时间)
        due_time: 时间窗结束时间 (最晚服务时间)
        service_time: 服务时间 (取餐/送餐耗时)
        pair_id: 配对节点ID (取货点对应的送货点ID, 或送货点对应的取货点ID)
        order_id: 所属订单ID
    """
    
    def __init__(
        self,
        node_id: int,
        x: float,
        y: float,
        node_type: NodeType,
        demand: float = 0,
        ready_time: float = 0,
        due_time: float = float('inf'),
        service_time: float = 0,
        pair_id: Optional[int] = None,
        order_id: Optional[int] = None
    ):
        self.id = node_id
        self.x = x
        self.y = y
        self.node_type = node_type
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time
        self.pair_id = pair_id  # 配对点ID
        self.order_id = order_id  # 所属订单ID
    
    def distance_to(self, other: 'Node') -> float:
        """计算到另一个节点的欧氏距离"""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
    
    def travel_time_to(self, other: 'Node', speed: float = 1.0) -> float:
        """计算到另一个节点的行驶时间"""
        return self.distance_to(other) / speed
    
    def is_pickup(self) -> bool:
        """是否为取货点"""
        return self.node_type == NodeType.PICKUP
    
    def is_delivery(self) -> bool:
        """是否为送货点"""
        return self.node_type == NodeType.DELIVERY
    
    def is_depot(self) -> bool:
        """是否为配送站"""
        return self.node_type == NodeType.DEPOT
    
    def copy(self) -> 'Node':
        """创建节点的深拷贝"""
        return Node(
            node_id=self.id,
            x=self.x,
            y=self.y,
            node_type=self.node_type,
            demand=self.demand,
            ready_time=self.ready_time,
            due_time=self.due_time,
            service_time=self.service_time,
            pair_id=self.pair_id,
            order_id=self.order_id
        )
    
    def __repr__(self) -> str:
        type_str = {
            NodeType.DEPOT: "Depot",
            NodeType.PICKUP: "P",
            NodeType.DELIVERY: "D"
        }
        return f"{type_str[self.node_type]}{self.id}"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Node):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        return hash(self.id)


class Order:
    """
    订单类
    每个订单包含一对取货点和送货点
    
    Attributes:
        id: 订单ID
        pickup_node: 取货点
        delivery_node: 送货点
        demand: 货物数量
    """
    
    def __init__(
        self,
        order_id: int,
        pickup_node: Node,
        delivery_node: Node,
        demand: float = 1
    ):
        self.id = order_id
        self.pickup_node = pickup_node
        self.delivery_node = delivery_node
        self.demand = demand
        
        # 确保节点关联正确
        self.pickup_node.order_id = order_id
        self.delivery_node.order_id = order_id
        self.pickup_node.pair_id = delivery_node.id
        self.delivery_node.pair_id = pickup_node.id
        self.pickup_node.demand = demand
        self.delivery_node.demand = -demand
    
    def __repr__(self) -> str:
        return f"Order{self.id}({self.pickup_node} -> {self.delivery_node})"
