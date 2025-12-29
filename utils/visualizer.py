# -*- coding: utf-8 -*-
"""
路径可视化 (Solution Visualizer)
使用matplotlib绘制配送路径图
"""

from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.solution import Solution
from models.vehicle import Vehicle
from models.node import Node, NodeType
import config


class SolutionVisualizer:
    """
    解的可视化器
    
    绘制配送网络和骑手路径
    """
    
    # 预定义颜色列表 (用于不同骑手)
    COLORS = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
        '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000'
    ]
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        self.figsize = figsize
    
    def plot(
        self,
        solution: Solution,
        title: str = "外卖配送路径规划",
        show_labels: bool = True,
        show_time_windows: bool = False,
        highlight_violations: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制解的可视化图
        
        Args:
            solution: 解对象
            title: 图表标题
            show_labels: 是否显示节点标签
            show_time_windows: 是否显示时间窗信息
            highlight_violations: 是否高亮显示违反约束的点
            save_path: 保存路径
        
        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 绘制配送站
        depot = solution.depot
        ax.scatter(
            depot.x, depot.y, 
            marker='s', s=300, c='black', 
            zorder=10, label='配送站'
        )
        if show_labels:
            ax.annotate('Depot', (depot.x, depot.y), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        # 2. 绘制所有节点
        pickup_nodes = []
        delivery_nodes = []
        
        for order in solution.orders.values():
            pickup_nodes.append(order.pickup_node)
            delivery_nodes.append(order.delivery_node)
        
        # 取货点 (红色三角形)
        pickup_x = [n.x for n in pickup_nodes]
        pickup_y = [n.y for n in pickup_nodes]
        ax.scatter(
            pickup_x, pickup_y, 
            marker='^', s=100, c='red', 
            alpha=0.7, zorder=5, label='取货点 (商家)'
        )
        
        # 送货点 (蓝色圆形)
        delivery_x = [n.x for n in delivery_nodes]
        delivery_y = [n.y for n in delivery_nodes]
        ax.scatter(
            delivery_x, delivery_y, 
            marker='o', s=100, c='blue', 
            alpha=0.7, zorder=5, label='送货点 (顾客)'
        )
        
        # 绘制节点标签
        if show_labels:
            for node in pickup_nodes:
                ax.annotate(
                    f'P{node.order_id}', (node.x, node.y),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=8, color='darkred'
                )
            for node in delivery_nodes:
                ax.annotate(
                    f'D{node.order_id}', (node.x, node.y),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=8, color='darkblue'
                )
        
        # 3. 绘制骑手路径
        for idx, vehicle in enumerate(solution.vehicles):
            if len(vehicle.route) == 0:
                continue
            
            color = self.COLORS[idx % len(self.COLORS)]
            full_route = vehicle.full_route
            
            # 绘制路径线
            for i in range(len(full_route) - 1):
                start = full_route[i]
                end = full_route[i + 1]
                
                # 绘制箭头
                ax.annotate(
                    '', xy=(end.x, end.y), xytext=(start.x, start.y),
                    arrowprops=dict(
                        arrowstyle='->', color=color,
                        lw=2, alpha=0.7,
                        connectionstyle='arc3,rad=0.05'
                    ),
                    zorder=3
                )
        
        # 4. 高亮显示未分配订单
        unassigned_orders = solution.unassigned_orders
        if len(unassigned_orders) > 0:
            unassigned_pickup_x = [o.pickup_node.x for o in unassigned_orders]
            unassigned_pickup_y = [o.pickup_node.y for o in unassigned_orders]
            ax.scatter(
                unassigned_pickup_x, unassigned_pickup_y,
                marker='^', s=150, facecolors='none', edgecolors='gray',
                linewidths=2, zorder=6
            )
            
            unassigned_delivery_x = [o.delivery_node.x for o in unassigned_orders]
            unassigned_delivery_y = [o.delivery_node.y for o in unassigned_orders]
            ax.scatter(
                unassigned_delivery_x, unassigned_delivery_y,
                marker='o', s=150, facecolors='none', edgecolors='gray',
                linewidths=2, zorder=6
            )
        
        # 5. 添加图例
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='black', 
                   markersize=12, label='配送站'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                   markersize=10, label='取货点 (商家)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, label='送货点 (顾客)'),
        ]
        
        # 为每个使用中的骑手添加图例
        for idx, vehicle in enumerate(solution.vehicles):
            if len(vehicle.route) > 0:
                color = self.COLORS[idx % len(self.COLORS)]
                legend_elements.append(
                    Line2D([0], [0], color=color, lw=2, 
                           label=f'骑手 {vehicle.id}')
                )
        
        if len(unassigned_orders) > 0:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor='none', markeredgecolor='gray',
                       markersize=10, label=f'未分配 ({len(unassigned_orders)})')
            )
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # 6. 设置标题和标签
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X 坐标', fontsize=11)
        ax.set_ylabel('Y 坐标', fontsize=11)
        
        # 添加统计信息
        stats = solution.get_statistics()
        info_text = (
            f"总成本: {stats['total_cost']:.2f}\n"
            f"总距离: {stats['total_distance']:.2f}\n"
            f"时间违反: {stats['total_time_violation']:.2f}\n"
            f"使用骑手: {stats['num_vehicles_used']}/{len(solution.vehicles)}\n"
            f"未分配订单: {stats['num_unassigned']}"
        )
        
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        # 7. 设置坐标轴范围
        ax.set_xlim(-5, config.GRID_SIZE + 5)
        ax.set_ylim(-5, config.GRID_SIZE + 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图片已保存至: {save_path}")
        
        return fig
    
    def plot_convergence(
        self,
        best_costs: List[float],
        current_costs: Optional[List[float]] = None,
        title: str = "ALNS 收敛曲线",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制算法收敛曲线
        
        Args:
            best_costs: 每次迭代的最优成本
            current_costs: 每次迭代的当前成本
            title: 标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        iterations = range(1, len(best_costs) + 1)
        
        ax.plot(iterations, best_costs, 'b-', linewidth=2, label='最优成本')
        
        if current_costs:
            ax.plot(iterations, current_costs, 'g-', alpha=0.5, 
                   linewidth=1, label='当前成本')
        
        ax.set_xlabel('迭代次数', fontsize=11)
        ax.set_ylabel('成本', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 标注起始和最终值
        ax.annotate(
            f'初始: {best_costs[0]:.2f}',
            xy=(1, best_costs[0]),
            xytext=(len(best_costs) * 0.1, best_costs[0]),
            fontsize=10
        )
        ax.annotate(
            f'最终: {best_costs[-1]:.2f}',
            xy=(len(best_costs), best_costs[-1]),
            xytext=(len(best_costs) * 0.8, best_costs[-1] * 1.1),
            fontsize=10
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"收敛曲线已保存至: {save_path}")
        
        return fig
    
    def plot_operator_weights(
        self,
        destroy_weights: Dict[str, float],
        repair_weights: Dict[str, float],
        title: str = "算子权重分布",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制算子权重分布
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 破坏算子
        names = list(destroy_weights.keys())
        weights = list(destroy_weights.values())
        colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(names)))
        
        ax1.bar(names, weights, color=colors)
        ax1.set_title('破坏算子权重', fontsize=12, fontweight='bold')
        ax1.set_ylabel('权重')
        ax1.tick_params(axis='x', rotation=45)
        
        # 修复算子
        names = list(repair_weights.keys())
        weights = list(repair_weights.values())
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(names)))
        
        ax2.bar(names, weights, color=colors)
        ax2.set_title('修复算子权重', fontsize=12, fontweight='bold')
        ax2.set_ylabel('权重')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def plot_solution(
    solution: Solution,
    title: str = "外卖配送路径规划",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    便捷函数: 绘制解的可视化图
    """
    visualizer = SolutionVisualizer()
    fig = visualizer.plot(solution, title=title, save_path=save_path)
    
    if show:
        plt.show()
    
    return fig
