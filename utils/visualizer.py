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
        
        # 1. 绘制配送站（支持多站点）
        # 优先使用depots列表，如果存在且不为空
        depots = solution.depots if hasattr(solution, 'depots') and solution.depots else [solution.depot]
        
        # 使用不同颜色显示不同站点
        depot_colors = ['black', 'darkred', 'darkgreen', 'darkblue', 'purple']
        for depot_idx, depot in enumerate(depots):
            color = depot_colors[depot_idx % len(depot_colors)]
            marker_size = 300 if len(depots) == 1 else 200
            
            ax.scatter(
                depot.x, depot.y, 
                marker='s', s=marker_size, c=color, 
                edgecolors='white', linewidths=2,
                zorder=10
            )
            if show_labels:
                label_text = 'Depot' if len(depots) == 1 else f'D{depot_idx}'
                ax.annotate(label_text, (depot.x, depot.y), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold', color='white',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
        
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
        legend_elements = []
        
        # 添加配送站图例（支持多站点）
        # 使用原始solution.depots判断，确保显示所有站点
        actual_depots = solution.depots if hasattr(solution, 'depots') and solution.depots else [solution.depot]
        if len(actual_depots) == 1:
            legend_elements.append(
                Line2D([0], [0], marker='s', color='w', markerfacecolor='black', 
                       markersize=12, label='配送站')
            )
        else:
            depot_colors = ['black', 'darkred', 'darkgreen', 'darkblue', 'purple']
            for depot_idx, depot in enumerate(actual_depots):
                color = depot_colors[depot_idx % len(depot_colors)]
                legend_elements.append(
                    Line2D([0], [0], marker='s', color='w', markerfacecolor=color,
                           markeredgecolor='white', markeredgewidth=1.5,
                           markersize=10, label=f'站点{depot_idx} ({depot.x:.0f},{depot.y:.0f})')
                )
        
        legend_elements.extend([
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                   markersize=10, label='取货点 (商家)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, label='送货点 (顾客)'),
        ])
        
        # 为所有骑手添加图例
        for idx, vehicle in enumerate(solution.vehicles):
            color = self.COLORS[idx % len(self.COLORS)]
            route_status = '' if len(vehicle.route) > 0 else ' (空闲)'
            legend_elements.append(
                Line2D([0], [0], color=color, lw=2, 
                       label=f'骑手 {vehicle.id}{route_status}')
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
        # 计算总骑手数（所有车辆）
        total_vehicles = len(solution.vehicles)
        info_text = (
            f"总成本: {stats['total_cost']:.2f}\n"
            f"总距离: {stats['total_distance']:.2f}\n"
            f"时间违反: {stats['total_time_violation']:.2f}\n"
            f"使用骑手: {stats['num_vehicles_used']}/{total_vehicles}\n"
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
        
        # 破坏算子 - 标注新增的美团SOTA算子
        names = list(destroy_weights.keys())
        weights = list(destroy_weights.values())
        colors = []
        for name in names:
            if name in ['spatial_proximity_removal', 'deadline_based_removal']:
                colors.append('#FF4500')  # 新算子用亮橙色
            else:
                colors.append('#DC143C')  # 旧算子用深红色
        
        bars1 = ax1.bar(names, weights, color=colors)
        ax1.set_title('破坏算子权重 (🆕=美团SOTA)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('权重')
        ax1.tick_params(axis='x', rotation=45)
        
        # 为新算子添加标记
        for i, (name, weight) in enumerate(zip(names, weights)):
            if name in ['spatial_proximity_removal', 'deadline_based_removal']:
                ax1.text(i, weight, '🆕', ha='center', va='bottom', fontsize=16)
        
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
    
    def plot_operator_statistics(
        self,
        destroy_ops,
        repair_ops,
        title: str = "ALNS Operator Statistics (UCB-based)",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制详细的算子统计信息（UCB、使用次数、平均奖励）
        优化版：修复中文显示问题，使用英文+中文混合标签
        
        Args:
            destroy_ops: DestroyOperators实例
            repair_ops: RepairOperators实例
            title: 标题
            save_path: 保存路径
        """
        # 设置中文字体 - 优先使用系统中文字体
        import matplotlib.font_manager as fm
        
        # 尝试多种中文字体
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
        font_found = None
        for font_name in chinese_fonts:
            try:
                font_path = fm.findfont(fm.FontProperties(family=font_name))
                if font_path and 'ttf' in font_path.lower():
                    font_found = font_name
                    break
            except:
                continue
        
        if font_found:
            plt.rcParams['font.sans-serif'] = [font_found, 'DejaVu Sans']
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        fig = plt.figure(figsize=(16, 12))
        
        # 配色方案
        COLOR_NEW = '#FF6B35'   # 新算子 - 活力橙
        COLOR_OLD = '#4ECDC4'   # 旧算子 - 青色
        COLOR_REPAIR = '#95D5B2'  # 修复算子 - 薄荷绿
        
        # === 获取数据 ===
        destroy_names = [name for name, _ in destroy_ops.operators]
        destroy_counts = [destroy_ops.usage_counts.get(name, 0) for name in destroy_names]
        destroy_rewards = [destroy_ops.avg_rewards.get(name, 0) for name in destroy_names]
        
        repair_names = [name for name, _ in repair_ops.operators]
        repair_counts = [repair_ops.usage_counts.get(name, 0) for name in repair_names]
        repair_rewards = [repair_ops.avg_rewards.get(name, 0) for name in repair_names]
        
        # 算子名称映射（使用英文为主，避免字体问题）
        destroy_labels = []
        for name in destroy_names:
            if name == 'spatial_proximity_removal':
                destroy_labels.append('[NEW] h2-Spatial')
            elif name == 'deadline_based_removal':
                destroy_labels.append('[NEW] h7-Deadline')
            elif name == 'random_removal':
                destroy_labels.append('Random')
            elif name == 'worst_removal':
                destroy_labels.append('Worst')
            elif name == 'shaw_removal':
                destroy_labels.append('Shaw')
            elif name == 'route_removal':
                destroy_labels.append('Route')
            else:
                destroy_labels.append(name[:12])
        
        repair_labels = []
        for name in repair_names:
            if name == 'greedy_insertion':
                repair_labels.append('Greedy')
            elif name == 'regret_2_insertion':
                repair_labels.append('Regret-2')
            elif name == 'regret_3_insertion':
                repair_labels.append('Regret-3')
            elif name == 'random_insertion':
                repair_labels.append('Random')
            else:
                repair_labels.append(name[:12])
        
        # 标记新算子颜色
        destroy_colors = [COLOR_NEW if '[NEW]' in label else COLOR_OLD for label in destroy_labels]
        
        # 计算UCB分数
        def calc_ucb_scores(names, counts, rewards, ops):
            scores = []
            for name, count, reward in zip(names, counts, rewards):
                if count > 0 and ops.total_iterations > 0:
                    exploration = ops.ucb_c * np.sqrt(2 * np.log(ops.total_iterations) / count)
                    scores.append(reward + exploration)
                else:
                    scores.append(reward)
            return scores
        
        destroy_ucb = calc_ucb_scores(destroy_names, destroy_counts, destroy_rewards, destroy_ops)
        repair_ucb = calc_ucb_scores(repair_names, repair_counts, repair_rewards, repair_ops)
        
        # === 布局: 3行2列 ===
        # 第1行: 信息面板
        # 第2行: 破坏算子 (使用次数 | UCB评分)
        # 第3行: 修复算子 (使用次数 | UCB评分)
        
        gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 1.5, 1.2], 
                              hspace=0.35, wspace=0.25,
                              top=0.92, bottom=0.08, left=0.10, right=0.95)
        
        # === 第1行: 信息面板 ===
        ax_info = fig.add_subplot(gs[0, :])
        ax_info.axis('off')
        
        info_lines = [
            f"UCB Selection: {'Enabled' if destroy_ops.use_ucb else 'Disabled'}",
            f"Exploration Coefficient C = {destroy_ops.ucb_c}",
            f"Total Iterations = {destroy_ops.total_iterations}",
            f"New Operators: h2-Spatial Proximity, h7-Deadline Based",
            f"UCB Formula: Score = Avg_Reward + C * sqrt(2*ln(N)/n)"
        ]
        
        info_text = "  |  ".join(info_lines[:3]) + "\n" + "  |  ".join(info_lines[3:])
        
        ax_info.text(0.5, 0.5, info_text, fontsize=11, 
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='#E8F4F8', 
                             edgecolor='#2E86AB', linewidth=2, alpha=0.95),
                    family='DejaVu Sans')
        
        # === 第2行: 破坏算子 ===
        y_pos_d = np.arange(len(destroy_labels))
        
        # 左: 使用次数
        ax1 = fig.add_subplot(gs[1, 0])
        bars1 = ax1.barh(y_pos_d, destroy_counts, color=destroy_colors, 
                        edgecolor='#333333', linewidth=0.8, height=0.7)
        ax1.set_yticks(y_pos_d)
        ax1.set_yticklabels(destroy_labels, fontsize=11)
        ax1.set_xlabel('Usage Count', fontsize=11, fontweight='bold')
        ax1.set_title('Destroy Operators - Usage Frequency', fontsize=13, fontweight='bold', pad=12)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        ax1.set_xlim(0, max(destroy_counts) * 1.15 if max(destroy_counts) > 0 else 1)
        
        for i, count in enumerate(destroy_counts):
            if count > 0:
                ax1.text(count + max(destroy_counts)*0.02, i, str(count), 
                        va='center', fontsize=10, fontweight='bold')
        
        # 右: UCB评分
        ax2 = fig.add_subplot(gs[1, 1])
        bars2 = ax2.barh(y_pos_d, destroy_ucb, color=destroy_colors,
                        edgecolor='#333333', linewidth=0.8, height=0.7)
        ax2.set_yticks(y_pos_d)
        ax2.set_yticklabels(destroy_labels, fontsize=11)
        ax2.set_xlabel('UCB Score', fontsize=11, fontweight='bold')
        ax2.set_title('Destroy Operators - UCB Selection Score', fontsize=13, fontweight='bold', pad=12)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        ax2.set_xlim(0, max(destroy_ucb) * 1.15 if max(destroy_ucb) > 0 else 1)
        
        for i, score in enumerate(destroy_ucb):
            if score > 0:
                ax2.text(score + max(destroy_ucb)*0.02, i, f'{score:.2f}', 
                        va='center', fontsize=10, fontweight='bold')
        
        # === 第3行: 修复算子 ===
        y_pos_r = np.arange(len(repair_labels))
        repair_colors = [COLOR_REPAIR] * len(repair_labels)
        
        # 左: 使用次数
        ax3 = fig.add_subplot(gs[2, 0])
        bars3 = ax3.barh(y_pos_r, repair_counts, color=repair_colors,
                        edgecolor='#2D6A4F', linewidth=0.8, height=0.6)
        ax3.set_yticks(y_pos_r)
        ax3.set_yticklabels(repair_labels, fontsize=11)
        ax3.set_xlabel('Usage Count', fontsize=11, fontweight='bold')
        ax3.set_title('Repair Operators - Usage Frequency', fontsize=13, fontweight='bold', pad=12)
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
        ax3.set_xlim(0, max(repair_counts) * 1.15 if max(repair_counts) > 0 else 1)
        
        for i, count in enumerate(repair_counts):
            if count > 0:
                ax3.text(count + max(repair_counts)*0.02, i, str(count),
                        va='center', fontsize=10, fontweight='bold')
        
        # 右: UCB评分
        ax4 = fig.add_subplot(gs[2, 1])
        bars4 = ax4.barh(y_pos_r, repair_ucb, color=repair_colors,
                        edgecolor='#2D6A4F', linewidth=0.8, height=0.6)
        ax4.set_yticks(y_pos_r)
        ax4.set_yticklabels(repair_labels, fontsize=11)
        ax4.set_xlabel('UCB Score', fontsize=11, fontweight='bold')
        ax4.set_title('Repair Operators - UCB Selection Score', fontsize=13, fontweight='bold', pad=12)
        ax4.grid(axis='x', alpha=0.3, linestyle='--')
        ax4.set_xlim(0, max(repair_ucb) * 1.15 if max(repair_ucb) > 0 else 1)
        
        for i, score in enumerate(repair_ucb):
            if score > 0:
                ax4.text(score + max(repair_ucb)*0.02, i, f'{score:.2f}',
                        va='center', fontsize=10, fontweight='bold')
        
        # 添加图例说明
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLOR_NEW, edgecolor='#333', label='New Operators (h2/h7)'),
            Patch(facecolor=COLOR_OLD, edgecolor='#333', label='Traditional Operators'),
            Patch(facecolor=COLOR_REPAIR, edgecolor='#2D6A4F', label='Repair Operators')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
                   fontsize=10, bbox_to_anchor=(0.5, 0.01))
        
        # 总标题
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Operator statistics saved to: {save_path}")
        
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
