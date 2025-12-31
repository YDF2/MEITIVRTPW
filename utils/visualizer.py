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
        title: str = "美团SOTA算法 - UCB算子统计",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制详细的算子统计信息（UCB、使用次数、平均奖励）
        优化版：更清晰、更美观、更直观
        
        Args:
            destroy_ops: DestroyOperators实例
            repair_ops: RepairOperators实例
            title: 标题
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(18, 11))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3, 
                              top=0.93, bottom=0.05, left=0.08, right=0.97)
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 配色方案
        COLOR_NEW = '#FF6B35'  # 新算子 - 活力橙
        COLOR_OLD = '#4ECDC4'  # 旧算子 - 青色
        COLOR_REPAIR = '#A8E6CF'  # 修复算子 - 薄荷绿
        
        # === 第一行：UCB参数展示（整行） ===
        ax_info = fig.add_subplot(gs[0, :])
        ax_info.axis('off')
        
        # 创建信息框
        info_text = f"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║  美团SOTA算法改进 - 基于UCB的自适应算子选择                                                      ║
╠═══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                           ║
║  【核心改进】                                                                               ║
║    ✓ UCB算子选择：启用={destroy_ops.use_ucb}  |  探索系数C={destroy_ops.ucb_c}  |  总迭代={destroy_ops.total_iterations}次          ║
║    ✓ 新增算子(h2/h7)：空间邻近移除 + 截止时间移除                                               ║
║    ✓ 风险决策：Matching Score = 0.7×Cost + 0.3×Risk                                       ║
║    ✓ 真实建模：共享取货点(≤1/3) + 5km配送限制                                                  ║
║                                                                                           ║
║  【UCB公式】Score = 平均奖励 + C × √(2×ln(N)/n)  ➜  智能平衡探索与利用                          ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
        """
        ax_info.text(0.5, 0.5, info_text, fontsize=10.5, family='monospace',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='#E8F4F8', 
                             edgecolor='#2E86AB', linewidth=2, alpha=0.9))
        
        # === 第二行：破坏算子对比图 ===
        destroy_names = [name for name, _ in destroy_ops.operators]
        destroy_counts = [destroy_ops.usage_counts.get(name, 0) for name in destroy_names]
        destroy_rewards = [destroy_ops.avg_rewards.get(name, 0) for name in destroy_names]
        
        # 简化算子名称以便显示
        destroy_names_short = []
        for name in destroy_names:
            if name == 'spatial_proximity_removal':
                destroy_names_short.append('🆕 h2-空间邻近')
            elif name == 'deadline_based_removal':
                destroy_names_short.append('🆕 h7-截止时间')
            elif name == 'random_removal':
                destroy_names_short.append('随机移除')
            elif name == 'worst_removal':
                destroy_names_short.append('最差移除')
            elif name == 'shaw_removal':
                destroy_names_short.append('Shaw移除')
            elif name == 'route_removal':
                destroy_names_short.append('路径移除')
            else:
                destroy_names_short.append(name[:10])
        
        # 颜色编码
        colors = [COLOR_NEW if 'h2' in n or 'h7' in n else COLOR_OLD 
                  for n in destroy_names_short]
        
        # 左图：破坏算子使用次数
        ax1 = fig.add_subplot(gs[1, 0])
        y_pos = np.arange(len(destroy_names_short))
        bars = ax1.barh(y_pos, destroy_counts, color=colors, edgecolor='black', linewidth=1.2)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(destroy_names_short, fontsize=10)
        ax1.set_xlabel('使用次数', fontsize=11, fontweight='bold')
        ax1.set_title('破坏算子 - 使用频率', fontsize=12, fontweight='bold', pad=10)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for i, (bar, count) in enumerate(zip(bars, destroy_counts)):
            if count > 0:
                ax1.text(count, i, f' {count}', va='center', fontsize=9, fontweight='bold')
        
        # 中图：破坏算子平均奖励
        ax2 = fig.add_subplot(gs[1, 1])
        bars = ax2.barh(y_pos, destroy_rewards, color=colors, edgecolor='black', linewidth=1.2)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(destroy_names_short, fontsize=10)
        ax2.set_xlabel('平均奖励(UCB)', fontsize=11, fontweight='bold')
        ax2.set_title('破坏算子 - 奖励评分', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for i, (bar, reward) in enumerate(zip(bars, destroy_rewards)):
            if reward > 0:
                ax2.text(reward, i, f' {reward:.2f}', va='center', fontsize=9, fontweight='bold')
        
        # 右图：UCB Score可视化
        ax3 = fig.add_subplot(gs[1, 2])
        ucb_scores = []
        for name, count, reward in zip(destroy_names, destroy_counts, destroy_rewards):
            if count > 0 and destroy_ops.total_iterations > 0:
                exploration = destroy_ops.ucb_c * np.sqrt(
                    2 * np.log(destroy_ops.total_iterations) / count
                )
                ucb_scores.append(reward + exploration)
            else:
                ucb_scores.append(reward)
        
        bars = ax3.barh(y_pos, ucb_scores, color=colors, edgecolor='black', linewidth=1.2)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(destroy_names_short, fontsize=10)
        ax3.set_xlabel('UCB总分', fontsize=11, fontweight='bold')
        ax3.set_title('破坏算子 - UCB选择评分', fontsize=12, fontweight='bold', pad=10)
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, (bar, score) in enumerate(zip(bars, ucb_scores)):
            if score > 0:
                ax3.text(score, i, f' {score:.2f}', va='center', fontsize=9, fontweight='bold')
        
        # === 第三行：修复算子统计 ===
        repair_names = [name for name, _ in repair_ops.operators]
        repair_counts = [repair_ops.usage_counts.get(name, 0) for name in repair_names]
        repair_rewards = [repair_ops.avg_rewards.get(name, 0) for name in repair_names]
        
        # 简化修复算子名称
        repair_names_short = []
        for name in repair_names:
            if name == 'greedy_insertion':
                repair_names_short.append('贪婪插入')
            elif name == 'regret_2_insertion':
                repair_names_short.append('Regret-2插入')
            elif name == 'regret_3_insertion':
                repair_names_short.append('Regret-3插入')
            elif name == 'random_insertion':
                repair_names_short.append('随机插入')
            else:
                repair_names_short.append(name[:10])
        
        colors_repair = [COLOR_REPAIR] * len(repair_names_short)
        y_pos_repair = np.arange(len(repair_names_short))
        
        # 左图：修复算子使用次数
        ax4 = fig.add_subplot(gs[2, 0])
        bars = ax4.barh(y_pos_repair, repair_counts, color=colors_repair, 
                       edgecolor='#2D6A4F', linewidth=1.2)
        ax4.set_yticks(y_pos_repair)
        ax4.set_yticklabels(repair_names_short, fontsize=10)
        ax4.set_xlabel('使用次数', fontsize=11, fontweight='bold')
        ax4.set_title('修复算子 - 使用频率', fontsize=12, fontweight='bold', pad=10)
        ax4.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, (bar, count) in enumerate(zip(bars, repair_counts)):
            if count > 0:
                ax4.text(count, i, f' {count}', va='center', fontsize=9, fontweight='bold')
        
        # 中图：修复算子平均奖励
        ax5 = fig.add_subplot(gs[2, 1])
        bars = ax5.barh(y_pos_repair, repair_rewards, color=colors_repair,
                       edgecolor='#2D6A4F', linewidth=1.2)
        ax5.set_yticks(y_pos_repair)
        ax5.set_yticklabels(repair_names_short, fontsize=10)
        ax5.set_xlabel('平均奖励(UCB)', fontsize=11, fontweight='bold')
        ax5.set_title('修复算子 - 奖励评分', fontsize=12, fontweight='bold', pad=10)
        ax5.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, (bar, reward) in enumerate(zip(bars, repair_rewards)):
            if reward > 0:
                ax5.text(reward, i, f' {reward:.2f}', va='center', fontsize=9, fontweight='bold')
        
        # 右图：修复算子UCB Score
        ax6 = fig.add_subplot(gs[2, 2])
        ucb_scores_repair = []
        for name, count, reward in zip(repair_names, repair_counts, repair_rewards):
            if count > 0 and repair_ops.total_iterations > 0:
                exploration = repair_ops.ucb_c * np.sqrt(
                    2 * np.log(repair_ops.total_iterations) / count
                )
                ucb_scores_repair.append(reward + exploration)
            else:
                ucb_scores_repair.append(reward)
        
        bars = ax6.barh(y_pos_repair, ucb_scores_repair, color=colors_repair,
                       edgecolor='#2D6A4F', linewidth=1.2)
        ax6.set_yticks(y_pos_repair)
        ax6.set_yticklabels(repair_names_short, fontsize=10)
        ax6.set_xlabel('UCB总分', fontsize=11, fontweight='bold')
        ax6.set_title('修复算子 - UCB选择评分', fontsize=12, fontweight='bold', pad=10)
        ax6.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, (bar, score) in enumerate(zip(bars, ucb_scores_repair)):
            if score > 0:
                ax6.text(score, i, f' {score:.2f}', va='center', fontsize=9, fontweight='bold')
        
        # === 第四行：算子性能对比雷达图 ===
        ax7 = fig.add_subplot(gs[3, :], projection='polar')
        
        # 准备雷达图数据
        categories = destroy_names_short
        values_count = [c / max(destroy_counts) if max(destroy_counts) > 0 else 0 
                       for c in destroy_counts]
        values_reward = [r / max(destroy_rewards) if max(destroy_rewards) > 0 else 0 
                        for r in destroy_rewards]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values_count += values_count[:1]
        values_reward += values_reward[:1]
        angles += angles[:1]
        
        ax7.plot(angles, values_count, 'o-', linewidth=2, label='使用频率(归一化)', color='#FF6B35')
        ax7.fill(angles, values_count, alpha=0.25, color='#FF6B35')
        ax7.plot(angles, values_reward, 's-', linewidth=2, label='平均奖励(归一化)', color='#4ECDC4')
        ax7.fill(angles, values_reward, alpha=0.25, color='#4ECDC4')
        
        ax7.set_xticks(angles[:-1])
        ax7.set_xticklabels(categories, fontsize=9)
        ax7.set_ylim(0, 1)
        ax7.set_title('破坏算子性能雷达图', fontsize=12, fontweight='bold', pad=20)
        ax7.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)
        ax7.grid(True, alpha=0.3)
        
        # 总标题
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✓ 算子统计图已保存至: {save_path}")
        
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
