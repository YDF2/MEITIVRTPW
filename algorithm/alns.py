# -*- coding: utf-8 -*-
"""
ALNS (Adaptive Large Neighborhood Search) 主逻辑
自适应大邻域搜索算法
"""

from typing import List, Dict, Tuple, Optional, Callable
import random
import math
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.solution import Solution
from models.node import Order
from algorithm.base_solver import BaseSolver
from algorithm.operators import DestroyOperators, RepairOperators
from algorithm.objective import ObjectiveFunction, check_validity
from algorithm.greedy import GreedyInsertion
import config


class ALNS(BaseSolver):
    """
    自适应大邻域搜索算法 (Adaptive Large Neighborhood Search)
    
    主要特点:
    1. 使用破坏-修复框架进行邻域搜索
    2. 多种破坏和修复算子
    3. 自适应选择算子 (根据历史表现调整权重)
    4. 使用模拟退火作为接受准则
    5. 参数根据问题规模自适应调整
    """
    
    def __init__(
        self,
        max_iterations: int = None,
        initial_temperature: float = None,
        cooling_rate: float = None,
        min_temperature: float = None,
        random_seed: int = None,
        verbose: bool = True,
        num_orders: int = None,  # 用于自适应参数调整
        num_vehicles: int = None  # 用于候选骑手筛选优化
    ):
        # 调用父类构造函数
        super().__init__(random_seed=random_seed, verbose=verbose)
        
        # 保存订单数量用于自适应调整
        self._num_orders = num_orders
        self._num_vehicles = num_vehicles
        
        # 算法参数（根据问题规模自适应）
        self.max_iterations = max_iterations or config.MAX_ITERATIONS
        self.initial_temperature = initial_temperature or config.INITIAL_TEMPERATURE
        self.min_temperature = min_temperature or config.MIN_TEMPERATURE
        
        # 冷却率根据问题规模自适应调整
        if cooling_rate is not None:
            self.cooling_rate = cooling_rate
        else:
            self.cooling_rate = self._adaptive_cooling_rate(num_orders)
        
        # 随机种子
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # 目标函数
        self.objective = ObjectiveFunction()
        
        # 算子（传入骑手数量用于空间筛选优化）
        self.destroy_ops = DestroyOperators(random_seed=random_seed)
        self.repair_ops = RepairOperators(random_seed=random_seed, num_vehicles=num_vehicles)
        
        # 初始解生成器
        self.greedy = GreedyInsertion(self.objective)
        
        # 统计信息
        self.history: List[Dict] = []
        self.best_cost_history: List[float] = []
        self.current_cost_history: List[float] = []
        
        # 算子得分参数
        self.sigma_1 = config.SIGMA_1  # 新全局最优
        self.sigma_2 = config.SIGMA_2  # 比当前解更好
        self.sigma_3 = config.SIGMA_3  # 接受了差解
    
    def _adaptive_cooling_rate(self, num_orders: int = None) -> float:
        """
        根据问题规模自适应计算冷却率
        
        健康的收敛曲线需要三个阶段：
        1. 快速下降期：高温，接受差解，快速探索
        2. 震荡寻优期：中温，偶尔接受差解，跳出局部最优
        3. 平稳收敛期：低温，只接受更优解
        
        冷却率决定了从高温到低温的速度：
        - 太快：过早收敛，陷入局部最优
        - 太慢：收敛缓慢，浪费计算资源
        
        Args:
            num_orders: 订单数量
            
        Returns:
            冷却率
        """
        if num_orders is None:
            return config.COOLING_RATE
        
        # 根据问题规模调整
        # 大规模问题需要更慢的冷却（更多探索时间）
        if num_orders <= 20:
            return 0.995   # 小规模：较快冷却
        elif num_orders <= 50:
            return 0.9975  # 中规模
        elif num_orders <= 100:
            return 0.998   # 较大规模
        else:
            return 0.999   # 大规模：慢冷却
    
    def _calculate_initial_temperature(
        self, 
        initial_cost: float, 
        num_orders: int = 20
    ) -> float:
        """
        自适应计算初始温度
        
        根据 SA 原理，初始温度应使接受差解的概率 P ≈ 50%
        T0 = -delta / ln(0.5)
        
        优化策略：
        1. tau 参数根据问题规模动态调整
        2. 小规模问题需要更高探索率，大规模问题需要更稳定
        3. 确保初始温度足够高以避免过早收敛
        
        Args:
            initial_cost: 初始解的成本
            num_orders: 订单数量（用于调整tau）
            
        Returns:
            初始温度
        """
        if initial_cost <= 0:
            return self.initial_temperature
        
        # 根据问题规模动态调整tau
        # 小规模：需要更多探索，使用较大的tau
        # 大规模：解空间大，使用较小的tau避免过度随机
        if num_orders <= 20:
            tau = 0.08  # 小规模: 高探索
        elif num_orders <= 50:
            tau = 0.06  # 中规模
        elif num_orders <= 100:
            tau = 0.05  # 较大规模
        else:
            tau = 0.04  # 大规模
        
        # 计算初始温度: T0 = -delta / ln(0.5)
        # delta 表示平均成本变化量
        delta = tau * initial_cost
        temperature = -delta / np.log(0.5)  # ln(0.5) ≈ -0.693
        
        # 确保温度在合理范围内
        # 最低不能低于成本的0.5%，最高不超过成本的20%
        min_temp = initial_cost * 0.005
        max_temp = initial_cost * 0.20
        temperature = max(min_temp, min(max_temp, temperature))
        
        if self.verbose:
            print(f"自适应温度初始化: T0 = {temperature:.2f} (tau={tau}, 基于初始成本 {initial_cost:.2f})")
        
        return round(temperature, 4)
    
    def solve(self, initial_solution: Solution) -> Solution:
        """
        执行ALNS算法求解
        
        Args:
            initial_solution: 初始解 (可以是空解)
        
        Returns:
            最优解
        """
        start_time = time.time()
        
        # 生成初始可行解
        if self.verbose:
            print("=" * 60)
            print(f"ALNS 算法开始")
            print("=" * 60)
            print("生成初始解...")
        
        # 使用贪心算法生成初始解
        current_solution = self.greedy.generate_initial_solution(initial_solution)
        
        current_cost = self.objective.calculate(current_solution)
        
        best_solution = current_solution.copy()
        best_cost = current_cost
        
        if self.verbose:
            print(f"初始解成本: {current_cost:.2f}")
            print(f"已分配订单: {len(initial_solution.orders) - current_solution.num_unassigned}")
            print(f"未分配订单: {current_solution.num_unassigned}")
            print("-" * 60)
        
        # 自适应温度初始化（根据问题规模调整）
        # T0 = -delta / ln(0.5)，其中delta = tau * initial_cost
        # tau参数根据订单数量动态调整
        num_orders = len(initial_solution.orders)
        num_vehicles = len(initial_solution.vehicles)
        temperature = self._calculate_initial_temperature(current_cost, num_orders)
        
        # 根据问题规模自适应调整冷却率（如果未在构造函数中设置）
        if self._num_orders is None:
            self._num_orders = num_orders
            self.cooling_rate = self._adaptive_cooling_rate(num_orders)
        
        if self.verbose:
            print(f"冷却率: {self.cooling_rate} (自适应，订单数={num_orders})")
            print(f"候选骑手筛选: 启用，最多{self.repair_ops.max_candidates}个候选")
        
        # 主循环
        iterations_since_improvement = 0
        
        for iteration in range(self.max_iterations):
            # 1. 选择破坏和修复算子
            destroy_name, destroy_op = self.destroy_ops.select_operator()
            repair_name, repair_op = self.repair_ops.select_operator()
            
            # 2. 复制当前解
            temp_solution = current_solution.copy()
            
            # 3. 执行破坏
            removed_orders = destroy_op(temp_solution)
            
            # 4. 执行修复
            repair_op(temp_solution, removed_orders)
            
            # 5. 计算新解成本
            temp_cost = self.objective.calculate(temp_solution)
            
            # 6. 决定是否接受新解
            accept = False
            score = 0
            
            if temp_cost < best_cost:
                # 新全局最优
                best_solution = temp_solution.copy()
                best_cost = temp_cost
                accept = True
                score = self.sigma_1
                iterations_since_improvement = 0
                
                if self.verbose and iteration % 50 == 0:
                    print(f"[迭代 {iteration}] 🌟 新最优解! 成本: {best_cost:.2f}")
            
            elif temp_cost < current_cost:
                # 比当前解更好
                accept = True
                score = self.sigma_2
            
            else:
                # 使用模拟退火接受准则
                delta = temp_cost - current_cost
                acceptance_prob = math.exp(-delta / temperature) if temperature > 0 else 0
                
                if random.random() < acceptance_prob:
                    accept = True
                    score = self.sigma_3
            
            # 7. 更新当前解
            if accept:
                current_solution = temp_solution
                current_cost = temp_cost
            else:
                iterations_since_improvement += 1
            
            # 8. 更新算子权重
            self.destroy_ops.update_weights(destroy_name, score)
            self.repair_ops.update_weights(repair_name, score)
            
            # 9. 降温
            temperature = max(self.min_temperature, temperature * self.cooling_rate)
            
            # 10. 记录历史
            self.best_cost_history.append(best_cost)
            self.current_cost_history.append(current_cost)
            
            self.history.append({
                'iteration': iteration,
                'destroy_op': destroy_name,
                'repair_op': repair_name,
                'temp_cost': temp_cost,
                'current_cost': current_cost,
                'best_cost': best_cost,
                'temperature': temperature,
                'accepted': accept
            })
            
            # 进度输出
            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"[迭代 {iteration + 1}/{self.max_iterations}] "
                      f"当前: {current_cost:.2f}, 最优: {best_cost:.2f}, "
                      f"温度: {temperature:.4f}")
        
        # 算法结束
        elapsed_time = time.time() - start_time
        
        if self.verbose:
            print("-" * 60)
            print("ALNS 算法结束")
            print(f"总迭代次数: {self.max_iterations}")
            print(f"运行时间: {elapsed_time:.2f} 秒")
            print(f"最终最优成本: {best_cost:.2f}")
            print("=" * 60)
        
        return best_solution
    
    def get_statistics(self) -> Dict:
        """获取算法运行统计信息"""
        if len(self.history) == 0:
            return {}
        
        accepted_count = sum(1 for h in self.history if h['accepted'])
        
        destroy_usage = {}
        repair_usage = {}
        
        for h in self.history:
            destroy_usage[h['destroy_op']] = destroy_usage.get(h['destroy_op'], 0) + 1
            repair_usage[h['repair_op']] = repair_usage.get(h['repair_op'], 0) + 1
        
        return {
            'total_iterations': len(self.history),
            'accepted_count': accepted_count,
            'acceptance_rate': accepted_count / len(self.history),
            'initial_cost': self.history[0]['current_cost'],
            'final_cost': self.history[-1]['best_cost'],
            'improvement': self.history[0]['current_cost'] - self.history[-1]['best_cost'],
            'destroy_usage': destroy_usage,
            'repair_usage': repair_usage,
            'destroy_weights': self.destroy_ops.weights.copy(),
            'repair_weights': self.repair_ops.weights.copy()
        }
    
    def reset(self):
        """重置算法状态"""
        self.history = []
        self.best_cost_history = []
        self.current_cost_history = []
        
        # 重置算子权重
        self.destroy_ops = DestroyOperators()
        self.repair_ops = RepairOperators()


class ParallelALNS:
    """
    并行ALNS (多起点)
    
    从多个不同的初始解开始, 并行运行ALNS
    """
    
    def __init__(
        self,
        num_runs: int = 5,
        max_iterations: int = None,
        random_seed: int = None,
        verbose: bool = True
    ):
        self.num_runs = num_runs
        self.max_iterations = max_iterations or config.MAX_ITERATIONS
        self.random_seed = random_seed
        self.verbose = verbose
    
    def solve(self, initial_solution: Solution) -> Solution:
        """
        执行多次ALNS并返回最优解
        """
        best_solution = None
        best_cost = float('inf')
        
        for run in range(self.num_runs):
            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"运行 {run + 1}/{self.num_runs}")
                print(f"{'=' * 60}")
            
            # 使用不同的随机种子
            seed = (self.random_seed + run) if self.random_seed else None
            
            alns = ALNS(
                max_iterations=self.max_iterations,
                random_seed=seed,
                verbose=self.verbose
            )
            
            solution = alns.solve(initial_solution.copy())
            cost = alns.objective.calculate(solution)
            
            if cost < best_cost:
                best_solution = solution
                best_cost = cost
                
                if self.verbose:
                    print(f"✓ 更新最优解! 成本: {best_cost:.2f}")
        
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"所有运行完成, 最优成本: {best_cost:.2f}")
            print(f"{'=' * 60}")
        
        return best_solution


def solve_pdptw(
    initial_solution: Solution,
    max_iterations: int = None,
    random_seed: int = None,
    verbose: bool = True,
    num_orders: int = None,
    num_vehicles: int = None
) -> Solution:
    """
    便捷函数: 求解PDPTW问题
    
    Args:
        initial_solution: 初始解 (包含订单和骑手信息)
        max_iterations: 最大迭代次数
        random_seed: 随机种子
        verbose: 是否输出过程信息
        num_orders: 订单数量（用于自适应参数）
        num_vehicles: 骑手数量（用于候选筛选优化）
    
    Returns:
        最优解
    """
    # 自动推断订单和骑手数量
    if num_orders is None:
        num_orders = len(initial_solution.orders)
    if num_vehicles is None:
        num_vehicles = len(initial_solution.vehicles)
    
    alns = ALNS(
        max_iterations=max_iterations,
        random_seed=random_seed,
        verbose=verbose,
        num_orders=num_orders,
        num_vehicles=num_vehicles
    )
    
    return alns.solve(initial_solution)
