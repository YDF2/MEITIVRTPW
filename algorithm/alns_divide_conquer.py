# -*- coding: utf-8 -*-
"""
ALNS分治求解器 - 使用ALNS求解子问题
"""

import numpy as np
from typing import List, Optional
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cluster import KMeans

from models.solution import Solution
from models.node import Order
from algorithm.base_solver import BaseSolver
from algorithm.divide_and_conquer import DivideAndConquerSolver


class ALNSDivideAndConquerSolver(BaseSolver):
    """
    ALNS分治求解器
    
    使用 ALNS 求解子问题
    """
    
    def __init__(
        self,
        num_clusters: int = None,
        sub_iterations: int = 300,
        global_iterations: int = 50,
        random_seed: int = 112,
        verbose: bool = True,
        use_parallel: bool = True,
        max_workers: int = None,
        skip_global_optimization: bool = False
    ):
        """
        Args:
            num_clusters: 聚类数量（None时自动确定）
            sub_iterations: 每个子问题的ALNS迭代次数
            global_iterations: 全局优化迭代次数
            random_seed: 随机种子
            verbose: 是否输出详细信息
            use_parallel: 是否使用多进程并行
            max_workers: 最大工作进程数
            skip_global_optimization: 是否跳过全局优化
        """
        super().__init__(random_seed=random_seed, verbose=verbose)
        
        self.num_clusters = num_clusters
        self.sub_iterations = sub_iterations
        self.global_iterations = global_iterations
        self.skip_global_optimization = skip_global_optimization
        self.use_parallel = use_parallel
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)
        
        # 用于保存全局ALNS实例以便可视化
        self.global_alns_solver = None
        self.destroy_ops = None
        self.repair_ops = None
        self.best_cost_history = []
        self.current_cost_history = []
        
        if self.verbose:
            print(f"  ✓ ALNS分治求解器初始化")
            if self.use_parallel:
                print(f"  ✓ 多进程并行已启用，最大工作进程: {self.max_workers}")
    
    def solve(self, initial_solution: Solution) -> Solution:
        """
        使用DivideAndConquer框架，但子问题用ALNS求解
        
        实际上复用DivideAndConquerSolver
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("ALNS 分治求解器")
            print("=" * 60)
            print(f"策略: 聚类 → ALNS并行求解 → 合并 → 全局优化")
            print("-" * 60)
        
        start_time = time.time()
        
        # 使用DivideAndConquerSolver框架
        dc_solver = DivideAndConquerSolver(
            num_clusters=self.num_clusters,
            sub_iterations=self.sub_iterations,
            global_iterations=self.global_iterations,
            random_seed=self.random_seed,
            verbose=self.verbose,
            use_parallel=self.use_parallel,
            max_workers=self.max_workers,
            skip_global_optimization=self.skip_global_optimization
        )
        
        solution = dc_solver.solve(initial_solution)
        
        # 获取全局ALNS的统计信息用于可视化
        if hasattr(dc_solver, 'global_alns_solver') and dc_solver.global_alns_solver is not None:
            self.global_alns_solver = dc_solver.global_alns_solver
            self.destroy_ops = dc_solver.global_alns_solver.destroy_ops
            self.repair_ops = dc_solver.global_alns_solver.repair_ops
            self.best_cost_history = dc_solver.global_alns_solver.best_cost_history
            self.current_cost_history = dc_solver.global_alns_solver.current_cost_history
            if self.verbose:
                print(f"  ✓ 已保存全局ALNS统计信息: {len(self.best_cost_history)} 次迭代")
        elif self.verbose:
            print(f"  ⚠ 警告: 未找到全局ALNS统计信息")
        
        elapsed = time.time() - start_time
        
        if self.verbose:
            print("-" * 60)
            print(f"ALNS分治求解完成")
            print(f"总耗时: {elapsed:.2f} 秒")
            print(f"最终成本: {solution.calculate_cost():.2f}")
            print("=" * 60)
        
        return solution
    
    def get_statistics(self):
        """获取统计信息"""
        stats = {
            'solver_type': 'ALNS-DivideAndConquer',
            'sub_iterations': self.sub_iterations,
            'global_iterations': self.global_iterations,
            'use_parallel': self.use_parallel,
            'max_workers': self.max_workers
        }
        
        # 如果有全局ALNS统计信息，也包含进来
        if self.global_alns_solver is not None:
            stats['has_global_alns'] = True
            stats['total_iterations'] = len(self.best_cost_history)
        else:
            stats['has_global_alns'] = False
        
        return stats
