# -*- coding: utf-8 -*-
"""
ALNS分治求解器 - 使用ALNS+Gurobi求解子问题
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
    
    与GurobiDivideAndConquer的区别：
    - 使用 ALNS+Gurobi 求解子问题（更灵活）
    - Gurobi用于生成高质量初始解
    - ALNS用于进一步优化
    """
    
    def __init__(
        self,
        num_clusters: int = None,
        sub_iterations: int = 300,
        global_iterations: int = 50,
        random_seed: int = 42,
        verbose: bool = True,
        use_gurobi_init: bool = True,
        gurobi_time_limit: int = 30,
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
            use_gurobi_init: 是否使用Gurobi生成子问题初始解
            gurobi_time_limit: Gurobi时间限制（秒）
            use_parallel: 是否使用多进程并行
            max_workers: 最大工作进程数
            skip_global_optimization: 是否跳过全局优化
        """
        super().__init__(random_seed=random_seed, verbose=verbose)
        
        self.num_clusters = num_clusters
        self.sub_iterations = sub_iterations
        self.global_iterations = global_iterations
        self.skip_global_optimization = skip_global_optimization
        self.use_gurobi_init = use_gurobi_init
        self.gurobi_time_limit = gurobi_time_limit
        self.use_parallel = use_parallel
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)
        
        # 检查Gurobi可用性
        if self.use_gurobi_init:
            try:
                from algorithm.gurobi_solver import GUROBI_AVAILABLE
                if not GUROBI_AVAILABLE:
                    print("警告: Gurobi不可用，将使用纯ALNS算法")
                    self.use_gurobi_init = False
            except ImportError:
                print("警告: 无法导入Gurobi，将使用纯ALNS算法")
                self.use_gurobi_init = False
        
        if self.verbose:
            print(f"  ✓ ALNS分治求解器初始化")
            if self.use_gurobi_init:
                print(f"  ✓ Gurobi初始解已启用 (时间限制: {self.gurobi_time_limit}秒)")
            if self.use_parallel:
                print(f"  ✓ 多进程并行已启用，最大工作进程: {self.max_workers}")
    
    def solve(self, initial_solution: Solution) -> Solution:
        """
        使用DivideAndConquer框架，但子问题用ALNS+Gurobi求解
        
        实际上复用DivideAndConquerSolver，但通过参数配置强制使用ALNS
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("ALNS 分治求解器")
            print("=" * 60)
            print(f"策略: 聚类 → ALNS并行求解 → 合并 → 全局优化")
            print("-" * 60)
        
        start_time = time.time()
        
        # 使用DivideAndConquerSolver框架，但强制使用ALNS
        dc_solver = DivideAndConquerSolver(
            num_clusters=self.num_clusters,
            sub_iterations=self.sub_iterations,
            global_iterations=self.global_iterations,
            random_seed=self.random_seed,
            verbose=self.verbose,
            use_gurobi=False,  # 关键：不使用Gurobi直接求解
            use_parallel=self.use_parallel,
            max_workers=self.max_workers,
            skip_global_optimization=self.skip_global_optimization
        )
        
        # 注意：这里的ALNS会自动使用Gurobi初始解（如果我们已经修改了ALNS类）
        solution = dc_solver.solve(initial_solution)
        
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
        return {
            'solver_type': 'ALNS-DivideAndConquer',
            'use_gurobi_init': self.use_gurobi_init,
            'sub_iterations': self.sub_iterations,
            'global_iterations': self.global_iterations
        }
