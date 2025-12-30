# -*- coding: utf-8 -*-
"""
求解器基类
为所有求解器提供统一的接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from models.solution import Solution


class BaseSolver(ABC):
    """
    求解器抽象基类
    
    所有求解器都应该继承这个类并实现solve方法
    """
    
    def __init__(self, random_seed: Optional[int] = None, verbose: bool = True):
        """
        初始化求解器
        
        Args:
            random_seed: 随机种子
            verbose: 是否输出详细信息
        """
        self.random_seed = random_seed
        self.verbose = verbose
        self.solver_name = self.__class__.__name__
    
    @abstractmethod
    def solve(self, initial_solution: Solution) -> Solution:
        """
        求解PDPTW问题
        
        Args:
            initial_solution: 初始解（包含订单和骑手信息）
        
        Returns:
            最优解
        """
        pass
    
    def get_statistics(self) -> Dict:
        """
        获取求解统计信息（可选实现）
        
        Returns:
            统计信息字典
        """
        return {}
    
    def reset(self):
        """
        重置求解器状态（可选实现）
        """
        pass
    
    def __str__(self) -> str:
        return self.solver_name
    
    def __repr__(self) -> str:
        return f"{self.solver_name}(random_seed={self.random_seed})"
