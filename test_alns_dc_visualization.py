# -*- coding: utf-8 -*-
"""
测试ALNS分治求解器的算子可视化功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.generator import generate_problem_instance
from algorithm.alns_divide_conquer import ALNSDivideAndConquerSolver

# 生成测试问题
print("生成测试问题...")
initial_solution = generate_problem_instance(
    num_orders=60,
    num_vehicles=12,
    random_seed=42
)

# 创建求解器
print("\n创建ALNS分治求解器...")
solver = ALNSDivideAndConquerSolver(
    num_clusters=None,
    sub_iterations=100,
    global_iterations=30,
    random_seed=42,
    verbose=True,
    use_parallel=False,  # 串行以便调试
    skip_global_optimization=False  # 启用全局优化
)

# 求解
print("\n开始求解...")
solution = solver.solve(initial_solution)

# 检查统计信息
print("\n" + "="*60)
print("统计信息检查:")
print("="*60)

print(f"是否有global_alns_solver: {solver.global_alns_solver is not None}")
print(f"是否有destroy_ops: {solver.destroy_ops is not None}")
print(f"是否有repair_ops: {solver.repair_ops is not None}")
print(f"best_cost_history长度: {len(solver.best_cost_history)}")
print(f"current_cost_history长度: {len(solver.current_cost_history)}")

if solver.destroy_ops is not None:
    print(f"\n破坏算子权重:")
    for name, weight in solver.destroy_ops.weights.items():
        print(f"  {name}: {weight:.2f}")

if solver.repair_ops is not None:
    print(f"\n修复算子权重:")
    for name, weight in solver.repair_ops.weights.items():
        print(f"  {name}: {weight:.2f}")

print("\n" + "="*60)
print("测试完成!")
print("="*60)
