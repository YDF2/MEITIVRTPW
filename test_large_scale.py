# -*- coding: utf-8 -*-
"""
大规模问题测试 - ALNS 算法测试（包含分治策略）
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.generator import generate_problem_instance
from algorithm.alns import ALNS
from algorithm.divide_and_conquer import DivideAndConquerSolver

def test_solver(solver_name, num_orders, num_vehicles, **kwargs):
    """测试单个求解器"""
    print(f"\n{'='*70}")
    print(f"  测试 {solver_name} - {num_orders} 订单, {num_vehicles} 骑手")
    print(f"{'='*70}")
    
    # 生成问题
    print("生成问题实例...")
    time_start = time.time()
    solution = generate_problem_instance(
        num_orders=num_orders,
        num_vehicles=num_vehicles,
        random_seed=42
    )
    time_gen = time.time() - time_start
    print(f"  生成耗时: {time_gen:.3f} 秒")
    
    # 求解
    print(f"开始 {solver_name} 求解...")
    time_start = time.time()
    
    if solver_name == "分治求解器":
        solver = DivideAndConquerSolver(
            num_clusters=kwargs.get('num_clusters', None),
            sub_iterations=kwargs.get('sub_iterations', 300),
            global_iterations=kwargs.get('global_iterations', 100),
            random_seed=42,
            verbose=False
        )
        best = solver.solve(solution)
    else:  # ALNS
        max_iterations = kwargs.get('max_iterations', 500)
        alns = ALNS(max_iterations=max_iterations, random_seed=42, verbose=False)
        best = alns.solve(solution)
    
    time_solve = time.time() - time_start
    
    if best is None:
        print(f"  >>> {solver_name} 求解失败!")
        return None
    
    # 统计
    stats = best.get_statistics()
    print(f"\n  结果:")
    print(f"    总成本:       {stats['total_cost']:.2f}")
    print(f"    总距离:       {stats['total_distance']:.2f}")
    print(f"    时间窗违反:   {stats['total_time_violation']:.2f}")
    print(f"    使用骑手数:   {stats['num_vehicles_used']}/{num_vehicles}")
    print(f"    未分配订单:   {stats['num_unassigned']}")
    print(f"    求解时间:     {time_solve:.2f} 秒")
    
    return {
        'solver': solver_name,
        'cost': stats['total_cost'],
        'distance': stats['total_distance'],
        'violation': stats['total_time_violation'],
        'vehicles_used': stats['num_vehicles_used'],
        'unassigned': stats['num_unassigned'],
        'time': time_solve
    }

def main():
    print("="*70)
    print("  大规模问题求解器测试 (ALNS vs 分治策略)")
    print("="*70)
    
    # 测试不同规模
    test_cases = [
        (20, 5),   # 小规模 - 两种方法都测试
        (50, 10),  # 中规模 - 两种方法都测试
        (100, 20), # 大规模 - 主要测试分治
        (200, 40), # 超大规模 - 仅测试分治
    ]
    
    results = []
    
    for num_orders, num_vehicles in test_cases:
        # 小规模问题测试标准 ALNS
        if num_orders <= 50:
            result_alns = test_solver("ALNS", num_orders, num_vehicles, max_iterations=300)
            if result_alns:
                results.append(result_alns)
        
        # 所有规模都测试分治策略
        result_dc = test_solver(
            "分治求解器", 
            num_orders, 
            num_vehicles,
            sub_iterations=200,
            global_iterations=50
        )
        if result_dc:
            results.append(result_dc)
        
        print()
    
    # 汇总结果
    print("\n" + "="*70)
    print("  汇总结果")
    print("="*70)
    print(f"{'求解器':<12} {'订单':<6} {'成本':<12} {'距离':<12} {'时间(秒)':<12}")
    print("-"*70)
    
    for r in results:
        print(f"{r['solver']:<12} {'-':<6} {r['cost']:<12.2f} {r['distance']:<12.2f} {r['time']:<12.2f}")
    
    print("="*70)

if __name__ == "__main__":
    main()
