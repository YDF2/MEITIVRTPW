# -*- coding: utf-8 -*-
"""
快速性能测试 - 对比不同全局优化配置
"""

import time
from utils.generator import generate_problem_instance
from algorithm.divide_and_conquer import DivideAndConquerSolver

def test_configuration(config_name, skip_global, global_iter, num_orders=200, num_vehicles=40):
    """测试单个配置"""
    print(f"\n{'='*70}")
    print(f"测试配置: {config_name}")
    print(f"{'='*70}")
    
    # 生成问题
    solution = generate_problem_instance(
        num_orders=num_orders,
        num_vehicles=num_vehicles,
        random_seed=42
    )
    
    # 创建求解器
    solver = DivideAndConquerSolver(
        skip_global_optimization=skip_global,
        global_iterations=global_iter,
        verbose=True
    )
    
    # 求解
    start_time = time.time()
    result = solver.solve(solution)
    total_time = time.time() - start_time
    
    # 输出结果
    print(f"\n总耗时: {total_time:.2f}秒")
    print(f"最终成本: {result.calculate_cost():.2f}")
    print(f"使用骑手: {result.num_used_vehicles}/{len(result.vehicles)}")
    print(f"未分配订单: {result.num_unassigned}")
    
    return {
        'name': config_name,
        'time': total_time,
        'cost': result.calculate_cost(),
        'vehicles': result.num_used_vehicles
    }


if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     全局优化性能对比测试                                      ║
    ║     200订单 + 40骑手                                          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    results = []
    
    # 配置1: 跳过全局优化（最快）
    print("\n[配置1] 跳过全局优化（推荐生产环境）")
    r1 = test_configuration(
        "跳过全局优化",
        skip_global=True,
        global_iter=50
    )
    results.append(r1)
    
    # 配置2: 30次迭代（快速）
    print("\n[配置2] 30次全局迭代（快速模式）")
    r2 = test_configuration(
        "30次迭代",
        skip_global=False,
        global_iter=30
    )
    results.append(r2)
    
    # 配置3: 50次迭代（默认）
    print("\n[配置3] 50次全局迭代（默认配置）")
    r3 = test_configuration(
        "50次迭代",
        skip_global=False,
        global_iter=50
    )
    results.append(r3)
    
    # 打印对比表
    print(f"\n\n{'='*80}")
    print("性能对比总结")
    print(f"{'='*80}")
    print(f"{'配置':<20} {'总耗时':<15} {'成本':<15} {'骑手数':<10}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['name']:<20} {r['time']:>10.2f}秒   {r['cost']:>12.2f}   {r['vehicles']:>7}")
    
    # 计算加速比
    base_time = results[2]['time']  # 50次迭代作为基准
    print(f"\n相对于50次迭代的加速比:")
    print(f"  跳过全局优化: {base_time/results[0]['time']:.2f}x")
    print(f"  30次迭代: {base_time/results[1]['time']:.2f}x")
    
    # 计算质量损失
    base_cost = results[2]['cost']
    print(f"\n相对于50次迭代的质量差异:")
    print(f"  跳过全局优化: {((results[0]['cost']-base_cost)/base_cost*100):+.2f}%")
    print(f"  30次迭代: {((results[1]['cost']-base_cost)/base_cost*100):+.2f}%")
    
    print(f"\n{'='*80}")
    print("✅ 测试完成！")
    print(f"{'='*80}\n")
