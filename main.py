# -*- coding: utf-8 -*-
"""
FoodDelivery_Optimizer - 外卖配送路径规划系统
主程序入口

基于ALNS (自适应大邻域搜索) 算法求解
PDPTW (带时间窗的取送货路径问题)
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from models.solution import Solution
from models.node import Node, NodeType, Order
from models.vehicle import Vehicle
from algorithm.alns import ALNS, solve_pdptw
from algorithm.objective import ObjectiveFunction, check_validity
from algorithm.greedy import GreedyInsertion
from utils.generator import DataGenerator, generate_problem_instance
from utils.visualizer import SolutionVisualizer, plot_solution
from utils.file_io import (
    save_solution_to_json, 
    load_solution_from_json,
    save_experiment_results,
    save_problem_to_json
)


def run_experiment(
    num_orders: int = 20,
    num_vehicles: int = 5,
    max_iterations: int = 500,
    random_seed: int = 42,
    save_results: bool = True,
    visualize: bool = True,
    experiment_name: str = None
):
    """
    运行完整的优化实验
    
    Args:
        num_orders: 订单数量
        num_vehicles: 骑手数量
        max_iterations: ALNS最大迭代次数
        random_seed: 随机种子
        save_results: 是否保存结果
        visualize: 是否可视化
        experiment_name: 实验名称
    """
    solver_name = "ALNS"
    
    print("=" * 70)
    print(f"   外卖配送路径规划系统 (PDPTW + {solver_name})")
    print("=" * 70)
    print(f"求解器:     {solver_name}")
    print(f"订单数量:   {num_orders}")
    print(f"骑手数量:   {num_vehicles}")
    print(f"最大迭代:   {max_iterations}")
    print(f"随机种子:   {random_seed}")
    print("-" * 70)
    
    # 1. 生成问题实例
    print("\n[步骤1] 生成问题实例...")
    time_start_gen = time.time()
    
    initial_solution = generate_problem_instance(
        num_orders=num_orders,
        num_vehicles=num_vehicles,
        random_seed=random_seed
    )
    
    time_gen = time.time() - time_start_gen
    
    print(f"  ✓ 配送站位置: ({initial_solution.depot.x:.1f}, {initial_solution.depot.y:.1f})")
    print(f"  ✓ 生成订单: {len(initial_solution.orders)} 个")
    print(f"  ✓ 生成骑手: {len(initial_solution.vehicles)} 个")
    print(f"  ✓ 生成耗时: {time_gen:.3f} 秒")
    
    # 2. 执行优化
    print(f"\n[步骤2] 执行 {solver_name} 优化...")
    time_start_solve = time.time()
    
    alns = ALNS(
        max_iterations=max_iterations,
        random_seed=random_seed,
        verbose=True
    )
    best_solution = alns.solve(initial_solution)
    alns_stats = alns.get_statistics()
    
    time_solve = time.time() - time_start_solve
    
    # 3. 验证结果
    print("\n[步骤3] 验证解的合法性...")
    is_valid, violations = check_validity(best_solution)
    
    if is_valid:
        print("  ✓ 解通过所有约束检查")
    else:
        print("  ✗ 发现约束违反:")
        for v in violations:
            print(f"    - {v}")
    
    # 4. 输出统计信息
    print("\n[步骤4] 优化结果统计")
    print("-" * 70)
    
    stats = best_solution.get_statistics()
    
    print(f"  总成本:       {stats['total_cost']:.2f}")
    print(f"  总行驶距离:   {stats['total_distance']:.2f}")
    print(f"  时间窗违反:   {stats['total_time_violation']:.2f}")
    print(f"  使用骑手数:   {stats['num_vehicles_used']}/{num_vehicles}")
    print(f"  未分配订单:   {stats['num_unassigned']}")
    print(f"  解可行性:     {'是' if stats['is_feasible'] else '否'}")
    print("-" * 70)
    print(f"  问题生成:     {time_gen:.3f} 秒")
    print(f"  求解时间:     {time_solve:.2f} 秒")
    print(f"  总时间:       {time_gen + time_solve:.2f} 秒")
    print(f"  总迭代次数:   {alns_stats['total_iterations']}")
    print(f"  接受率:       {alns_stats['acceptance_rate']:.2%}")
    print(f"  成本改进:     {alns_stats['improvement']:.2f}")
    
    # 5. 输出路径详情
    print("\n[步骤5] 骑手路径详情")
    print("-" * 70)
    
    for vehicle in best_solution.vehicles:
        if len(vehicle.route) > 0:
            route_str = " -> ".join([str(n) for n in vehicle.full_route])
            distance = vehicle.calculate_distance()
            violation = vehicle.calculate_time_violation()
            orders = vehicle.get_order_ids()
            
            print(f"  骑手 {vehicle.id}:")
            print(f"    路径: {route_str}")
            print(f"    距离: {distance:.2f}, 超时: {violation:.2f}")
            print(f"    订单: {sorted(orders)}")
    
    # 6. 保存结果
    if save_results:
        print("\n[步骤6] 保存实验结果...")
        
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_dir = os.path.join(PROJECT_ROOT, "data", "results", experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备额外信息
        additional_info = {
            "solver_type": "alns",
            "num_orders": num_orders,
            "num_vehicles": num_vehicles,
            "random_seed": random_seed,
            "max_iterations": max_iterations,
            "total_iterations": alns_stats['total_iterations'],
            "acceptance_rate": alns_stats['acceptance_rate'],
            "improvement": alns_stats['improvement']
        }
        
        # 保存解（包含时间信息）
        save_solution_to_json(
            best_solution,
            "solution.json",
            output_dir,
            solver_name=solver_name,
            solve_time=time_solve,
            generation_time=time_gen,
            additional_info=additional_info
        )
        
        # 保存问题实例
        save_problem_to_json(
            initial_solution,
            "problem_instance.json",
            output_dir
        )
        
        # 保存详细统计信息
        summary_data = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "solver": "ALNS",
            "problem": {
                "num_orders": num_orders,
                "num_vehicles": num_vehicles,
                "random_seed": random_seed
            },
            "timing": {
                "generation_time_seconds": time_gen,
                "solve_time_seconds": time_solve,
                "total_time_seconds": time_gen + time_solve
            },
            "results": stats,
            "solver_specific": additional_info
        }
        
        summary_path = os.path.join(output_dir, "experiment_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ 结果已保存至: {output_dir}")
    
    # 7. 可视化
    if visualize:
        print("\n[步骤7] 生成可视化图...")
        
        visualizer = SolutionVisualizer()
        
        # 绘制路径图
        fig1 = visualizer.plot(
            best_solution,
            title=f"外卖配送路径规划 (ALNS) (订单: {num_orders}, 骑手: {num_vehicles})",
            save_path=os.path.join(output_dir, "route_visualization.png") if save_results else None
        )
        
        # 绘制收敛曲线
        fig2 = visualizer.plot_convergence(
            alns.best_cost_history,
            alns.current_cost_history,
            title="ALNS 算法收敛曲线",
            save_path=os.path.join(output_dir, "convergence.png") if save_results else None
        )
        
        # 绘制算子权重
        fig3 = visualizer.plot_operator_weights(
            alns.destroy_ops.weights,
            alns.repair_ops.weights,
            title="算子权重分布",
            save_path=os.path.join(output_dir, "operator_weights.png") if save_results else None
        )
        
        print("  ✓ 可视化图已生成")
        
        # 显示图形
        import matplotlib.pyplot as plt
        plt.show()
    
    print("\n" + "=" * 70)
    print("   实验完成!")
    print("=" * 70)
    
    return best_solution, alns_stats


def run_benchmark(
    order_sizes: list = [10, 20, 30, 50],
    num_runs: int = 3,
    max_iterations: int = 500
):
    """
    运行基准测试，比较不同规模问题的求解性能
    """
    print("=" * 70)
    print("   基准测试模式")
    print("=" * 70)
    
    results = []
    
    for num_orders in order_sizes:
        print(f"\n测试规模: {num_orders} 订单")
        print("-" * 50)
        
        run_costs = []
        run_times = []
        
        for run in range(num_runs):
            seed = 42 + run
            
            # 生成问题
            solution = generate_problem_instance(
                num_orders=num_orders,
                num_vehicles=max(3, num_orders // 5),
                random_seed=seed
            )
            
            # 求解
            start_time = time.time()
            alns = ALNS(max_iterations=max_iterations, random_seed=seed, verbose=False)
            best = alns.solve(solution)
            elapsed = time.time() - start_time
            
            cost = best.calculate_cost()
            run_costs.append(cost)
            run_times.append(elapsed)
            
            print(f"  运行 {run + 1}: 成本 = {cost:.2f}, 时间 = {elapsed:.2f}s")
        
        avg_cost = sum(run_costs) / len(run_costs)
        avg_time = sum(run_times) / len(run_times)
        
        results.append({
            'num_orders': num_orders,
            'avg_cost': avg_cost,
            'min_cost': min(run_costs),
            'max_cost': max(run_costs),
            'avg_time': avg_time
        })
        
        print(f"  平均成本: {avg_cost:.2f}, 平均时间: {avg_time:.2f}s")
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("基准测试结果汇总")
    print("=" * 70)
    print(f"{'订单数':<10} {'平均成本':<15} {'最优成本':<15} {'平均时间':<15}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['num_orders']:<10} {r['avg_cost']:<15.2f} {r['min_cost']:<15.2f} {r['avg_time']:<15.2f}s")
    
    return results


def demo_mode():
    """
    演示模式 - 使用小规模问题进行演示
    """
    print("=" * 70)
    print("   演示模式 - 小规模问题")
    print("=" * 70)
    
    # 使用较小的问题规模
    run_experiment(
        num_orders=10,
        num_vehicles=3,
        max_iterations=200,
        random_seed=42,
        save_results=True,
        visualize=True,
        experiment_name="demo"
    )


def main():
    """
    主函数 - 解析命令行参数并运行
    """
    parser = argparse.ArgumentParser(
        description='外卖配送路径规划系统 (PDPTW + ALNS)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --demo                              # 运行演示
  python main.py --orders 20 --vehicles 5            # 自定义规模
  python main.py --orders 100 --vehicles 20          # 大规模问题
  python main.py --benchmark                         # 运行基准测试
        """
    )
    
    parser.add_argument('--demo', action='store_true', 
                       help='运行演示模式 (小规模问题)')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行基准测试')
    parser.add_argument('--orders', type=int, default=20,
                       help='订单数量 (默认: 20)')
    parser.add_argument('--vehicles', type=int, default=5,
                       help='骑手数量 (默认: 5)')
    parser.add_argument('--iterations', type=int, default=500,
                       help='ALNS最大迭代次数 (默认: 500)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存结果')
    parser.add_argument('--no-viz', action='store_true',
                       help='不显示可视化')
    
    args = parser.parse_args()
 
    if args.demo:
        demo_mode()
    elif args.benchmark:
        run_benchmark()
    else:
        run_experiment(
            num_orders=args.orders,
            num_vehicles=args.vehicles,
            max_iterations=args.iterations,
            random_seed=args.seed,
            save_results=not args.no_save,
            visualize=not args.no_viz
        )


if __name__ == "__main__":
    main()
