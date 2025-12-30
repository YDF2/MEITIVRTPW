# -*- coding: utf-8 -*-
"""
查看实验结果工具
"""

import json
import os
import sys
from datetime import datetime

def view_experiment_summary(exp_dir):
    """查看实验总结"""
    summary_path = os.path.join(exp_dir, "experiment_summary.json")
    
    if not os.path.exists(summary_path):
        print(f"错误: 找不到 {summary_path}")
        return
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 70)
    print(f"  实验结果: {data['experiment_name']}")
    print("=" * 70)
    print(f"时间: {data['timestamp']}")
    print(f"求解器: {data['solver']}")
    print()
    
    print("问题规模:")
    for key, value in data['problem'].items():
        print(f"  {key}: {value}")
    print()
    
    print("时间统计:")
    timing = data['timing']
    print(f"  问题生成: {timing['generation_time_seconds']:.3f} 秒")
    print(f"  求解时间: {timing['solve_time_seconds']:.2f} 秒")
    print(f"  总时间:   {timing['total_time_seconds']:.2f} 秒")
    print()
    
    print("结果统计:")
    results = data['results']
    print(f"  总成本:       {results['total_cost']:.2f}")
    print(f"  总距离:       {results['total_distance']:.2f}")
    print(f"  时间窗违反:   {results['total_time_violation']:.2f}")
    print(f"  使用骑手数:   {results['num_vehicles_used']}")
    print(f"  未分配订单:   {results['num_unassigned']}")
    print(f"  可行性:       {results['is_feasible']}")
    print()
    
    if 'solver_specific' in data:
        print("求解器特定信息:")
        for key, value in data['solver_specific'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    print("=" * 70)

def list_experiments(results_dir="data/results"):
    """列出所有实验"""
    if not os.path.exists(results_dir):
        print(f"结果目录不存在: {results_dir}")
        return []
    
    experiments = []
    for name in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, name)
        if os.path.isdir(exp_path):
            summary_path = os.path.join(exp_path, "experiment_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                experiments.append({
                    'name': name,
                    'path': exp_path,
                    'timestamp': data.get('timestamp', 'Unknown'),
                    'solver': data.get('solver', 'Unknown'),
                    'orders': data.get('problem', {}).get('num_orders', 0),
                    'cost': data.get('results', {}).get('total_cost', 0),
                    'time': data.get('timing', {}).get('solve_time_seconds', 0)
                })
    
    # 按时间排序
    experiments.sort(key=lambda x: x['timestamp'], reverse=True)
    
    print("\n可用的实验结果:")
    print("-" * 100)
    print(f"{'编号':<6} {'实验名称':<30} {'求解器':<12} {'订单数':<8} {'成本':<12} {'时间(秒)':<10}")
    print("-" * 100)
    
    for i, exp in enumerate(experiments, 1):
        print(f"{i:<6} {exp['name']:<30} {exp['solver']:<12} {exp['orders']:<8} {exp['cost']:<12.2f} {exp['time']:<10.2f}")
    
    return experiments

def main():
    if len(sys.argv) > 1:
        # 查看指定实验
        exp_name = sys.argv[1]
        exp_dir = os.path.join("data/results", exp_name)
        view_experiment_summary(exp_dir)
    else:
        # 列出所有实验
        experiments = list_experiments()
        
        if experiments:
            print("\n使用方法: python view_results.py <实验名称>")
            print(f"例如: python view_results.py {experiments[0]['name']}")

if __name__ == "__main__":
    main()
