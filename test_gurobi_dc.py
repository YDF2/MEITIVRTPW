# 测试Gurobi分治求解器
from algorithm.divide_and_conquer import DivideAndConquerSolver
from utils.generator import DataGenerator
from models.solution import Solution

# 生成25订单问题
gen = DataGenerator(random_seed=42)
depot = gen.generate_depot()
orders = [gen.generate_order(i) for i in range(25)]
vehicles = [gen.generate_vehicle(i, depot) for i in range(6)]
sol = Solution(vehicles, orders, depot)

print(f"生成问题: {len(orders)}订单, {len(vehicles)}骑手")

# 测试Gurobi分治（禁用并行）
solver = DivideAndConquerSolver(
    num_clusters=2,
    use_gurobi=True,
    use_parallel=False,
    verbose=True
)

result = solver.solve(sol)

print(f"\n结果:")
print(f"总成本: {result.calculate_cost():.2f}")
print(f"已分配: {len(orders) - len(result.unassigned_orders)}/{len(orders)}")
print(f"使用骑手: {sum(1 for v in result.vehicles if len(v.route) > 0)}/{len(vehicles)}")
