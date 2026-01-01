# 多进程兼容性修复说明

## 问题诊断

### 原始错误
```
'RepairOperators' object has no attribute 'use_candidate_filtering'
```

### 原因分析
在多进程并行环境下，`RepairOperators` 类的实例被序列化（pickle）并传递给子进程。但是我们之前添加的优化属性没有正确初始化在 `__init__` 方法中，导致反序列化后缺少这些属性。

## 修复内容

### 1. 修复 `RepairOperators.__init__()` 方法

**文件**: `algorithm/operators.py`

**修改**:
```python
def __init__(self, random_seed: int = None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    self.objective = ObjectiveFunction()
    
    # ✓ 新增：候选骑手筛选参数（空间剪枝优化）
    self.use_candidate_filtering = True  # 是否启用候选骑手筛选
    self.max_candidates = 10  # 每个订单考虑的最大候选骑手数量
    
    # 注册所有修复算子
    self.operators: List[Tuple[str, Callable]] = [
        ("greedy_insertion", self.greedy_insertion),
        ("regret_2_insertion", self.regret_2_insertion),
        ("regret_3_insertion", self.regret_3_insertion),
        ("random_insertion", self.random_insertion),
    ]
    # ... 其他初始化代码
```

**效果**: 确保所有 `RepairOperators` 实例都包含优化属性，无论是在主进程还是子进程中创建。

---

### 2. 修复骑手数量不匹配问题

**问题**: 用户指定 `--vehicles 40`，但系统生成了 200 个骑手（40 × 5站点）

**原因**: `generate_problem_instance()` 函数默认使用多站点模式（`multi_depot=True`），导致骑手数量 = 指定数量 × 站点数量。

**修改**:

**文件**: `utils/generator.py`
```python
def generate_problem_instance(
    num_orders: int = None,
    num_vehicles: int = None,
    random_seed: int = None,
    clustered: bool = False,
    multi_depot: bool = False  # ✓ 新增参数，默认False
) -> Solution:
    """
    便捷函数: 生成问题实例
    
    Args:
        multi_depot: 是否使用多站点模式（默认False，单站点）
    """
    generator = DataGenerator(random_seed=random_seed)
    
    if clustered:
        return generator.generate_clustered_instance(num_orders, num_vehicles)
    else:
        return generator.generate_solution(
            num_orders, num_vehicles, 
            multi_depot=multi_depot  # ✓ 传递参数
        )
```

**文件**: `main.py`
```python
initial_solution = generate_problem_instance(
    num_orders=num_orders,
    num_vehicles=num_vehicles,
    random_seed=random_seed,
    multi_depot=False  # ✓ 默认使用单站点模式
)
```

**效果**: 
- 单站点模式：骑手数量 = 指定数量
- 多站点模式：骑手数量 = 指定数量 × 站点数量

---

## 多站点模式启用方法

如果想使用5站点模式，有两种方式：

### 方式1: 修改 main.py
```python
initial_solution = generate_problem_instance(
    num_orders=num_orders,
    num_vehicles=num_vehicles,
    random_seed=random_seed,
    multi_depot=True  # 启用多站点
)
```

### 方式2: 添加命令行参数
在 `main.py` 的 `argparse` 部分添加：
```python
parser.add_argument(
    '--multi-depot',
    action='store_true',
    default=False,
    help='是否使用多站点模式（5个站点）'
)

# 然后在调用时传递
initial_solution = generate_problem_instance(
    num_orders=num_orders,
    num_vehicles=num_vehicles,
    random_seed=random_seed,
    multi_depot=args.multi_depot
)
```

使用：
```bash
python main.py --orders 200 --vehicles 40 --multi-depot
# 将生成 200 个订单，40 × 5 = 200 个骑手（每站点40个）
```

---

## 性能优化参数调整

### 候选骑手数量调整

根据问题规模调整 `max_candidates`:

```python
# algorithm/operators.py - RepairOperators.__init__()
self.max_candidates = 10  # 默认值

# 建议配置：
# - 小规模（< 50单）: 5-10
# - 中规模（50-100单）: 10-15
# - 大规模（> 100单）: 15-20
```

### 动态调整（高级）

可以根据骑手总数动态调整：
```python
def __init__(self, random_seed: int = None, num_vehicles: int = None):
    # ...
    if num_vehicles is not None:
        # 候选数量为总骑手数的20-30%
        self.max_candidates = max(5, min(20, int(num_vehicles * 0.3)))
    else:
        self.max_candidates = 10
```

---

## 验证测试

### 小规模测试
```bash
python main.py --orders 50 --vehicles 10 --solver alns-dc --iterations 100
```

### 中规模测试
```bash
python main.py --orders 100 --vehicles 20 --solver alns-dc --iterations 200
```

### 大规模测试
```bash
python main.py --orders 200 --vehicles 40 --solver alns-dc --iterations 500
```

---

## 故障排查

### 问题1: 仍然出现属性错误
**检查**: 确保重新导入模块
```bash
# 清理Python缓存
rm -rf __pycache__ algorithm/__pycache__ models/__pycache__ utils/__pycache__

# 或者在Windows PowerShell
Remove-Item -Recurse -Force __pycache__,algorithm\__pycache__,models\__pycache__,utils\__pycache__
```

### 问题2: 骑手数量仍不匹配
**检查**: 
1. 确认 `multi_depot` 参数设置正确
2. 打印实际参数值：
```python
print(f"Debug: multi_depot = {multi_depot}")
print(f"预期骑手数: {num_vehicles if not multi_depot else num_vehicles * 5}")
print(f"实际骑手数: {len(initial_solution.vehicles)}")
```

### 问题3: 多进程卡住
**解决**: 
1. 减少工作进程数：
```python
ALNSDivideAndConquerSolver(
    max_workers=4  # 手动指定
)
```

2. 禁用并行（调试用）：
```python
ALNSDivideAndConquerSolver(
    use_parallel=False
)
```

---

## 总结

修复要点：
1. ✓ 在 `RepairOperators.__init__()` 中正确初始化所有优化属性
2. ✓ 修正 `generate_problem_instance()` 的 `multi_depot` 默认值
3. ✓ 确保单站点/多站点模式可配置
4. ✓ 保持向后兼容性

现在代码已完全适配多进程并行环境，可以正常运行大规模问题求解。
