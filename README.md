# FoodDelivery_Optimizer

## 外卖配送路径规划系统 (PDPTW + ALNS)

基于**自适应大邻域搜索 (ALNS)** 算法求解**带时间窗的取送货路径问题 (PDPTW)**。

## 项目结构

```
FoodDelivery_Optimizer/
│
├── config.py             # 全局配置（参数、权重、惩罚系数）
├── main.py               # 程序入口（运行实验、保存结果）
│
├── models/               # 数据模型层
│   ├── __init__.py
│   ├── node.py           # 节点类（商家/顾客/配送站）
│   ├── vehicle.py        # 骑手类
│   └── solution.py       # 解的封装（包含路径、成本计算）
│
├── algorithm/            # 核心算法层
│   ├── __init__.py
│   ├── alns.py           # ALNS主逻辑类
│   ├── operators.py      # 破坏与修复算子 (Destroy & Repair)
│   ├── objective.py      # 目标函数与约束检查
│   └── greedy.py         # 初始解生成器
│
├── utils/                # 工具层
│   ├── __init__.py
│   ├── generator.py      # 随机数据生成器
│   ├── visualizer.py     # 路径可视化绘图
│   └── file_io.py        # JSON数据读写
│
└── data/                 # 存放输入输出数据
    └── results/          # 实验结果
```

## 安装依赖

```bash
pip install numpy matplotlib
```

## 快速开始

### 1. 运行演示模式

```bash
python main.py --demo
```

### 2. 自定义规模运行

```bash
python main.py --orders 20 --vehicles 5 --iterations 500
```

### 3. 运行基准测试

```bash
python main.py --benchmark
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--demo` | 运行演示模式 | - |
| `--benchmark` | 运行基准测试 | - |
| `--orders` | 订单数量 | 20 |
| `--vehicles` | 骑手数量 | 5 |
| `--iterations` | ALNS最大迭代次数 | 500 |
| `--seed` | 随机种子 | 42 |
| `--no-save` | 不保存结果 | False |
| `--no-viz` | 不显示可视化 | False |

## 问题建模

### 数学模型

**目标函数:**

$$\text{Minimize } Z = w_1 \sum_{k \in K} \sum_{(i,j) \in A} d_{ij} x_{ijk} + w_2 \sum_{i \in V} \max(0, T_i - l_i) + w_3 \sum_{k \in K} F_k$$

**核心约束:**

1. **配对约束**: 同一订单的取货点和送货点必须由同一骑手完成
2. **顺序约束**: 必须先取货后送货
3. **容量约束**: 骑手任何时刻的载重不能超过容量上限
4. **时间窗约束**: 软约束,允许超时但有惩罚

## 算法实现

### ALNS (自适应大邻域搜索)

**破坏算子 (Destroy Operators):**
- `random_removal`: 随机移除
- `worst_removal`: 移除成本贡献最差的订单
- `shaw_removal`: 移除相似度高的订单组
- `route_removal`: 移除整条路径的订单

**修复算子 (Repair Operators):**
- `greedy_insertion`: 贪婪插入 (成本最小位置)
- `regret_2_insertion`: Regret-2 插入
- `regret_3_insertion`: Regret-3 插入
- `random_insertion`: 随机可行插入

**接受准则:**
- 模拟退火 (Simulated Annealing)

## 配置参数

在 `config.py` 中可以调整以下参数:

```python
# 目标函数权重
WEIGHT_DISTANCE = 1.0      # 距离成本
WEIGHT_TIME_PENALTY = 100.0  # 超时惩罚
WEIGHT_UNASSIGNED = 1000.0   # 未分配惩罚

# ALNS 参数
MAX_ITERATIONS = 1000
INITIAL_TEMPERATURE = 100.0
COOLING_RATE = 0.995
```

## 输出结果

运行后会生成:
- `solution.json`: 最优解详情
- `route_visualization.png`: 路径可视化图
- `convergence.png`: 收敛曲线图
- `operator_weights.png`: 算子权重分布图

## 示例输出

```
============================================================
   外卖配送路径规划系统 (PDPTW + ALNS)
============================================================
订单数量: 20
骑手数量: 5
最大迭代: 500
------------------------------------------------------------
[步骤4] 优化结果统计
------------------------------------------------------------
  总成本:       856.42
  总行驶距离:   456.42
  时间窗违反:   0.00
  使用骑手数:   4/5
  未分配订单:   0
  解可行性:     是
```

## 作者

FoodDelivery_Optimizer Team

## 许可证

MIT License
