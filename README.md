# FoodDelivery_Optimizer

## 外卖配送路径规划系统 (PDPTW + ALNS + 分治策略 + 强化学习)

基于**自适应大邻域搜索 (ALNS)** 算法求解**带时间窗的取送货路径问题 (PDPTW)**。
对于大规模问题（>=100订单），自动启用**分治策略 (Divide and Conquer)** 以提升求解效率。
**新增强化学习求解器（Q-Learning）**，适用于小规模问题的快速求解和算法研究。

## 🎯 支持的求解器

| 求解器 | 说明 | 适用规模 | 特点 |
|--------|------|---------|------|
| **alns** | 标准ALNS算法 | <100订单 | 纯启发式，效率高 |
| **alns-dc** | ALNS分治并行 | >100订单 | 自动聚类+并行求解 |
| **rl** | 强化学习(Q-Learning) | <20订单 | 在线学习，适合研究 |

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
│   ├── greedy.py         # 初始解生成器
│   ├── divide_and_conquer.py  # 分治策略求解器（大规模问题）
│   └── reinforcement_learning.py  # 强化学习求解器（Q-Learning）
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
pip install numpy matplotlib scikit-learn
```

**注意**：`scikit-learn` 用于大规模问题的K-Means聚类。

## 快速开始

### 1. 运行演示模式

```bash
python main.py --demo
```

### 2. 自定义规模运行

```bash
# 使用ALNS算法
python main.py --orders 20 --vehicles 5 --iterations 500

# 使用强化学习算法（推荐小规模）
python main.py --orders 15 --vehicles 4 --solver rl --iterations 100

# 使用ALNS分治（推荐大规模）
python main.py --orders 200 --vehicles 40 --solver alns-dc
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
| `--solver` | 求解器类型 (alns/alns-dc/rl) | 自动选择 |
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

---

## 🤖 强化学习求解器

### Q-Learning算法

本项目实现了基于Q-learning的强化学习求解器，适用于小规模PDPTW问题。

**核心特性：**
- **状态空间**：订单分配状态 + 骑手路径状态
- **动作空间**：(订单ID, 骑手ID, 插入位置)
- **奖励函数**：-(成本) - 100×(时间窗违反) - 10×(路径过长惩罚)
- **学习策略**：ε-greedy（探索与利用平衡）
- **Q值更新**：Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]

### 使用强化学习

```bash
# 运行RL算法
python main.py --orders 15 --vehicles 4 --solver rl --iterations 100

# 运行测试套件
python test_reinforcement_learning.py

# 查看演示
python demo_rl.py
```

### 性能对比

| 问题规模 | RL | ALNS | ALNS-DC | 推荐 |
|---------|-----|------|---------|------|
| <20订单 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | RL/ALNS |
| 20-50订单 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ALNS |
| >50订单 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ALNS-DC |

### 相关文档

- **技术文档**：[REINFORCEMENT_LEARNING_DOC.md](REINFORCEMENT_LEARNING_DOC.md)
- **实现总结**：[RL_IMPLEMENTATION_SUMMARY.md](RL_IMPLEMENTATION_SUMMARY.md)
- **算法代码**：[algorithm/reinforcement_learning.py](algorithm/reinforcement_learning.py)
- **测试脚本**：[test_reinforcement_learning.py](test_reinforcement_learning.py)

---

## 作者

YDF

## 许可证

MIT License

