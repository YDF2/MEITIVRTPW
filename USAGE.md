# 外卖配送路径规划系统 - 算法使用指南

## 快速开始

### 1. 安装依赖

```bash
# 基础依赖（必需）
pip install numpy matplotlib
```

### 2. 基本使用

#### 使用 ALNS 算法

```bash
# 演示模式（10订单，3骑手）
python main.py --demo

# 自定义规模
python main.py --orders 20 --vehicles 5

# 更多迭代提高解质量
python main.py --orders 20 --vehicles 5 --iterations 1000
```

### 3. 命令行参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--orders` | 20 | 订单数量 |
| `--vehicles` | 5 | 骑手数量 |
| `--iterations` | 500 | ALNS最大迭代次数 |
| `--seed` | 42 | 随机种子 |
| `--no-save` | False | 不保存结果文件 |
| `--no-viz` | False | 不显示可视化图形 |

### 4. 性能参考

| 问题规模 | ALNS (500迭代) |
|---------|----------------|
| 10 订单 | ~0.8 秒 |
| 20 订单 | ~2 秒 |
| 50 订单 | ~10 秒 |
| 100 订单 | ~30 秒 |
| 500 订单 | >300 秒 |

### 5. 输出结果

运行后会在 `data/results/exp_YYYYMMDD_HHMMSS/` 目录下生成：

- `solution.json` - 优化后的解
- `problem_instance.json` - 问题实例数据
- `route_visualization.png` - 路径可视化图
- `convergence.png` - 收敛曲线
- `operator_weights.png` - 算子权重分布

### 6. 高级功能

#### 运行基准测试
```bash
python main.py --benchmark
```

#### 测试不同规模
```bash
python test_large_scale.py
```

### 7. 常见问题

**Q: 如何调整优化参数？**
A: 编辑 `config.py` 文件，修改：
- 权重系数（WEIGHT_DISTANCE, WEIGHT_TIME_PENALTY 等）
- 时间窗宽度（TIME_WINDOW_WIDTH）
- 骑手容量（VEHICLE_CAPACITY）

**Q: 如何提高解的质量？**
A: 增加迭代次数 `--iterations 2000`

## 示例

```bash
# 小规模快速测试
python main.py --orders 10 --vehicles 3 --iterations 100

# 中等规模
python main.py --orders 50 --vehicles 10

# 大规模问题
python main.py --orders 100 --vehicles 20 --iterations 1000 --no-viz

# 测试不同规模
python test_large_scale.py
```
