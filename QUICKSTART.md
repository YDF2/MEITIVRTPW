# 快速开始指南

## 运行测试

验证所有优化功能：

```bash
python test_optimizations.py
```

## 运行主程序

使用优化后的模型求解问题：

```bash
python main.py
```

## 关键参数配置

编辑 `config.py` 调整参数：

```python
# 问题规模
NUM_ORDERS = 50          # 订单数量（建议20-100）
NUM_VEHICLES = 5         # 每站点骑手数（建议3-10）

# 多站点配置
NUM_DEPOTS = 5           # 站点数量（1或5）
# 如果只想测试单站点模式，在generator.py中设置multi_depot=False

# 优化参数（在operators.py中）
max_candidates = 10      # 候选骑手数（建议5-20）
use_fast_eval = True     # 启用快速增量评估
```

## 性能对比

### 优化前
- 20订单，5骑手：约5-10秒
- 50订单，10骑手：约30-60秒
- 100订单，20骑手：超时（>5分钟）

### 优化后（预期）
- 20订单，5骑手：约0.5-1秒（**10x提速**）
- 50订单，10骑手：约3-5秒（**10x提速**）
- 100订单，20骑手：约15-30秒（**可求解！**）

## 优化亮点

1. **开放式VRP** ✓
   - 骑手不再强制返回配送站
   - 更符合外卖业务实际

2. **多站点模型** ✓
   - 5个固定站点覆盖全区域
   - 每个站点独立运营

3. **空间剪枝** ✓
   - 只考虑最近的K个候选骑手
   - 避免无效遍历

4. **增量评估** ✓
   - 快速距离计算（无深拷贝）
   - 两阶段评估策略

5. **静态缓存** ✓
   - 预计算距离矩阵
   - O(1)距离查询

## 故障排查

### 问题：测试失败

**解决方法**：
```bash
# 检查Python环境
conda activate optimization

# 安装依赖
pip install numpy scipy

# 重新运行测试
python test_optimizations.py
```

### 问题：性能未提升

**解决方法**：
1. 确认启用了优化参数：
   ```python
   # operators.py
   self.use_candidate_filtering = True
   ```

2. 调整候选骑手数：
   ```python
   self.max_candidates = 10  # 尝试5-20之间的值
   ```

## 下一步

1. 运行完整ALNS算法：`python main.py`
2. 查看结果：`data/results/exp_*/`
3. 调整参数并对比性能
4. （可选）启用并行计算进一步提速

详细文档见：[OPTIMIZATION_IMPLEMENTATION.md](OPTIMIZATION_IMPLEMENTATION.md)
