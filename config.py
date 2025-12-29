# -*- coding: utf-8 -*-
"""
全局配置文件
包含ALNS算法参数、权重系数、惩罚因子等
"""

# ================== 问题规模 ==================
NUM_ORDERS = 20          # 订单数量 (每个订单包含1个取货点+1个送货点)
NUM_VEHICLES = 5         # 骑手数量
GRID_SIZE = 100          # 坐标范围 [0, GRID_SIZE]

# ================== 骑手参数 ==================
VEHICLE_CAPACITY = 10    # 骑手最大载重能力
VEHICLE_SPEED = 1.0      # 骑手速度 (单位距离/单位时间)

# ================== 时间窗参数 ==================
TIME_HORIZON = 480       # 时间跨度 (例如8小时 = 480分钟)
SERVICE_TIME_PICKUP = 3  # 取餐服务时间 (分钟)
SERVICE_TIME_DELIVERY = 2  # 送餐服务时间 (分钟)
TIME_WINDOW_WIDTH = 30   # 时间窗宽度 (分钟)

# ================== 目标函数权重 ==================
WEIGHT_DISTANCE = 1.0    # w1: 距离成本权重
WEIGHT_TIME_PENALTY = 100.0  # w2: 超时惩罚权重 (软时间窗)
WEIGHT_UNASSIGNED = 1000.0   # w3: 未分配订单惩罚权重
WEIGHT_VEHICLE_USAGE = 50.0  # w4: 骑手使用成本权重

# ================== ALNS 算法参数 ==================
MAX_ITERATIONS = 1000    # 最大迭代次数
SEGMENT_SIZE = 100       # 权重更新的段大小

# 模拟退火参数
INITIAL_TEMPERATURE = 100.0  # 初始温度
COOLING_RATE = 0.995         # 冷却系数
MIN_TEMPERATURE = 0.01       # 最低温度

# 破坏算子参数
DESTROY_RATE_MIN = 0.1   # 最小破坏比例
DESTROY_RATE_MAX = 0.4   # 最大破坏比例

# 算子权重更新参数
SIGMA_1 = 33  # 找到新的全局最优解
SIGMA_2 = 9   # 找到比当前解更好的解
SIGMA_3 = 13  # 接受了比当前解差的解
DECAY_RATE = 0.8  # 权重衰减率

# ================== 随机种子 ==================
RANDOM_SEED = 42

# ================== 文件路径 ==================
DATA_DIR = "data"
OUTPUT_DIR = "data/results"
