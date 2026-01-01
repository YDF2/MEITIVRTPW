# -*- coding: utf-8 -*-
"""
强化学习求解器 (Reinforcement Learning Solver)
基于Q-learning的PDPTW路径规划算法

参考原Java实现，针对PDPTW问题进行优化：
- 状态空间：订单分配状态 + 骑手当前状态
- 动作空间：将订单插入到骑手路径的某个位置
- 奖励函数：考虑成本、时间窗违反、路径可行性
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import copy
import time

from models.solution import Solution
from models.vehicle import Vehicle
from models.node import Order, Node, NodeType
from algorithm.base_solver import BaseSolver
from algorithm.objective import ObjectiveFunction, check_validity


class ReinforcementLearningSolver(BaseSolver):
    """
    强化学习求解器（Q-learning）
    
    核心思想：
    1. 将PDPTW问题建模为序列决策问题
    2. 每步决策：选择一个未分配订单，插入到某个骑手的路径中
    3. 通过Q-learning学习最优插入策略
    """
    
    def __init__(
        self,
        episodes: int = 500,
        learning_rate: float = 0.5,  # 增大学习率
        discount_factor: float = 0.95,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.99,
        min_epsilon: float = 0.05,
        random_seed: int = 42,
        verbose: bool = True,
        use_greedy_init: bool = True,
        # 新增参数
        use_softmax: bool = True,  # 使用Softmax温度退火替代ε-greedy
        initial_temperature: float = 10.0,  # 增大初始温度
        temperature_decay: float = 0.97,  # 更慢的衰减
        min_temperature: float = 0.5,  # 提高最小温度保持探索
        use_experience_replay: bool = True,  # 经验回放
        replay_buffer_size: int = 1000,  # 回放缓冲区大小
        batch_size: int = 32,  # 批次大小
        diversity_weight: float = 0.5  # 增大多样性奖励权重
    ):
        """
        初始化强化学习求解器
        
        改进的强化学习算法（针对路径配送优化）：
        
        核心改进：
        1. Softmax温度退火：替代ε-greedy，提供更平滑的探索-利用权衡
        2. 经验回放：重用历史经验，提高样本效率
        3. 多样性奖励：鼓励探索不同的解，避免局部最优
        4. 改进的奖励塑形：考虑解的质量差异
        5. 自适应学习率：根据学习进展调整
        
        Args:
            episodes: 训练轮次
            learning_rate: 学习率 α
            discount_factor: 折扣因子 γ
            epsilon: ε-greedy探索率（use_softmax=False时使用）
            epsilon_decay: ε衰减率
            min_epsilon: 最小ε值
            random_seed: 随机种子
            verbose: 是否输出详细信息
            use_greedy_init: 是否使用贪心算法初始化
            use_softmax: 是否使用Softmax温度退火
            initial_temperature: 初始温度（Softmax）
            temperature_decay: 温度衰减率
            min_temperature: 最小温度
            use_experience_replay: 是否使用经验回放
            replay_buffer_size: 经验回放缓冲区大小
            batch_size: 批量学习大小
            diversity_weight: 多样性奖励权重
        """
        super().__init__(random_seed)
        
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.verbose = verbose
        self.use_greedy_init = use_greedy_init
        
        # Softmax温度退火参数
        self.use_softmax = use_softmax
        self.temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        
        # 经验回放
        self.use_experience_replay = use_experience_replay
        self.replay_buffer: List[Tuple] = []  # (state, action, reward, next_state, done)
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        
        # 多样性奖励
        self.diversity_weight = diversity_weight
        self.visited_solutions: List[str] = []  # 记录访问过的解
        
        # Q表：存储状态-动作值
        self.q_table: Dict[Tuple[str, str], float] = {}
        
        # 统计信息
        self.episode_rewards: List[float] = []
        self.episode_costs: List[float] = []
        self.best_episode: int = 0
        self.best_cost = float('inf')
        self.no_improvement_count = 0  # 连续无改进次数
        
        np.random.seed(random_seed)
    
    def solve(self, initial_solution: Solution) -> Solution:
        """
        使用Q-learning求解PDPTW问题
        
        Args:
            initial_solution: 初始解（包含问题实例）
        
        Returns:
            最优解
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("改进的强化学习求解器 (Enhanced Q-Learning)")
            print("=" * 60)
            print(f"订单数量: {len(initial_solution.orders)}")
            print(f"骑手数量: {len(initial_solution.vehicles)}")
            print(f"训练轮次: {self.episodes}")
            print(f"学习率α: {self.learning_rate}")
            print(f"折扣因子γ: {self.discount_factor}")
            if self.use_softmax:
                print(f"探索策略: Softmax温度退火 (T: {self.temperature:.2f} → {self.min_temperature:.2f})")
            else:
                print(f"探索策略: ε-greedy (ε: {self.epsilon:.2f} → {self.min_epsilon:.2f})")
            print(f"经验回放: {'启用' if self.use_experience_replay else '禁用'}")
            print(f"多样性权重: {self.diversity_weight}")
            print("-" * 60)
        
        # 初始化最优解
        best_solution = None
        best_cost = float('inf')
        
        # 使用贪心算法生成初始解作为baseline
        if self.use_greedy_init:
            if self.verbose:
                print("\n[初始化] 使用贪心算法生成初始解...")
            greedy_solution = self._generate_greedy_solution(initial_solution)
            greedy_cost = greedy_solution.calculate_cost()
            if self.verbose:
                print(f"  ✓ 贪心解成本: {greedy_cost:.2f}")
            
            if greedy_cost < best_cost:
                best_solution = greedy_solution
                best_cost = greedy_cost
        
        # Q-learning训练
        start_time = time.time()
        
        for episode in range(self.episodes):
            # 生成一个episode（一次完整的路径构建过程）
            solution, episode_reward = self._run_episode(initial_solution)
            
            # 计算成本
            cost = solution.calculate_cost()
            self.episode_rewards.append(episode_reward)
            self.episode_costs.append(cost)
            
            # 更新最优解
            if cost < best_cost:
                best_cost = cost
                best_solution = solution.copy()
                self.best_episode = episode
                self.best_cost = best_cost
                self.no_improvement_count = 0  # 重置无改进计数
                
                if self.verbose and (episode % 50 == 0 or episode < 10):
                    elapsed = time.time() - start_time
                    unassigned = solution.num_unassigned
                    explore_param = self.temperature if self.use_softmax else self.epsilon
                    param_name = "T" if self.use_softmax else "ε"
                    print(f"Episode {episode:4d}: 新最优解! 成本={cost:.2f}, "
                          f"未分配={unassigned}, {param_name}={explore_param:.4f}, 时间={elapsed:.1f}s")
            else:
                self.no_improvement_count += 1
                
                # 连续无改进时增加探索
                if self.no_improvement_count > 50 and self.use_softmax:
                    self.temperature = min(self.initial_temperature * 1.5, self.temperature * 2.0)
                    if self.verbose and episode % 50 == 0:
                        print(f"  提示: 连续{self.no_improvement_count}轮无改进，大幅增加探索 (T={self.temperature:.2f})")
                
                if self.verbose and episode % 50 == 0:
                    elapsed = time.time() - start_time
                    explore_param = self.temperature if self.use_softmax else self.epsilon
                    param_name = "T" if self.use_softmax else "ε"
                    print(f"Episode {episode:4d}: 成本={cost:.2f}, "
                          f"最优={best_cost:.2f}, {param_name}={explore_param:.4f}")
            
            # 衰减探索参数
            if self.use_softmax:
                self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)
            else:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # 经验回放学习
            if self.use_experience_replay and len(self.replay_buffer) >= self.batch_size:
                self._replay_experience()
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print("\n" + "-" * 60)
            print(f"训练完成! 总时间: {total_time:.2f}秒")
            print(f"最优解: Episode {self.best_episode}, 成本: {best_cost:.2f}")
            print(f"Q表大小: {len(self.q_table)} 个状态-动作对")
            
            # 验证解的合法性
            is_valid, violations = check_validity(best_solution)
            if is_valid:
                print("✓ 解通过所有约束检查")
            else:
                print(f"✗ 发现 {len(violations)} 个约束违反")
            
            print("=" * 60)
        
        return best_solution
    
    def _run_episode(self, initial_solution: Solution) -> Tuple[Solution, float]:
        """
        运行一个episode（一次完整的解构建过程）
        
        优化：使用增量成本计算奖励，而非绝对成本
        
        Returns:
            (solution, total_reward)
        """
        # 创建空解（重新构建骑手，订单从空开始）
        vehicles = [
            Vehicle(
                vehicle_id=v.id,
                capacity=v.capacity,
                speed=v.speed,
                detour_factor=v.detour_factor,
                depot=initial_solution.depot
            )
            for v in initial_solution.vehicles
        ]
        solution = Solution(
            vehicles=vehicles,
            orders=[],  # 开始时没有已分配订单
            depot=initial_solution.depot
        )
        
        # 【修复】保留多站点信息
        if hasattr(initial_solution, 'depots') and initial_solution.depots:
            solution.depots = initial_solution.depots
        
        # 复制所有订单作为待分配订单
        all_orders = list(initial_solution.orders.values())
        unassigned_orders = all_orders.copy()
        
        # 按时间窗排序（优先处理紧急订单）
        unassigned_orders.sort(key=lambda o: o.pickup_node.due_time)
        
        total_reward = 0.0
        
        # 逐个分配订单
        while unassigned_orders:
            # 获取当前状态
            state = self._get_state(solution, unassigned_orders)
            
            # 选择动作（订单+骑手+位置）
            action = self._select_action(solution, unassigned_orders, state)
            
            if action is None:
                # 无法找到可行动作，将剩余订单标记为未分配
                for order in unassigned_orders:
                    if order.id not in solution.unassigned_orders:
                        solution.unassigned_orders.append(order)
                # 给予大的负奖励
                reward = -10000.0 * len(unassigned_orders)
                total_reward += reward
                break
            
            order, vehicle_id, insert_pos = action
            
            # 记录插入前的成本
            vehicle = solution.vehicles[vehicle_id]
            cost_before = vehicle.calculate_distance()
            violation_before = vehicle.calculate_time_violation()
            
            # 执行动作
            success = self._insert_order(vehicle, order, insert_pos)
            
            if success:
                # 计算插入后的成本增量
                cost_after = vehicle.calculate_distance()
                violation_after = vehicle.calculate_time_violation()
                
                # 使用增量作为奖励（负值，成本越低奖励越高）
                cost_increase = cost_after - cost_before
                violation_increase = violation_after - violation_before
                
                # 基础增量奖励
                reward = self._calculate_reward_incremental(
                    cost_increase, violation_increase, vehicle
                )
                
                # 添加多样性奖励（鼓励探索新的解）
                solution_signature = self._get_solution_signature(solution)
                if solution_signature not in self.visited_solutions:
                    reward += 50.0 * self.diversity_weight  # 奖励新解
                    self.visited_solutions.append(solution_signature)
                    # 限制缓存大小
                    if len(self.visited_solutions) > 100:
                        self.visited_solutions.pop(0)
                
                total_reward += reward
                
                # 更新solution状态
                unassigned_orders.remove(order)
                solution.orders[order.id] = order
                if order.id in solution.unassigned_orders:
                    del solution.unassigned_orders[order.id]
                
                # 更新Q值
                next_state = self._get_state(solution, unassigned_orders)
                done = (len(unassigned_orders) == 0)
                
                # 立即更新Q值
                self._update_q_value(state, action, reward, next_state, unassigned_orders)
                
                # 存储经验用于回放
                if self.use_experience_replay:
                    experience = (state, action, reward, next_state, done)
                    self.replay_buffer.append(experience)
                    # 限制缓冲区大小
                    if len(self.replay_buffer) > self.replay_buffer_size:
                        self.replay_buffer.pop(0)
            else:
                # 插入失败，给予负奖励
                reward = -5000.0
                total_reward += reward
                
                # 移除该订单，标记为未分配
                unassigned_orders.remove(order)
                if order.id not in solution.unassigned_orders:
                    solution.unassigned_orders.append(order)
        
        return solution, total_reward
    
    def _get_state(self, solution: Solution, unassigned_orders: List[Order]) -> str:
        """
        获取当前状态的哈希表示
        
        改进的状态表示：
        1. 已分配/未分配订单数量
        2. 每个骑手的负载情况（离散化）
        3. 平均路径长度（离散化）
        """
        assigned_count = len(solution.orders)
        unassigned_count = len(unassigned_orders)
        
        # 计算骑手负载分布（按路径长度离散化）
        load_bins = [0, 0, 0, 0]  # [0-5, 6-10, 11-20, 20+]
        total_distance = 0
        
        for vehicle in solution.vehicles:
            route_len = len(vehicle.route)
            if route_len <= 5:
                load_bins[0] += 1
            elif route_len <= 10:
                load_bins[1] += 1
            elif route_len <= 20:
                load_bins[2] += 1
            else:
                load_bins[3] += 1
            
            total_distance += vehicle.calculate_distance()
        
        # 平均距离（离散化为10的倍数）
        avg_dist = int(total_distance / max(1, len(solution.vehicles)) / 10) * 10
        
        state_str = f"a{assigned_count}_u{unassigned_count}_l{load_bins[0]}{load_bins[1]}{load_bins[2]}{load_bins[3]}_d{avg_dist}"
        return state_str
    
    def _get_action_hash(self, action: Tuple[Order, int, int]) -> str:
        """获取动作的哈希表示"""
        order, vehicle_id, insert_pos = action
        return f"o{order.id}_v{vehicle_id}_p{insert_pos}"
    
    def _select_action(
        self,
        solution: Solution,
        unassigned_orders: List[Order],
        state: str
    ) -> Optional[Tuple[Order, int, int]]:
        """
        选择动作：支持ε-greedy和Softmax两种策略
        
        Softmax优势：
        - 所有动作都有被选中的概率（避免完全忽略某些动作）
        - 温度参数提供更平滑的探索-利用权衡
        - 高Q值动作更可能被选中，但不是绝对
        
        Returns:
            (order, vehicle_id, insert_position) 或 None
        """
        if not unassigned_orders:
            return None
        
        # 生成所有可能的动作
        possible_actions = self._generate_possible_actions(solution, unassigned_orders)
        
        if not possible_actions:
            return None
        
        if self.use_softmax:
            # Softmax温度退火策略
            return self._select_action_softmax(state, possible_actions)
        else:
            # ε-greedy策略
            if np.random.random() < self.epsilon:
                # 探索：随机选择
                return possible_actions[np.random.randint(len(possible_actions))]
            else:
                # 利用：选择Q值最大的动作
                return self._select_best_action(state, possible_actions)
    
    def _select_action_softmax(
        self,
        state: str,
        possible_actions: List[Tuple[Order, int, int]]
    ) -> Tuple[Order, int, int]:
        """
        使用Softmax温度退火选择动作
        
        概率分布: P(a) = exp(Q(s,a)/T) / Σ exp(Q(s,a')/T)
        - T大：接近均匀分布（探索）
        - T小：接近贪心选择（利用）
        """
        # 获取所有动作的Q值
        q_values = []
        for action in possible_actions:
            action_hash = self._get_action_hash(action)
            q_key = (state, action_hash)
            q_value = self.q_table.get(q_key, 0.0)
            q_values.append(q_value)
        
        # Softmax计算概率
        q_values = np.array(q_values)
        # 数值稳定性：减去最大值
        q_values = q_values - np.max(q_values)
        exp_q = np.exp(q_values / self.temperature)
        probabilities = exp_q / np.sum(exp_q)
        
        # 根据概率分布采样
        selected_idx = np.random.choice(len(possible_actions), p=probabilities)
        return possible_actions[selected_idx]
    
    def _select_best_action(
        self,
        state: str,
        possible_actions: List[Tuple[Order, int, int]]
    ) -> Tuple[Order, int, int]:
        """
        选择Q值最大的动作
        """
        best_action = None
        best_q_value = float('-inf')
        
        for action in possible_actions:
            action_hash = self._get_action_hash(action)
            q_key = (state, action_hash)
            q_value = self.q_table.get(q_key, 0.0)
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        return best_action
    
    def _generate_possible_actions(
        self,
        solution: Solution,
        unassigned_orders: List[Order]
    ) -> List[Tuple[Order, int, int]]:
        """
        生成所有可能的动作（带启发式过滤）
        
        优化策略：
        1. 优先考虑时间窗紧迫的订单
        2. 优先考虑负载较轻的骑手
        3. 限制动作空间大小避免组合爆炸
        
        动作 = (order, vehicle_id, insert_position)
        """
        actions = []
        
        # 限制考虑的订单数量（取前3个最紧急的）
        orders_to_consider = unassigned_orders[:min(3, len(unassigned_orders))]
        
        # 找出负载较轻的骑手（前50%）
        vehicle_loads = [(v, len(v.route)) for v in solution.vehicles]
        vehicle_loads.sort(key=lambda x: x[1])
        vehicles_to_consider = [v for v, _ in vehicle_loads[:max(1, len(vehicle_loads) // 2 + 1)]]
        
        for order in orders_to_consider:
            for vehicle in vehicles_to_consider:
                # 只尝试有限的插入位置
                route_len = len(vehicle.route)
                
                if route_len == 0:
                    # 空路径，只有一个位置
                    actions.append((order, vehicle.id, 0))
                elif route_len <= 5:
                    # 短路径，尝试所有位置
                    for pos in range(route_len + 1):
                        actions.append((order, vehicle.id, pos))
                else:
                    # 长路径，只尝试前面、中间、后面几个位置
                    positions = [0, route_len // 2, route_len]
                    for pos in positions:
                        actions.append((order, vehicle.id, pos))
        
        return actions
    
    def _insert_order(self, vehicle: Vehicle, order: Order, position: int) -> bool:
        """
        将订单插入到骑手路径的指定位置
        
        插入规则：
        1. 先插入pickup节点
        2. 再插入delivery节点（在pickup之后）
        
        Returns:
            是否插入成功
        """
        try:
            # 确保position有效
            position = min(position, len(vehicle.route))
            
            # 插入pickup节点
            vehicle.route.insert(position, order.pickup_node)
            
            # 插入delivery节点（在pickup之后）
            delivery_pos = position + 1
            vehicle.route.insert(delivery_pos, order.delivery_node)
            
            # 检查是否违反约束（简单检查）
            if len(vehicle.route) > 50:  # 路径过长
                # 回滚
                vehicle.route.pop(delivery_pos)
                vehicle.route.pop(position)
                return False
            
            return True
        except Exception as e:
            return False
    
    def _calculate_reward_incremental(
        self,
        cost_increase: float,
        violation_increase: float,
        vehicle: Vehicle
    ) -> float:
        """
        计算增量奖励函数（文献优化方法）
        
        关键改进：使用成本增量而非绝对成本
        这样可以正确引导智能体最小化每次插入的成本
        
        奖励设计：
        1. 基础：负的距离增量（成本越低越好）
        2. 时间窗：违反时间窗的重惩罚
        3. 负载均衡：鼓励使用负载较轻的骑手
        4. 成功奖励：成功插入给予基础正奖励
        """
        # 成功插入的基础奖励
        reward = 100.0
        
        # 距离增量惩罚（归一化）
        reward -= cost_increase * 10.0
        
        # 时间窗违反重惩罚
        if violation_increase > 0:
            reward -= violation_increase * 500.0  # 严重惩罚违反时间窗
        
        # 负载均衡奖励（鼓励使用负载轻的骑手）
        route_length = len(vehicle.route)
        if route_length < 10:
            reward += 50.0  # 奖励使用负载轻的骑手
        elif route_length > 25:
            reward -= 200.0  # 惩罚路径过长
        
        return reward
    
    def _calculate_reward(
        self,
        solution: Solution,
        order: Order,
        vehicle: Vehicle
    ) -> float:
        """
        保留旧的奖励函数用于兼容（已弃用，使用_calculate_reward_incremental）
        """
        return self._calculate_reward_incremental(0, 0, vehicle)
    
    def _update_q_value(
        self,
        state: str,
        action: Tuple[Order, int, int],
        reward: float,
        next_state: str,
        unassigned_orders: List[Order]
    ):
        """
        更新Q值（Q-learning更新规则）
        
        Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        
        优化：
        1. 更准确的max_next_q估计
        2. 自适应学习率
        """
        action_hash = self._get_action_hash(action)
        q_key = (state, action_hash)
        
        # 当前Q值
        current_q = self.q_table.get(q_key, 0.0)
        
        # 计算下一状态的最大Q值
        if len(unassigned_orders) > 0:  # 非终止状态
            # 估算下一状态的最大Q值
            max_next_q = self._estimate_max_q(next_state)
        else:  # 终止状态
            # 终止状态没有未来回报
            max_next_q = 0.0
        
        # Q-learning更新
        td_error = reward + self.discount_factor * max_next_q - current_q
        new_q = current_q + self.learning_rate * td_error
        
        self.q_table[q_key] = new_q
    
    def _estimate_max_q(self, state: str) -> float:
        """
        估算某状态下的最大Q值（启发式）
        """
        # 查找所有与该状态相关的Q值
        related_q_values = [
            q_value for (s, a), q_value in self.q_table.items()
            if s == state
        ]
        
        if related_q_values:
            return max(related_q_values)
        else:
            return 0.0
    
    def _get_solution_signature(self, solution: Solution) -> str:
        """
        获取解的签名用于多样性检测
        
        签名包括：
        1. 每个骑手服务的订单集合（排序后）
        2. 总距离（离散化）
        """
        vehicle_orders = []
        for vehicle in solution.vehicles:
            orders = sorted(vehicle.get_order_ids())
            if orders:
                vehicle_orders.append(tuple(orders))
        
        vehicle_orders.sort()  # 排序以保证顺序一致性
        # 计算总距离
        total_dist = sum(v.calculate_distance() for v in solution.vehicles)
        total_dist = int(total_dist / 10) * 10
        
        signature = f"v{len(vehicle_orders)}_d{total_dist}_" + "_".join([str(o) for o in vehicle_orders])
        return signature
    
    def _replay_experience(self):
        """
        经验回放：从缓冲区随机采样批次进行学习
        
        优势：
        1. 打破样本相关性
        2. 重用历史经验
        3. 提高样本效率
        """
        # 随机采样
        batch_size = min(self.batch_size, len(self.replay_buffer))
        batch_indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        
        for idx in batch_indices:
            state, action, reward, next_state, done = self.replay_buffer[idx]
            
            # 重新计算未分配订单数（用于判断是否终止）
            unassigned_orders = [] if done else [None]  # 简化处理
            
            # 更新Q值（使用较小的学习率避免过度拟合旧经验）
            old_lr = self.learning_rate
            self.learning_rate = old_lr * 0.5  # 回放时使用一半学习率
            self._update_q_value(state, action, reward, next_state, unassigned_orders)
            self.learning_rate = old_lr
    
    def _generate_greedy_solution(self, initial_solution: Solution) -> Solution:
        """
        使用贪心算法生成初始解
        
        策略：每次将订单插入到成本增量最小的位置
        """
        # 创建空解
        vehicles = [
            Vehicle(
                vehicle_id=v.id,
                capacity=v.capacity,
                speed=v.speed,
                detour_factor=v.detour_factor,
                depot=initial_solution.depot
            )
            for v in initial_solution.vehicles
        ]
        solution = Solution(
            vehicles=vehicles,
            orders=[],
            depot=initial_solution.depot
        )
        
        # 【修复】保留多站点信息
        if hasattr(initial_solution, 'depots') and initial_solution.depots:
            solution.depots = initial_solution.depots
        
        # 复制订单列表
        all_orders = list(initial_solution.orders.values())
        
        # 按订单的pickup时间窗早期排序
        all_orders.sort(key=lambda o: o.pickup_node.ready_time)
        
        # 贪心插入每个订单
        for order in all_orders:
            best_vehicle = None
            best_position = None
            best_cost_increase = float('inf')
            
            # 尝试每个骑手
            for vehicle in solution.vehicles:
                # 尝试每个位置
                for pos in range(len(vehicle.route) + 1):
                    # 计算插入成本
                    cost_before = vehicle.calculate_distance()
                    
                    # 临时插入
                    vehicle.route.insert(pos, order.pickup_node)
                    vehicle.route.insert(pos + 1, order.delivery_node)
                    
                    cost_after = vehicle.calculate_distance()
                    cost_increase = cost_after - cost_before
                    
                    # 回滚
                    vehicle.route.pop(pos + 1)
                    vehicle.route.pop(pos)
                    
                    # 更新最优插入
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_vehicle = vehicle
                        best_position = pos
            
            # 执行最优插入
            if best_vehicle is not None:
                best_vehicle.route.insert(best_position, order.pickup_node)
                best_vehicle.route.insert(best_position + 1, order.delivery_node)
                solution.orders[order.id] = order
                if order.id in solution.unassigned_orders:
                    del solution.unassigned_orders[order.id]
            else:
                # 无法插入，添加到未分配列表
                if order.id not in solution.unassigned_orders:
                    solution.unassigned_orders[order.id] = order
        
        return solution
    
    def get_statistics(self) -> Dict:
        """获取算法统计信息"""
        return {
            'episodes': self.episodes,
            'best_episode': self.best_episode,
            'q_table_size': len(self.q_table),
            'final_epsilon': self.epsilon,
            'episode_costs': self.episode_costs,
            'episode_rewards': self.episode_rewards
        }
