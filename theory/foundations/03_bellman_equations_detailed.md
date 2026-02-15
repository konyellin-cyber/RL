# 贝尔曼方程详解

> **作者**: konyellin-cyber  
> **创建时间**: 2026-02-15  
> **文档目标**: 深入理解贝尔曼方程的数学原理、推导过程和实际应用

---

## 目录

1. [引言](#1-引言)
2. [基础概念回顾](#2-基础概念回顾)
3. [贝尔曼期望方程](#3-贝尔曼期望方程)
4. [贝尔曼最优方程](#4-贝尔曼最优方程)
5. [方程的几何解释](#5-方程的几何解释)
6. [算法应用](#6-算法应用)
7. [代码实现](#7-代码实现)
8. [实例分析](#8-实例分析)
9. [常见问题](#9-常见问题)
10. [进阶主题](#10-进阶主题)
11. [总结](#11-总结)

---

## 1. 引言

### 1.1 什么是贝尔曼方程？

**贝尔曼方程（Bellman Equations）** 是强化学习的核心数学工具，由理查德·贝尔曼（Richard Bellman）在 1950 年代提出。它建立了当前状态价值与后继状态价值之间的递归关系。

### 1.2 为什么重要？

贝尔曼方程是强化学习算法的理论基础：

- **动态规划**：策略迭代、价值迭代
- **蒙特卡洛方法**：MC 预测与控制
- **时序差分学习**：Q-Learning、SARSA、TD(λ)
- **深度强化学习**：DQN、A3C、PPO 等

### 1.3 方程的分类

贝尔曼方程主要分为两类：

| 类型 | 目的 | 应用 |
|------|------|------|
| **期望方程** | 评估给定策略的价值 | 策略评估、预测问题 |
| **最优方程** | 找到最优策略 | 策略改进、控制问题 |

---

## 2. 基础概念回顾

### 2.1 回报（Return）

从时间步 $t$ 开始的**累积折扣奖励**：

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

其中：
- $R_{t+k+1}$ 是在第 $t+k+1$ 步获得的即时奖励
- $\gamma \in [0, 1]$ 是折扣因子

**折扣因子的作用**：
- $\gamma = 0$：只关心即时奖励（短视）
- $\gamma \to 1$：同等重视未来奖励（远视）
- $\gamma < 1$：数学上保证无限序列收敛

### 2.2 价值函数

#### 状态价值函数 $V^\pi(s)$

定义为在状态 $s$ 下遵循策略 $\pi$ 的**期望回报**：

$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s]$$

**直观理解**："在状态 $s$ 下，按照策略 $\pi$ 行动，长期能获得多少奖励？"

#### 动作价值函数 $Q^\pi(s, a)$

定义为在状态 $s$ 下执行动作 $a$，然后遵循策略 $\pi$ 的**期望回报**：

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$$

**直观理解**："在状态 $s$ 下先执行动作 $a$，然后按照策略 $\pi$ 行动，能获得多少奖励？"

### 2.3 价值函数之间的关系

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) Q^\pi(s, a)$$

$$Q^\pi(s, a) = \sum_{s' \in \mathcal{S}} P(s'|s, a) [R(s, a, s') + \gamma V^\pi(s')]$$

---

## 3. 贝尔曼期望方程

### 3.1 状态价值函数的贝尔曼期望方程

#### 方程形式

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} P(s'|s, a) [R(s, a, s') + \gamma V^\pi(s')]$$

#### 数学推导

从定义出发：

$$
\begin{aligned}
V^\pi(s) &= \mathbb{E}_\pi[G_t \mid S_t = s] \\
&= \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s] \\
&= \mathbb{E}_\pi[R_{t+1} \mid S_t = s] + \gamma \mathbb{E}_\pi[G_{t+1} \mid S_t = s]
\end{aligned}
$$

**第一部分**（即时奖励的期望）：

$$\mathbb{E}_\pi[R_{t+1} \mid S_t = s] = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s, a) R(s, a, s')$$

**第二部分**（未来回报的折扣期望）：

$$
\begin{aligned}
\mathbb{E}_\pi[G_{t+1} \mid S_t = s] &= \sum_{a} \pi(a|s) \sum_{s'} P(s'|s, a) \mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s'] \\
&= \sum_{a} \pi(a|s) \sum_{s'} P(s'|s, a) V^\pi(s')
\end{aligned}
$$

**合并两部分**：

$$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V^\pi(s')]$$

#### 紧凑形式（矩阵表示）

$$\mathbf{V}^\pi = \mathbf{R}^\pi + \gamma \mathbf{P}^\pi \mathbf{V}^\pi$$

其中：
- $\mathbf{V}^\pi \in \mathbb{R}^{|S|}$ 是价值向量
- $\mathbf{R}^\pi \in \mathbb{R}^{|S|}$ 是期望奖励向量
- $\mathbf{P}^\pi \in \mathbb{R}^{|S| \times |S|}$ 是状态转移矩阵

**闭式解**：

$$\mathbf{V}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{R}^\pi$$

> ⚠️ **注意**：直接求逆的计算复杂度为 $O(|S|^3)$，不适用于大规模问题。

### 3.2 动作价值函数的贝尔曼期望方程

#### 方程形式

$$Q^\pi(s, a) = \sum_{s' \in \mathcal{S}} P(s'|s, a) [R(s, a, s') + \gamma \sum_{a' \in \mathcal{A}} \pi(a'|s') Q^\pi(s', a')]$$

#### 简化推导

$$
\begin{aligned}
Q^\pi(s, a) &= \mathbb{E}[R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a] \\
&= \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V^\pi(s')] \\
&= \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')]
\end{aligned}
$$

### 3.3 备份图（Backup Diagram）

状态价值函数的备份过程：

```
    s (当前状态)
    |
    +-- π(a|s) --> a (选择动作)
                   |
                   +-- P(s'|s,a) --> s' (下一状态)
                                     |
                                     R + γV(s')
```

**物理意义**：
- 从当前状态 $s$ 开始
- 按照策略 $\pi$ 选择动作
- 环境根据转移概率跳转到 $s'$
- 获得即时奖励 + 未来价值的折扣

---

## 4. 贝尔曼最优方程

### 4.1 最优价值函数的定义

#### 最优状态价值函数

$$
V^{\ast}(s) = \max_\pi V^\pi(s), \quad \forall s \in \mathcal{S}
$$

**含义**：所有可能策略中，状态 $s$ 能达到的最大价值。

#### 最优动作价值函数

$$
Q^{\ast}(s, a) = \max_\pi Q^\pi(s, a), \quad \forall s \in \mathcal{S}, a \in \mathcal{A}
$$

**含义**：所有可能策略中，在状态 $s$ 执行动作 $a$ 能达到的最大价值。

### 4.2 状态价值的贝尔曼最优方程

#### 方程形式

**形式1（完整展开）**：

$$
V^{\ast}(s) = \max_{a \in \mathcal{A}} \left[ \sum_{s' \in \mathcal{S}} P(s'|s, a) \left( R(s, a, s') + \gamma V^{\ast}(s') \right) \right]
$$

**形式2（使用Q函数）**：

$$
V^{\ast}(s) = \max_{a \in \mathcal{A}} Q^{\ast}(s, a)
$$

#### 推导过程

最优策略下的价值就是选择最优动作的价值：

$$
\begin{aligned}
V^{\ast}(s) &= \max_\pi V^\pi(s) \\[0.5em]
&= \max_a \mathbb{E}\left[R_{t+1} + \gamma V^{\ast}(S_{t+1}) \mid S_t = s, A_t = a\right] \\[0.5em]
&= \max_a \sum_{s'} P(s'|s, a) \left[R(s, a, s') + \gamma V^{\ast}(s')\right]
\end{aligned}
$$

**关键区别**：
- 期望方程：对动作求**加权平均**（$\sum_a \pi(a|s)$）
- 最优方程：对动作求**最大值**（$\max_a$）

### 4.3 动作价值的贝尔曼最优方程

#### 方程形式

$$
\begin{aligned}
Q^{\ast}(s, a) = \sum_{s' \in \mathcal{S}} P(s'|s, a) \left[ R(s, a, s') + \gamma \max_{a' \in \mathcal{A}} Q^{\ast}(s', a') \right]
\end{aligned}
$$

#### 推导

$$
\begin{aligned}
Q^{\ast}(s, a) &= \mathbb{E}\left[R_{t+1} + \gamma V^{\ast}(S_{t+1}) \mid S_t = s, A_t = a\right] \\[0.5em]
&= \sum_{s'} P(s'|s, a) \left[R(s, a, s') + \gamma \max_{a'} Q^{\ast}(s', a')\right]
\end{aligned}
$$

### 4.4 最优策略的提取

一旦得到 $V^{\ast}$ 或 $Q^{\ast}$，就可以提取最优策略：

#### 从 $V^{\ast}$ 提取

$$
\begin{aligned}
\pi^{\ast}(s) = \arg\max_{a \in \mathcal{A}} \sum_{s'} P(s'|s, a) \left[R(s, a, s') + \gamma V^{\ast}(s')\right]
\end{aligned}
$$

> ⚠️ **需要知道模型** $P$ 和 $R$

#### 从 $Q^{\ast}$ 提取（无需模型）

$$
\pi^{\ast}(s) = \arg\max_{a \in \mathcal{A}} Q^{\ast}(s, a)
$$

> ✅ **不需要环境模型**，这是 Q-Learning 的核心优势

---

## 5. 方程的几何解释

### 5.1 压缩映射视角

贝尔曼期望方程可以看作**算子方程**：

$$V = T^\pi V$$

其中 $T^\pi$ 是贝尔曼期望算子：

$$
\begin{aligned}
(T^\pi V)(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s, a) \left[R(s, a, s') + \gamma V(s')\right]
\end{aligned}
$$

**定理（Banach 不动点定理）**：
- $T^\pi$ 是关于 $\sup$ 范数的**压缩映射**（压缩系数为 $\gamma$）
- 存在**唯一不动点** $V^\pi$，使得 $T^\pi V^\pi = V^\pi$
- 重复应用算子将**收敛**到不动点

### 5.2 迭代过程的可视化

考虑简单的两状态系统：

```
状态空间：S = {s1, s2}
初始价值：V₀ = [0, 0]

第 1 次迭代：V₁ = T^π V₀
第 2 次迭代：V₂ = T^π V₁
...
第 k 次迭代：Vₖ = T^π Vₖ₋₁

收敛：Vₖ → V^π
```

**收敛速度**：

$$\|V_{k+1} - V^\pi\| \leq \gamma \|V_k - V^\pi\|$$

每次迭代误差减少因子 $\gamma$。

### 5.3 最优方程的非线性

贝尔曼最优方程包含 $\max$ 操作符，是**非线性**的：

$$V = T^* V$$

其中 $T^*$ 是贝尔曼最优算子：

$$(T^* V)(s) = \max_a \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V(s')]$$

**性质**：
- 同样是压缩映射
- 存在唯一不动点 $V^{\ast}$
- 但不是线性系统，无法直接求逆

---

## 6. 算法应用

### 6.1 策略迭代（Policy Iteration）

使用贝尔曼期望方程进行**策略评估**：

**算法流程**：

```
初始化：π₀ 为任意策略
循环：
    1. 策略评估：解 V^πₖ 从 V = T^πₖ V
    2. 策略改进：πₖ₊₁(s) = argmax_a Σ P(s'|s,a)[R + γV^πₖ(s')]
    3. 如果 πₖ₊₁ = πₖ，停止；否则继续
```

**策略评估的求解方法**：
- 迭代法：$V_{i+1} \leftarrow T^\pi V_i$
- 直接求解：$V^\pi = (I - \gamma P^\pi)^{-1} R^\pi$

### 6.2 价值迭代（Value Iteration）

直接使用贝尔曼最优方程迭代：

```
初始化：V₀(s) = 0, ∀s
循环 k = 0, 1, 2, ...：
    Vₖ₊₁(s) = max_a Σ P(s'|s,a)[R(s,a,s') + γVₖ(s')]
直到 max_s |Vₖ₊₁(s) - Vₖ(s)| < ε
```

**提取最优策略**：

$$\pi^{\ast}(s) = \arg\max_a \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V^{\ast}(s')]$$

### 6.3 Q-Learning（无模型）

使用贝尔曼最优方程的**采样版本**：

**更新规则**：

$$Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

**关键点**：
- 不需要知道 $P(s'|s, a)$
- 通过与环境交互采样 $(s, a, r, s')$
- 离策略学习（off-policy）

### 6.4 SARSA（在策略）

使用贝尔曼期望方程的采样版本：

$$Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma Q(s', a') - Q(s, a)]$$

**与 Q-Learning 的区别**：
- Q-Learning：$\max_{a'} Q(s', a')$（最优动作）
- SARSA：$Q(s', a')$（实际执行的动作）

---

## 7. 代码实现

### 7.1 环境定义：GridWorld

```python
import numpy as np

class GridWorld:
    """简单的网格世界环境"""
    
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # 上下左右
        
        # 定义动作
        self.actions = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        
        # 终止状态（左上角和右下角）
        self.terminal_states = [0, self.n_states - 1]
        
    def state_to_coord(self, state):
        """状态索引转坐标"""
        return state // self.size, state % self.size
    
    def coord_to_state(self, row, col):
        """坐标转状态索引"""
        return row * self.size + col
    
    def get_next_state(self, state, action):
        """确定性转移"""
        if state in self.terminal_states:
            return state  # 终止状态不变
        
        row, col = self.state_to_coord(state)
        d_row, d_col = self.actions[action]
        
        new_row = max(0, min(self.size - 1, row + d_row))
        new_col = max(0, min(self.size - 1, col + d_col))
        
        return self.coord_to_state(new_row, new_col)
    
    def get_reward(self, state, action, next_state):
        """奖励函数"""
        if state in self.terminal_states:
            return 0
        return -1  # 每步 -1，鼓励尽快到达终点
```

### 7.2 策略评估（贝尔曼期望方程）

```python
def policy_evaluation(env, policy, gamma=0.9, theta=1e-6):
    """
    策略评估：使用贝尔曼期望方程迭代求解 V^π
    
    参数:
        env: 环境
        policy: 策略 π(a|s)，形状 (n_states, n_actions)
        gamma: 折扣因子
        theta: 收敛阈值
    
    返回:
        V: 状态价值函数
    """
    V = np.zeros(env.n_states)
    
    iteration = 0
    while True:
        delta = 0
        V_old = V.copy()
        
        for s in range(env.n_states):
            if s in env.terminal_states:
                continue
            
            v = 0
            # 贝尔曼期望方程：V(s) = Σ_a π(a|s) Σ_s' P(s'|s,a)[R + γV(s')]
            for a in range(env.n_actions):
                s_next = env.get_next_state(s, a)
                reward = env.get_reward(s, a, s_next)
                
                # 确定性环境：P(s'|s,a) = 1
                v += policy[s, a] * (reward + gamma * V_old[s_next])
            
            V[s] = v
            delta = max(delta, abs(V[s] - V_old[s]))
        
        iteration += 1
        
        if delta < theta:
            print(f"策略评估收敛，迭代 {iteration} 次")
            break
    
    return V
```

### 7.3 价值迭代（贝尔曼最优方程）

```python
def value_iteration(env, gamma=0.9, theta=1e-6):
    """
    价值迭代：使用贝尔曼最优方程求解 V*
    
    参数:
        env: 环境
        gamma: 折扣因子
        theta: 收敛阈值
    
    返回:
        V: 最优状态价值函数
        policy: 最优策略
    """
    V = np.zeros(env.n_states)
    
    iteration = 0
    while True:
        delta = 0
        V_old = V.copy()
        
        for s in range(env.n_states):
            if s in env.terminal_states:
                continue
            
            # 贝尔曼最优方程：V*(s) = max_a Σ_s' P(s'|s,a)[R + γV*(s')]
            action_values = []
            for a in range(env.n_actions):
                s_next = env.get_next_state(s, a)
                reward = env.get_reward(s, a, s_next)
                action_values.append(reward + gamma * V_old[s_next])
            
            V[s] = max(action_values)
            delta = max(delta, abs(V[s] - V_old[s]))
        
        iteration += 1
        
        if delta < theta:
            print(f"价值迭代收敛，迭代 {iteration} 次")
            break
    
    # 提取最优策略
    policy = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        if s in env.terminal_states:
            policy[s] = 0.25  # 终止状态均匀分布
            continue
        
        action_values = []
        for a in range(env.n_actions):
            s_next = env.get_next_state(s, a)
            reward = env.get_reward(s, a, s_next)
            action_values.append(reward + gamma * V[s_next])
        
        best_action = np.argmax(action_values)
        policy[s, best_action] = 1.0  # 确定性最优策略
    
    return V, policy
```

### 7.4 Q-Learning 实现

```python
def q_learning(env, n_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Q-Learning：基于贝尔曼最优方程的无模型学习
    
    参数:
        env: 环境（需支持 step 方法）
        n_episodes: 训练回合数
        alpha: 学习率
        gamma: 折扣因子
        epsilon: ε-greedy 探索率
    
    返回:
        Q: 最优动作价值函数
    """
    Q = np.zeros((env.n_states, env.n_actions))
    
    for episode in range(n_episodes):
        state = np.random.choice([s for s in range(env.n_states) 
                                  if s not in env.terminal_states])
        
        while state not in env.terminal_states:
            # ε-greedy 策略
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[state])
            
            # 执行动作
            next_state = env.get_next_state(state, action)
            reward = env.get_reward(state, action, next_state)
            
            # Q-Learning 更新（贝尔曼最优方程）
            target = reward + gamma * np.max(Q[next_state])
            Q[state, action] += alpha * (target - Q[state, action])
            
            state = next_state
    
    return Q
```

### 7.5 完整示例

```python
# 创建环境
env = GridWorld(size=4)

# 1. 随机策略评估
print("=" * 50)
print("1. 策略评估（随机策略）")
random_policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
V_random = policy_evaluation(env, random_policy)

print("\n随机策略的状态价值：")
print(V_random.reshape(4, 4))

# 2. 价值迭代求最优策略
print("\n" + "=" * 50)
print("2. 价值迭代（求最优策略）")
V_optimal, optimal_policy = value_iteration(env)

print("\n最优状态价值：")
print(V_optimal.reshape(4, 4))

print("\n最优策略（0=上, 1=下, 2=左, 3=右）：")
policy_arrows = np.argmax(optimal_policy, axis=1).reshape(4, 4)
arrow_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
for row in policy_arrows:
    print(' '.join(arrow_map.get(a, '○') for a in row))

# 3. Q-Learning
print("\n" + "=" * 50)
print("3. Q-Learning（无模型学习）")
Q_learned = q_learning(env, n_episodes=5000)

V_learned = np.max(Q_learned, axis=1)
print("\nQ-Learning 学到的价值：")
print(V_learned.reshape(4, 4))

print("\nQ-Learning 学到的策略：")
policy_learned = np.argmax(Q_learned, axis=1).reshape(4, 4)
for row in policy_learned:
    print(' '.join(arrow_map.get(a, '○') for a in row))
```

---

## 8. 实例分析

### 8.1 案例：机器人导航

**场景描述**：

```
起点 S ─→ 路径 ─→ 终点 G
         ↓
       陷阱 T
```

**状态空间**：$S = \{S, A, B, G, T\}$  
**动作空间**：$A = \{\text{前进}, \text{等待}\}$  
**奖励**：
- 到达终点 G：+10
- 掉入陷阱 T：-10
- 其他：-1（时间惩罚）

**转移概率**（随机环境）：

从状态 A 前进：
- 70% 到达 B
- 30% 掉入陷阱 T

### 8.2 手工推导价值

假设 $\gamma = 0.9$，随机策略 $\pi(\text{前进}) = \pi(\text{等待}) = 0.5$

**状态 A 的价值计算**：

$$
\begin{aligned}
V^\pi(A) &= 0.5 \times V_{\text{前进}} + 0.5 \times V_{\text{等待}} \\
V_{\text{前进}} &= 0.7 \times (-1 + 0.9 V^\pi(B)) + 0.3 \times (-1 + 0.9 V^\pi(T)) \\
V_{\text{等待}} &= -1 + 0.9 V^\pi(A)
\end{aligned}
$$

**迭代求解**：

| 迭代 | $V(S)$ | $V(A)$ | $V(B)$ | $V(G)$ | $V(T)$ |
|------|--------|--------|--------|--------|--------|
| 0    | 0      | 0      | 0      | 0      | 0      |
| 1    | -1.0   | -1.0   | -1.0   | 10.0   | -10.0  |
| 2    | -1.9   | -4.15  | 8.1    | 10.0   | -10.0  |
| 3    | -2.74  | -4.89  | 8.19   | 10.0   | -10.0  |
| ...  | ...    | ...    | ...    | ...    | ...    |
| ∞    | -10.5  | -5.2   | 8.3    | 10.0   | -10.0  |

**结论**：状态 A 的期望价值为 -5.2，说明从 A 出发按随机策略不太有利。

### 8.3 最优策略分析

使用贝尔曼最优方程：

$$V^{\ast}(A) = \max\{V_{\text{前进}}, V_{\text{等待}}\}$$

计算得：
- $V_{\text{前进}} = 0.7 \times 8.3 + 0.3 \times (-10) = 2.81$
- $V_{\text{等待}} = -1 + 0.9 \times V^{\ast}(A)$（循环依赖）

**最优决策**：应该选择"前进"，因为期望价值更高。

---

## 9. 常见问题

### 9.1 为什么需要折扣因子 $\gamma$？

**数学原因**：
- 保证无限时间步的回报有界：$\sum_{k=0}^\infty \gamma^k R < \infty$
- 确保贝尔曼算子是压缩映射，保证收敛

**实际意义**：
- 未来的不确定性：远期奖励不如近期确定
- 模拟时间价值：经济学中的折现概念
- 鼓励尽快完成任务

### 9.2 贝尔曼期望方程 vs 最优方程

| 特性 | 期望方程 | 最优方程 |
|------|----------|----------|
| **目的** | 评估给定策略 | 找到最优策略 |
| **形式** | 线性（加权平均） | 非线性（max 操作） |
| **求解** | 线性系统/迭代 | 迭代/动态规划 |
| **收敛** | 单一不动点 | 唯一最优解 |
| **应用** | 策略评估、预测 | 策略改进、控制 |

### 9.3 如何处理连续状态空间？

贝尔曼方程依然成立，但求解方法不同：

**离散化**：
- 将连续空间划分为网格
- 缺点：维度灾难

**函数逼近**：
- 使用参数化函数表示价值：$V(s; \theta)$
- 线性逼近：$V(s) = \phi(s)^T \theta$
- 神经网络：$V(s; \theta) = \text{NN}(s; \theta)$

**示例（DQN）**：

$$\theta \leftarrow \theta + \alpha [r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)] \nabla_\theta Q(s, a; \theta)$$

### 9.4 为什么 Q-Learning 不需要环境模型？

贝尔曼最优方程：

$$Q^{\ast}(s, a) = \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma \max_{a'} Q^{\ast}(s', a')]$$

**关键观察**：
- 右侧是关于转移分布的期望
- 可以用**采样平均**估计期望

**采样版本**：

$$Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a')]$$

其中 $(s, a, r, s')$ 是实际采样的转移。

### 9.5 收敛性保证

**定理（Q-Learning 收敛性）**：

在以下条件下，Q-Learning 以概率 1 收敛到 $Q^{\ast}$：

1. 所有状态-动作对被无限次访问
2. 学习率满足：
   - $\sum_{t=1}^\infty \alpha_t = \infty$（足够大的总更新量）
   - $\sum_{t=1}^\infty \alpha_t^2 < \infty$（更新量逐渐减小）
3. 奖励有界

**典型学习率**：$\alpha_t = \frac{1}{t}$ 或 $\alpha_t = \frac{1}{N_t(s, a)}$

---

## 10. 进阶主题

### 10.1 资格迹与 TD(λ)

贝尔曼方程是 1 步预测，可以扩展到 n 步：

**n 步回报**：

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

**TD(λ) 结合所有 n 步**：

$$G_t^\lambda = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}$$

**资格迹更新**：

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$
$$e_t(s) = \gamma \lambda e_{t-1}(s) + \mathbb{1}(S_t = s)$$
$$V(s) \leftarrow V(s) + \alpha \delta_t e_t(s)$$

### 10.2 双重 Q-Learning

**问题**：Q-Learning 存在**高估偏差**（overestimation bias）

$$\max_{a'} Q(s', a') \geq \mathbb{E}[\max_{a'} Q^{\ast}(s', a')]$$

**解决方案**：使用两个 Q 函数解耦选择与评估

$$Q_1(s, a) \leftarrow Q_1(s, a) + \alpha [r + \gamma Q_2(s', \arg\max_{a'} Q_1(s', a')) - Q_1(s, a)]$$

### 10.3 连续控制中的贝尔曼方程

**确定性策略梯度**（DPG）：

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\pi} [\nabla_a Q^\pi(s, a)|_{a=\pi_\theta(s)} \nabla_\theta \pi_\theta(s)]$$

**DDPG 算法**：
- Actor：学习确定性策略 $\mu(s)$
- Critic：学习 Q 函数，使用贝尔曼方程作为目标

$$L(\phi) = \mathbb{E}[(Q_\phi(s, a) - (r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s'))))^2]$$

### 10.4 分布式贝尔曼方程

**标准贝尔曼方程**只关注期望值，**分布式版本**建模整个回报分布：

$$Z(s, a) \stackrel{D}{=} R(s, a) + \gamma Z(S', A')$$

其中 $Z$ 是随机变量（回报的分布）。

**应用**：C51、QR-DQN、IQN

---

## 11. 总结

### 11.1 核心要点

| 概念 | 期望方程 | 最优方程 |
|------|----------|----------|
| **目标** | 评估策略 $\pi$ | 找最优策略 $\pi^{\ast}$ |
| **V 形式** | $\sum_a \pi(a|s) \sum_{s'} P[\cdots]$ | $\max_a \sum_{s'} P[\cdots]$ |
| **Q 形式** | $\sum_{s'} P[R + \gamma \sum_{a'} \pi(a'|s') Q(s', a')]$ | $\sum_{s'} P[R + \gamma \max_{a'} Q(s', a')]$ |
| **算法** | 策略评估、SARSA | 价值迭代、Q-Learning |
| **收敛** | 线性系统 | 非线性迭代 |

### 11.2 记忆技巧

**贝尔曼方程的通用结构**：

$$\text{当前价值} = \text{即时奖励} + \text{折扣} \times \text{未来价值}$$

**两个关键区别**：
1. **期望 vs 最优**：加权平均 vs 最大值
2. **在策略 vs 离策略**：遵循当前策略 vs 选择最优动作

### 11.3 学习路线图

```
贝尔曼方程
    ↓
├─ 动态规划
│   ├─ 策略迭代
│   └─ 价值迭代
│
├─ 蒙特卡洛
│   └─ 完整回合采样
│
├─ 时序差分
│   ├─ TD(0)
│   ├─ SARSA (在策略)
│   └─ Q-Learning (离策略)
│
└─ 深度强化学习
    ├─ DQN (深度 Q 网络)
    ├─ A3C (演员-评论家)
    └─ PPO (近端策略优化)
```

### 11.4 实践建议

1. **从简单环境开始**：GridWorld、FrozenLake
2. **可视化价值函数**：观察收敛过程
3. **对比不同算法**：理解折衷
4. **调试技巧**：
   - 检查贝尔曼误差：$|V(s) - (r + \gamma V(s'))|$
   - 监控价值函数变化
   - 验证最优策略的合理性

### 11.5 进一步阅读

**经典教材**：
- Sutton & Barto: *Reinforcement Learning: An Introduction* (第 3-4 章)
- Bertsekas: *Dynamic Programming and Optimal Control*

**论文**：
- Watkins (1989): Q-Learning 原始论文
- Mnih et al. (2015): DQN，将贝尔曼方程与深度学习结合

**在线资源**：
- David Silver RL Course: Lecture 3 (Planning by DP)
- OpenAI Spinning Up: 贝尔曼方程基础

---

## 附录：数学符号表

| 符号 | 含义 |
|------|------|
| $s, s'$ | 状态 |
| $a, a'$ | 动作 |
| $r, R$ | 奖励 |
| $\gamma$ | 折扣因子 |
| $\pi$ | 策略 |
| $V^\pi(s)$ | 策略 $\pi$ 下状态 $s$ 的价值 |
| $Q^\pi(s, a)$ | 策略 $\pi$ 下在状态 $s$ 执行动作 $a$ 的价值 |
| $V^{\ast}(s)$ | 最优状态价值 |
| $Q^{\ast}(s, a)$ | 最优动作价值 |
| $P(s'|s, a)$ | 状态转移概率 |
| $\mathbb{E}_\pi[\cdot]$ | 关于策略 $\pi$ 的期望 |
| $T^\pi$ | 贝尔曼期望算子 |
| $T^*$ | 贝尔曼最优算子 |

---

**文档版本**: v1.0  
**最后更新**: 2026-02-15  
**作者**: konyellin-cyber  
**仓库**: https://github.com/konyellin-cyber/RL
