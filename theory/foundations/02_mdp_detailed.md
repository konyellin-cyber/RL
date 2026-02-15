# 马尔可夫决策过程（MDP）详解

## 目录

1. [引言](#1-引言)
2. [马尔可夫性质](#2-马尔可夫性质)
3. [MDP 形式化定义](#3-mdp-形式化定义)
4. [状态转移与奖励](#4-状态转移与奖励)
5. [策略定义](#5-策略定义)
6. [价值函数深入](#6-价值函数深入)
7. [贝尔曼方程推导](#7-贝尔曼方程推导)
8. [最优性与最优策略](#8-最优性与最优策略)
9. [实例分析](#9-实例分析)
10. [代码实现](#10-代码实现)
11. [常见问题](#11-常见问题)

---

## 1. 引言

### 1.1 为什么需要 MDP？

强化学习的核心问题是：**一个智能体如何在不确定的环境中，通过学习做出最优决策？**

马尔可夫决策过程（Markov Decision Process, MDP）为这个问题提供了严格的数学框架：

- 📊 **形式化建模**：将复杂的决策问题转化为数学模型
- 🎯 **最优性分析**：定义什么是"最优"策略
- 🔧 **算法设计**：为开发学习算法提供理论基础

### 1.2 MDP 的应用场景

- 🤖 **机器人导航**：在未知环境中寻找最短路径
- 🎮 **游戏 AI**：学习玩 Atari、围棋、DOTA 等游戏
- 💰 **金融投资**：优化投资组合配置
- 🏥 **医疗决策**：制定最优治疗方案
- 🚗 **自动驾驶**：实时决策控制车辆

---

## 2. 马尔可夫性质

### 2.1 定义

**马尔可夫性质（Markov Property）**：未来只依赖于现在，与过去无关。

#### 数学表述

给定当前状态 $s_t$，未来状态 $s_{t+1}$ 的概率分布与历史状态 $s_0, s_1, ..., s_{t-1}$ 无关：

$$
P(S_{t+1} = s' | S_t = s_t, S_{t-1} = s_{t-1}, ..., S_0 = s_0) = P(S_{t+1} = s' | S_t = s_t)
$$

### 2.2 直观理解

**状态是历史的充分统计量**。当前状态包含了所有做决策所需的信息。

#### 例子：国际象棋

- ✅ **满足马尔可夫性质**：只需知道当前棋盘状态，就能决定下一步
- ❌ **不需要知道**：前面走了哪些步骤（除非涉及特殊规则如三次重复）

#### 例子：扑克牌游戏

- ❌ **不满足马尔可夫性质**：需要记住已经出过哪些牌
- 🔧 **解决方法**：将"已出现的牌"纳入状态定义中

### 2.3 马尔可夫链 vs 马尔可夫决策过程

| 特性 | 马尔可夫链 | 马尔可夫决策过程 |
|------|------------|------------------|
| **状态转移** | 自动发生 | 由动作驱动 |
| **控制** | 无控制 | 智能体可选择动作 |
| **奖励** | 无奖励概念 | 有奖励信号 |
| **目标** | 分析状态分布 | 最大化累积奖励 |

---

## 3. MDP 形式化定义

### 3.1 五元组定义

一个 MDP 由以下五个元素定义：

$$
\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle
$$

#### 详细说明

| 元素 | 符号 | 说明 | 例子（机器人导航） |
|------|------|------|-------------------|
| **状态空间** | $\mathcal{S}$ | 所有可能状态的集合 | 网格中的所有位置 $(x, y)$ |
| **动作空间** | $\mathcal{A}$ | 所有可能动作的集合 | {上, 下, 左, 右} |
| **转移概率** | $\mathcal{P}$ | $P(s' \| s, a)$ 状态转移概率 | 移动成功概率 0.8，滑向其他方向 0.2 |
| **奖励函数** | $\mathcal{R}$ | $R(s, a, s')$ 或 $R(s, a)$ | 到达目标 +100，碰墙 -10，其他 -1 |
| **折扣因子** | $\gamma$ | $0 \leq \gamma \leq 1$ 未来奖励的权重 | $\gamma = 0.9$ |

### 3.2 有限 vs 无限 MDP

#### 有限 MDP
- 状态空间 $|\mathcal{S}| < \infty$
- 动作空间 $|\mathcal{A}| < \infty$
- 可以用表格存储所有值

#### 无限 MDP
- 状态或动作空间无限（如连续状态空间）
- 需要函数逼近（神经网络等）

### 3.3 确定性 vs 随机性 MDP

#### 确定性 MDP
$$
s' = T(s, a) \quad \text{（状态转移函数）}
$$

#### 随机性 MDP
$$
P(s' | s, a) \quad \text{（状态转移概率）}
$$

大多数实际问题是随机性的。

---

## 4. 状态转移与奖励

### 4.1 状态转移概率

给定状态 $s$ 和动作 $a$，转移到状态 $s'$ 的概率：

$$
P(s' | s, a) = P(S_{t+1} = s' | S_t = s, A_t = a)
$$

#### 性质

1. **归一化**：$\sum_{s' \in \mathcal{S}} P(s' | s, a) = 1$
2. **非负性**：$P(s' | s, a) \geq 0$

#### 例子：冰面滑行

在冰面上，机器人想向右移动，但可能滑到其他方向：

```
P(右边 | 当前, 向右) = 0.7   # 成功
P(上面 | 当前, 向右) = 0.1   # 滑偏
P(下面 | 当前, 向右) = 0.1   # 滑偏
P(当前 | 当前, 向右) = 0.1   # 原地不动
```

### 4.2 奖励函数

奖励函数有多种定义方式：

#### 形式 1：依赖三元组 $(s, a, s')$
$$
R(s, a, s') = \text{从状态 } s \text{ 执行动作 } a \text{ 转移到 } s' \text{ 获得的即时奖励}
$$

#### 形式 2：依赖二元组 $(s, a)$
$$
R(s, a) = \mathbb{E}_{s' \sim P(\cdot|s,a)}[R(s, a, s')]
$$

#### 形式 3：只依赖状态 $s$
$$
R(s) = \mathbb{E}_{a, s'}[R(s, a, s')]
$$

**最常用的是形式 2**。

### 4.3 折扣因子 $\gamma$

折扣因子控制对未来奖励的重视程度：

$$
\text{回报} = R_1 + \gamma R_2 + \gamma^2 R_3 + ... = \sum_{t=0}^{\infty} \gamma^t R_{t+1}
$$

#### $\gamma$ 的作用

| $\gamma$ 值 | 含义 | 特点 |
|-------------|------|------|
| $\gamma = 0$ | 只考虑即时奖励 | 短视，贪心 |
| $0 < \gamma < 1$ | 平衡当前与未来 | 最常用 |
| $\gamma = 1$ | 未来与现在等价 | 可能导致无限回报 |

#### 为什么需要折扣？

1. **数学便利**：保证无限时间步的回报有界
2. **不确定性**：未来的奖励不如现在确定
3. **生物学依据**：动物和人类倾向于即时满足

#### 计算例子

假设一个序列的奖励为：$R_1 = 1, R_2 = 2, R_3 = 3, ...$

- $\gamma = 0.9$ 时：$G = 1 + 0.9 \times 2 + 0.81 \times 3 + ... \approx 10$
- $\gamma = 0.5$ 时：$G = 1 + 0.5 \times 2 + 0.25 \times 3 + ... \approx 3.75$

---

## 5. 策略定义

### 5.1 什么是策略？

**策略（Policy）** $\pi$ 是从状态到动作的映射，定义了智能体的行为。

### 5.2 确定性策略

每个状态对应一个确定的动作：

$$
a = \pi(s)
$$

#### 例子
```python
policy = {
    'state_1': 'action_right',
    'state_2': 'action_up',
    'state_3': 'action_left'
}
```

### 5.3 随机策略

每个状态对应一个动作的概率分布：

$$
\pi(a | s) = P(A_t = a | S_t = s)
$$

#### 性质
$$
\sum_{a \in \mathcal{A}} \pi(a | s) = 1, \quad \pi(a | s) \geq 0
$$

#### 例子
```python
policy = {
    'state_1': {'right': 0.7, 'up': 0.2, 'left': 0.1},
    'state_2': {'right': 0.1, 'up': 0.8, 'left': 0.1},
}
```

### 5.4 为什么需要随机策略？

1. **探索**：在学习阶段需要尝试不同动作
2. **最优性**：某些问题的最优策略本身就是随机的（如石头剪刀布）
3. **部分可观测**：状态不完全可知时，随机性可以增强鲁棒性

### 5.5 平稳策略 vs 非平稳策略

- **平稳策略**：$\pi(a|s)$ 不随时间变化
- **非平稳策略**：$\pi_t(a|s)$ 依赖于时间步 $t$

**MDP 理论主要研究平稳策略**，因为可以证明：存在最优平稳策略。

---

## 6. 价值函数深入

### 6.1 回报（Return）

从时间步 $t$ 开始的**累积折扣奖励**：

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

#### 递归形式
$$
G_t = R_{t+1} + \gamma G_{t+1}
$$

### 6.2 状态价值函数 $V^\pi(s)$

在状态 $s$ 下，遵循策略 $\pi$ 的**期望回报**：

$$
V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \bigg| S_t = s \right]
$$

#### 直观理解
- "从状态 $s$ 开始，按策略 $\pi$ 行动，能获得多少总奖励？"
- **越大越好**

### 6.3 动作价值函数 $Q^\pi(s, a)$

在状态 $s$ 执行动作 $a$，然后遵循策略 $\pi$ 的**期望回报**：

$$
Q^\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]
$$

#### 直观理解
- "在状态 $s$ 执行动作 $a$，然后按策略 $\pi$ 行动，能获得多少总奖励？"

### 6.4 $V$ 和 $Q$ 的关系

#### $V$ 用 $Q$ 表示
$$
V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) Q^\pi(s, a)
$$

**含义**：状态价值是所有动作价值按策略概率的加权平均。

#### $Q$ 用 $V$ 表示
$$
Q^\pi(s, a) = \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]
$$

**含义**：动作价值是即时奖励加上下一状态价值的期望。

### 6.5 优势函数 $A^\pi(s, a)$

衡量动作 $a$ 比平均动作好多少：

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

- $A(s, a) > 0$：这个动作比平均好
- $A(s, a) < 0$：这个动作比平均差

在策略梯度算法中很重要。

---

## 7. 贝尔曼方程推导

### 7.1 贝尔曼期望方程（Bellman Expectation Equation）

#### 状态价值函数的贝尔曼期望方程

$$
V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]
$$

**推导过程**：

$$
\begin{align}
V^\pi(s) &= \mathbb{E}_\pi[G_t | S_t = s] \\
&= \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s] \\
&= \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \mathbb{E}_\pi[G_{t+1} | S_{t+1} = s'] \right] \\
&= \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]
\end{align}
$$

#### 动作价值函数的贝尔曼期望方程

$$
Q^\pi(s,a) = \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a') \right]
$$

### 7.2 贝尔曼最优方程（Bellman Optimality Equation）

#### 最优状态价值函数

$$
V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]
$$

**含义**：最优价值是选择最好动作后的期望回报。

#### 最优动作价值函数

$$
Q^*(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \max_{a'} Q^*(s', a') \right]
$$

### 7.3 矩阵形式

对于有限 MDP，可以写成矩阵形式：

$$
V^\pi = R^\pi + \gamma P^\pi V^\pi
$$

解析解：
$$
V^\pi = (I - \gamma P^\pi)^{-1} R^\pi
$$

其中：
- $V^\pi$：状态价值向量（$|\mathcal{S}| \times 1$）
- $R^\pi$：期望奖励向量
- $P^\pi$：状态转移矩阵（$|\mathcal{S}| \times |\mathcal{S}|$）

---

## 8. 最优性与最优策略

### 8.1 策略的偏序关系

策略 $\pi$ 优于策略 $\pi'$（记作 $\pi \geq \pi'$）当且仅当：

$$
V^\pi(s) \geq V^{\pi'}(s), \quad \forall s \in \mathcal{S}
$$

### 8.2 最优策略的存在性

**定理**：对于任何 MDP，存在至少一个最优策略 $\pi^*$，使得：

$$
\pi^* \geq \pi, \quad \forall \pi
$$

**性质**：
1. 所有最优策略有相同的最优状态价值函数 $V^*$
2. 所有最优策略有相同的最优动作价值函数 $Q^*$

### 8.3 从 $Q^*$ 提取最优策略

#### 确定性最优策略

$$
\pi^*(s) = \arg\max_a Q^*(s, a)
$$

#### 随机最优策略

如果多个动作都达到最大值，可以在它们之间均匀分配：

$$
\pi^*(a|s) = \begin{cases}
\frac{1}{|\mathcal{A}^*(s)|} & \text{if } a \in \mathcal{A}^*(s) \\
0 & \text{otherwise}
\end{cases}
$$

其中 $\mathcal{A}^*(s) = \arg\max_a Q^*(s, a)$ 是最优动作集合。

### 8.4 策略改进定理

如果对于所有状态 $s$：

$$
Q^\pi(s, \pi'(s)) \geq V^\pi(s)
$$

则 $\pi' \geq \pi$。

这是策略迭代算法的理论基础。

---

## 9. 实例分析

### 9.1 网格世界（Grid World）

#### 问题设定

- **状态空间**：$4 \times 4$ 网格，共 16 个状态
- **动作空间**：{上, 下, 左, 右}
- **转移**：确定性（除非撞墙则停留）
- **奖励**：
  - 到达终点（右下角）：+10
  - 掉入陷阱（中间某格）：-10
  - 其他：-1（鼓励尽快到达）
- **折扣因子**：$\gamma = 0.9$

#### 状态表示

```
[S] [ ] [ ] [ ]
[ ] [X] [ ] [ ]   S: 起点
[ ] [ ] [ ] [ ]   X: 陷阱
[ ] [ ] [ ] [G]   G: 终点
```

#### 最优策略示例

```
→ → → ↓
↑ X ↓ ↓
↑ ← ← ↓
→ → → G
```

### 9.2 简化示例：2 状态 MDP

#### 问题定义

- **状态**：$\mathcal{S} = \{s_1, s_2\}$
- **动作**：$\mathcal{A} = \{a_1, a_2\}$
- **转移概率**：
  $$
  P(s_1|s_1, a_1) = 0.8, \quad P(s_2|s_1, a_1) = 0.2
  $$
  $$
  P(s_1|s_1, a_2) = 0.3, \quad P(s_2|s_1, a_2) = 0.7
  $$
- **奖励**：
  $$
  R(s_1, a_1) = 5, \quad R(s_1, a_2) = 2
  $$
- **折扣**：$\gamma = 0.9$

#### 价值计算

假设策略 $\pi(a_1|s_1) = 1$（总是选择 $a_1$），计算 $V^\pi(s_1)$：

$$
V^\pi(s_1) = R(s_1, a_1) + \gamma \left[ 0.8 \cdot V^\pi(s_1) + 0.2 \cdot V^\pi(s_2) \right]
$$

需要知道 $V^\pi(s_2)$ 才能求解，通常用迭代方法。

---

## 10. 代码实现

### 10.1 定义简单的 MDP

```python
import numpy as np

class SimpleMDP:
    def __init__(self):
        # 状态：0, 1, 2 (2是终止状态)
        self.n_states = 3
        self.n_actions = 2
        
        # 转移概率 P[s][a] -> [(prob, next_state, reward)]
        self.transitions = {
            0: {
                0: [(0.8, 0, -1), (0.2, 1, -1)],  # 动作0: 大概率停留
                1: [(0.5, 1, -1), (0.5, 0, -1)]   # 动作1: 可能前进
            },
            1: {
                0: [(0.9, 1, -1), (0.1, 2, 10)],  # 动作0: 小概率到终点
                1: [(0.6, 2, 10), (0.4, 0, -1)]   # 动作1: 较大概率到终点
            },
            2: {  # 终止状态
                0: [(1.0, 2, 0)],
                1: [(1.0, 2, 0)]
            }
        }
        
        self.gamma = 0.9  # 折扣因子
    
    def get_transitions(self, state, action):
        """返回 (state, action) 下的所有可能转移"""
        return self.transitions[state][action]
```

### 10.2 策略评估（Policy Evaluation）

计算给定策略的价值函数：

```python
def policy_evaluation(mdp, policy, theta=0.01, max_iter=1000):
    """
    策略评估：迭代计算 V^π(s)
    
    参数:
        mdp: MDP 对象
        policy: 策略 policy[s][a] = π(a|s)
        theta: 收敛阈值
        max_iter: 最大迭代次数
    """
    V = np.zeros(mdp.n_states)
    
    for iteration in range(max_iter):
        delta = 0
        V_new = V.copy()
        
        for s in range(mdp.n_states):
            v = 0
            # 对所有动作求期望
            for a in range(mdp.n_actions):
                prob_action = policy[s][a]  # π(a|s)
                
                # 对所有下一状态求期望
                for prob, s_next, reward in mdp.get_transitions(s, a):
                    v += prob_action * prob * (reward + mdp.gamma * V[s_next])
            
            V_new[s] = v
            delta = max(delta, abs(V[s] - V_new[s]))
        
        V = V_new
        
        if delta < theta:
            print(f"收敛于第 {iteration + 1} 次迭代")
            break
    
    return V

# 示例：均匀随机策略
mdp = SimpleMDP()
random_policy = np.ones((mdp.n_states, mdp.n_actions)) / mdp.n_actions
V = policy_evaluation(mdp, random_policy)
print("状态价值:", V)
```

### 10.3 价值迭代（Value Iteration）

求解最优价值函数：

```python
def value_iteration(mdp, theta=0.01, max_iter=1000):
    """
    价值迭代：求解 V*(s)
    """
    V = np.zeros(mdp.n_states)
    
    for iteration in range(max_iter):
        delta = 0
        V_new = V.copy()
        
        for s in range(mdp.n_states):
            # 对所有动作求最大值
            action_values = []
            
            for a in range(mdp.n_actions):
                q_sa = 0
                for prob, s_next, reward in mdp.get_transitions(s, a):
                    q_sa += prob * (reward + mdp.gamma * V[s_next])
                action_values.append(q_sa)
            
            V_new[s] = max(action_values)
            delta = max(delta, abs(V[s] - V_new[s]))
        
        V = V_new
        
        if delta < theta:
            print(f"收敛于第 {iteration + 1} 次迭代")
            break
    
    return V

# 提取最优策略
def extract_policy(mdp, V):
    """从最优价值函数提取最优策略"""
    policy = np.zeros((mdp.n_states, mdp.n_actions))
    
    for s in range(mdp.n_states):
        action_values = []
        
        for a in range(mdp.n_actions):
            q_sa = 0
            for prob, s_next, reward in mdp.get_transitions(s, a):
                q_sa += prob * (reward + mdp.gamma * V[s_next])
            action_values.append(q_sa)
        
        # 最优动作
        best_action = np.argmax(action_values)
        policy[s][best_action] = 1.0
    
    return policy

V_star = value_iteration(mdp)
pi_star = extract_policy(mdp, V_star)
print("最优价值:", V_star)
print("最优策略:\n", pi_star)
```

### 10.4 完整示例：网格世界

```python
class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # 上下左右
        
        # 特殊状态
        self.goal = (size-1, size-1)
        self.trap = (1, 1)
        
        self.actions = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        
        self.gamma = 0.9
    
    def coord_to_state(self, row, col):
        return row * self.size + col
    
    def state_to_coord(self, state):
        return state // self.size, state % self.size
    
    def step(self, state, action):
        """执行动作，返回 (next_state, reward, done)"""
        row, col = self.state_to_coord(state)
        
        # 终点和陷阱是终止状态
        if (row, col) == self.goal:
            return state, 0, True
        if (row, col) == self.trap:
            return state, 0, True
        
        # 移动
        drow, dcol = self.actions[action]
        new_row, new_col = row + drow, col + dcol
        
        # 碰墙检查
        if not (0 <= new_row < self.size and 0 <= new_col < self.size):
            new_row, new_col = row, col  # 停留原地
        
        new_state = self.coord_to_state(new_row, new_col)
        
        # 计算奖励
        if (new_row, new_col) == self.goal:
            reward = 10
            done = True
        elif (new_row, new_col) == self.trap:
            reward = -10
            done = True
        else:
            reward = -1
            done = False
        
        return new_state, reward, done
    
    def visualize_policy(self, policy):
        """可视化策略"""
        symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        
        for row in range(self.size):
            for col in range(self.size):
                state = self.coord_to_state(row, col)
                
                if (row, col) == self.goal:
                    print('G', end=' ')
                elif (row, col) == self.trap:
                    print('X', end=' ')
                else:
                    action = np.argmax(policy[state])
                    print(symbols[action], end=' ')
            print()

# 运行
env = GridWorld(size=4)
# ... 运行价值迭代并提取策略 ...
# env.visualize_policy(optimal_policy)
```

---

## 11. 常见问题

### Q1: MDP 的假设是否现实？

**马尔可夫性假设** 并不总是成立，但我们可以通过扩展状态定义来满足：

- ❌ 原始问题不满足马尔可夫性
- ✅ 将历史信息编码到状态中

例如：扑克牌游戏，将"已出现的牌"纳入状态。

### Q2: 折扣因子如何选择？

| 场景 | 推荐 $\gamma$ | 原因 |
|------|--------------|------|
| 有明确终点的任务 | 0.9 - 0.99 | 平衡长期与短期 |
| 持续运行的任务 | 0.95 - 0.999 | 更重视长期收益 |
| 金融问题 | 接近 1 | 时间价值 |

### Q3: 如何处理连续状态空间？

MDP 理论适用于连续空间，但实际需要：

- **离散化**：将连续空间划分为有限个区域
- **函数逼近**：用神经网络等表示价值函数（深度强化学习）

### Q4: 最优策略是否唯一？

- **价值函数唯一**：$V^*$ 和 $Q^*$ 是唯一的
- **策略可能不唯一**：多个状态-动作对可能有相同的最大 Q 值

### Q5: MDP vs POMDP

| 特性 | MDP | POMDP |
|------|-----|-------|
| **状态可观测性** | 完全可观测 | 部分可观测 |
| **决策依据** | 当前状态 | 观测历史或信念状态 |
| **复杂度** | 相对简单 | 显著增加 |

POMDP（部分可观测马尔可夫决策过程）用于智能体无法完全观测状态的情况。

---

## 总结

### 核心要点

1. ✅ **MDP 是强化学习的数学基础**
2. ✅ **贝尔曼方程是价值函数的递归定义**
3. ✅ **最优策略可以从最优价值函数提取**
4. ✅ **动态规划方法可以求解有限 MDP**

### 下一步学习

- [ ] 深入学习 **动态规划方法**（策略迭代、价值迭代）
- [ ] 理解 **无模型方法**（Monte Carlo、TD Learning）
- [ ] 实现 **Q-Learning** 和 **SARSA** 算法
- [ ] 探索 **函数逼近**（深度强化学习的基础）

---

## 参考资料

1. **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction* (2nd ed.) - Chapter 3
2. **Bellman, R. (1957).** *Dynamic Programming* - 原始 MDP 理论
3. **David Silver's RL Lecture 2** - MDP 讲解
4. **CMU 10-703** - Deep Reinforcement Learning

---

**文档更新时间：2026-02-15**  
**作者：AI Assistant for RL Study**