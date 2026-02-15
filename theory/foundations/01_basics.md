# 强化学习基础理论

## 1. 什么是强化学习？

强化学习（Reinforcement Learning, RL）是机器学习的一个重要分支，研究智能体（Agent）如何在环境（Environment）中通过**试错**的方式学习最优策略，以最大化累积奖励。

### 核心概念

- **智能体（Agent）**：做出决策的主体
- **环境（Environment）**：智能体交互的外部世界
- **状态（State）**：描述环境当前情况的信息
- **动作（Action）**：智能体可以执行的操作
- **奖励（Reward）**：环境对智能体动作的即时反馈
- **策略（Policy）**：从状态到动作的映射

## 2. 马尔可夫决策过程（MDP）

强化学习问题通常被建模为**马尔可夫决策过程（Markov Decision Process, MDP）**。

### MDP 的定义

一个 MDP 由五元组定义：**⟨S, A, P, R, γ⟩**

- **S**：状态空间（State Space）
- **A**：动作空间（Action Space）  
- **P**：状态转移概率 P(s'|s,a)
- **R**：奖励函数 R(s,a,s')
- **γ**：折扣因子（Discount Factor），0 ≤ γ ≤ 1

### 马尔可夫性质

**当前状态包含了所有相关的历史信息**，即：

```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0) = P(s_{t+1} | s_t, a_t)
```

## 3. 回报与价值函数

### 回报（Return）

从时间步 t 开始的**累积折扣奖励**：

```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... = Σ_{k=0}^∞ γ^k R_{t+k+1}
```

### 价值函数

#### 状态价值函数 V(s)

在状态 s 下，遵循策略 π 的期望回报：

```
V^π(s) = E_π[G_t | S_t = s]
```

#### 动作价值函数 Q(s,a)

在状态 s 下执行动作 a，然后遵循策略 π 的期望回报：

```
Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
```

## 4. 贝尔曼方程

### 贝尔曼期望方程

**状态价值函数的递归关系**：

```
V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γV^π(s')]
```

**动作价值函数的递归关系**：

```
Q^π(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ Σ_{a'} π(a'|s') Q^π(s',a')]
```

### 贝尔曼最优方程

**最优状态价值函数**：

```
V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γV*(s')]
```

**最优动作价值函数**：

```
Q*(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ max_{a'} Q*(s',a')]
```

## 5. 策略

### 确定性策略

每个状态对应一个确定的动作：

```
π: S → A
```

### 随机策略

给出每个状态下各动作的概率分布：

```
π(a|s) = P(A_t = a | S_t = s)
```

### 最优策略

使价值函数最大的策略：

```
π* = arg max_π V^π(s), ∀s ∈ S
```

## 6. 探索与利用（Exploration vs Exploitation）

这是强化学习中的核心权衡问题：

- **探索（Exploration）**：尝试新的动作，以发现更好的策略
- **利用（Exploitation）**：选择当前已知的最优动作，以获得最大奖励

### 常用策略

#### ε-greedy 策略

以概率 ε 随机探索，以概率 1-ε 选择最优动作：

```python
if random() < ε:
    action = random_action()
else:
    action = argmax(Q[state])
```

#### Softmax（Boltzmann）策略

使用概率分布选择动作：

```
π(a|s) = exp(Q(s,a)/τ) / Σ_b exp(Q(s,b)/τ)
```

其中 τ 是温度参数。

## 7. 强化学习的分类

### 按模型分类

- **基于模型（Model-Based）**：学习环境模型 P 和 R
- **无模型（Model-Free）**：直接学习价值函数或策略

### 按策略分类

- **在策略（On-Policy）**：学习的是当前执行的策略（如 SARSA）
- **离策略（Off-Policy）**：学习的策略与执行的策略不同（如 Q-Learning）

### 按价值与策略

- **基于价值（Value-Based）**：学习价值函数（如 Q-Learning, DQN）
- **基于策略（Policy-Based）**：直接学习策略（如 Policy Gradient）
- **演员-评论家（Actor-Critic）**：同时学习价值和策略

## 8. 学习方法

### 动态规划（Dynamic Programming）

需要完整的环境模型，通过迭代更新价值函数。

- **策略迭代（Policy Iteration）**
- **价值迭代（Value Iteration）**

### 蒙特卡洛方法（Monte Carlo）

通过完整回合的经验进行学习，无需环境模型。

### 时序差分学习（Temporal Difference）

结合 DP 和 MC 的优点，可以在线学习。

- **TD(0)**
- **SARSA**
- **Q-Learning**

## 9. 关键概念对比

| 概念 | 说明 |
|------|------|
| **同策略 vs 异策略** | 学习策略是否与行为策略相同 |
| **在线 vs 离线** | 是否需要完整回合才能更新 |
| **自举 vs 采样** | 使用估计值更新 vs 使用实际采样 |
| **表格型 vs 函数逼近** | Q表存储 vs 神经网络近似 |

## 10. 下一步学习

- [ ] 深入学习 Q-Learning 算法
- [ ] 实现 SARSA 并对比差异
- [ ] 理解 TD(λ) 和资格迹
- [ ] 学习策略梯度方法

## 参考资料

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)
2. David Silver's RL Course: https://www.davidsilver.uk/teaching/
3. OpenAI Spinning Up: https://spinningup.openai.com/

---

**更新时间：2026-02-15**
