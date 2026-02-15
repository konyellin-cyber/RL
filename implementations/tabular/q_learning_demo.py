"""
Q-Learning 算法示例：解决 FrozenLake 环境

FrozenLake 是一个简单的网格世界环境，智能体需要从起点到达目标点，
同时避免掉入冰窟窿。这是学习表格型强化学习算法的经典入门案例。
"""

import numpy as np
import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt


class QLearningAgent:
    """Q-Learning 智能体"""
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        初始化 Q-Learning 智能体
        
        参数:
            env: Gym 环境
            learning_rate: 学习率 (alpha)
            discount_factor: 折扣因子 (gamma)
            epsilon: 初始探索率
            epsilon_decay: 探索率衰减系数
            epsilon_min: 最小探索率
        """
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q表：使用字典存储，默认值为0
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    
    def choose_action(self, state):
        """
        使用 epsilon-greedy 策略选择动作
        
        参数:
            state: 当前状态
            
        返回:
            action: 选择的动作
        """
        if np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return self.env.action_space.sample()
        else:
            # 利用：选择 Q 值最大的动作
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        更新 Q 值
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        
        if done:
            # 如果是终止状态，没有未来奖励
            target_q = reward
        else:
            # TD目标：即时奖励 + 折扣的未来最大Q值
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
        
        # Q-Learning 更新规则
        self.q_table[state][action] = current_q + self.lr * (target_q - current_q)
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(env, agent, episodes=10000, print_every=1000):
    """
    训练智能体
    
    参数:
        env: 环境
        agent: 智能体
        episodes: 训练回合数
        print_every: 打印间隔
    
    返回:
        rewards_history: 每个回合的奖励历史
    """
    rewards_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 选择动作
            action = agent.choose_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 更新 Q 值
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        # 衰减探索率
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        
        # 打印训练进度
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(rewards_history[-print_every:])
            print(f"Episode {episode + 1}/{episodes}, "
                  f"Avg Reward: {avg_reward:.3f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return rewards_history


def evaluate(env, agent, episodes=100):
    """
    评估智能体性能
    
    参数:
        env: 环境
        agent: 智能体
        episodes: 评估回合数
    
    返回:
        success_rate: 成功率
        avg_reward: 平均奖励
    """
    total_rewards = []
    successes = 0
    
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 使用贪婪策略（不探索）
            action = np.argmax(agent.q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        total_rewards.append(total_reward)
        if total_reward > 0:
            successes += 1
    
    success_rate = successes / episodes
    avg_reward = np.mean(total_rewards)
    
    return success_rate, avg_reward


def plot_training_curve(rewards_history, window=100):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 5))
    
    # 计算移动平均
    moving_avg = np.convolve(rewards_history, 
                             np.ones(window)/window, 
                             mode='valid')
    
    plt.plot(rewards_history, alpha=0.3, label='Episode Reward')
    plt.plot(range(window-1, len(rewards_history)), 
             moving_avg, 
             label=f'{window}-Episode Moving Average')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('q_learning_training.png', dpi=150, bbox_inches='tight')
    print("训练曲线已保存到 q_learning_training.png")


def main():
    """主函数"""
    print("=" * 60)
    print("Q-Learning 算法示例：FrozenLake 环境")
    print("=" * 60)
    
    # 创建环境
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    print(f"\n环境信息:")
    print(f"  状态空间大小: {env.observation_space.n}")
    print(f"  动作空间大小: {env.action_space.n}")
    print(f"  动作: 左(0), 下(1), 右(2), 上(3)\n")
    
    # 创建智能体
    agent = QLearningAgent(
        env=env,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # 训练
    print("开始训练...")
    rewards_history = train(env, agent, episodes=10000, print_every=2000)
    print("\n训练完成！\n")
    
    # 评估
    print("评估智能体性能...")
    success_rate, avg_reward = evaluate(env, agent, episodes=100)
    print(f"  成功率: {success_rate * 100:.1f}%")
    print(f"  平均奖励: {avg_reward:.3f}\n")
    
    # 绘制训练曲线
    plot_training_curve(rewards_history)
    
    # 展示学到的策略（部分Q表）
    print("学到的 Q 表（前10个状态）:")
    for state in range(min(10, env.observation_space.n)):
        q_values = agent.q_table[state]
        best_action = np.argmax(q_values)
        print(f"  状态 {state}: Q值 = {q_values}, 最优动作 = {best_action}")
    
    env.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
