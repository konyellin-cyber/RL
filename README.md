# 强化学习在生成式推荐系统中的应用 (RL for Generative Recommendation)

> 本项目聚焦于**生成式推荐系统**中强化学习的应用，包含理论基础、算法实现和可运行的开源案例，帮助理解如何用RL优化推荐策略和用户长期留存。

## 🎯 项目目标

- 📖 掌握推荐系统中的RL基础概念（MDP建模、价值函数、策略优化）
- 💻 提供可运行的开源实现（DQN、Actor-Critic、REINFORCE等）
- 🚀 理解生成式推荐的特殊性（大模型embedding、多目标优化、冷启动）
- 🔬 实验对比不同RL算法在推荐场景下的表现

## 📚 学习路线

### 第一阶段：推荐系统的RL基础
- [x] [马尔可夫决策过程（MDP）在推荐中的建模](theory/foundations/02_mdp_detailed.md)
  - 状态：用户画像+浏览历史+上下文
  - 动作：推荐内容+生成策略
  - 奖励：点击/完播/分享/留存
- [x] [贝尔曼方程详解](theory/foundations/03_bellman_equations_detailed.md)
  - 价值函数：用户状态价值 vs 推荐动作价值
  - 优势函数：衡量推荐效果
- [ ] 推荐系统的MDP建模案例
  - 短视频推荐的状态空间设计
  - 多目标奖励函数设计

### 第二阶段：推荐场景的经典RL算法
- [ ] **Q-Learning** for 推荐
  - 表格型Q表：小规模离散推荐
  - 实现：新闻推荐的Q-Learning
- [ ] **Deep Q-Network (DQN)** for 推荐
  - 处理高维状态（用户embedding）
  - 实现：基于DQN的视频推荐
- [ ] **Policy Gradient** 方法
  - REINFORCE算法
  - 直接优化推荐策略

### 第三阶段：生成式推荐的进阶算法
- [ ] **Actor-Critic** 架构
  - A2C/A3C在推荐中的应用
  - 平衡探索与利用
- [ ] **Proximal Policy Optimization (PPO)**
  - 稳定的策略优化
  - 处理推荐系统的延迟奖励
- [ ] **Deep Deterministic Policy Gradient (DDPG)**
  - 连续动作空间（生成参数调优）

### 第四阶段：推荐系统的特殊挑战
- [ ] **多目标强化学习**
  - 同时优化点击率、完播率、留存率
  - Pareto前沿与权衡策略
- [ ] **离线强化学习（Offline RL）**
  - 从历史日志学习（无需在线交互）
  - Batch RL、Conservative Q-Learning
- [ ] **探索与利用（Exploration-Exploitation）**
  - ε-greedy、UCB、Thompson Sampling
  - 避免信息茧房
- [ ] **冷启动问题**
  - 新用户的快速适应
  - Meta-Learning方法

## 📁 项目结构

```
RL/
├── theory/                      # 理论学习笔记
│   ├── foundations/            # 基础理论（MDP、贝尔曼方程）
│   ├── algorithms/             # 算法原理详解
│   └── papers/                 # 推荐系统RL论文笔记
├── implementations/            # 算法实现
│   ├── dqn_recommendation/     # DQN推荐实现
│   ├── policy_gradient/        # 策略梯度方法
│   ├── actor_critic/           # Actor-Critic系列
│   └── offline_rl/             # 离线强化学习
├── environments/               # 推荐系统模拟环境
│   ├── news_env.py            # 新闻推荐环境
│   ├── video_env.py           # 短视频推荐环境
│   └── ecommerce_env.py       # 电商推荐环境
├── experiments/                # 实验记录与结果
├── tools/                      # 工具脚本
│   └── md_to_docx.py          # Markdown转Word工具
└── resources/                  # 学习资源与数据集
```

## 💻 开源实现案例

### 1. 基于DQN的新闻推荐系统
**路径**: `implementations/dqn_recommendation/`
- **场景**: 新闻推荐中的用户序列建模
- **状态**: 用户历史点击的新闻embedding（维度：64）
- **动作**: 从候选池（1000篇文章）中选择推荐
- **奖励**: 点击+1，阅读完成+5，分享+10
- **算法**: DQN + Experience Replay
- **数据集**: MIND (Microsoft News Dataset) 子集

**快速运行**:
```bash
cd implementations/dqn_recommendation
python train.py --episodes 1000 --batch_size 128
python evaluate.py --checkpoint best_model.pth
```

### 2. Actor-Critic短视频推荐
**路径**: `implementations/actor_critic/`
- **场景**: 短视频平台的连续推荐优化
- **状态**: 用户画像 + 最近5个观看历史 + 会话时长
- **动作**: 推荐策略概率分布（10个候选类别）
- **奖励**: 观看时长（秒）+ 完播率 + 互动行为
- **算法**: A2C (Advantage Actor-Critic)
- **数据集**: 模拟环境（基于真实分布）

**快速运行**:
```bash
cd implementations/actor_critic
python train_a2c.py --env video --gamma 0.95
python visualize_policy.py --checkpoint checkpoints/a2c_final.pth
```

### 3. 离线强化学习推荐（Batch RL）
**路径**: `implementations/offline_rl/`
- **场景**: 从历史推荐日志学习最优策略（无需在线交互）
- **状态**: 用户特征 + 上下文
- **动作**: 历史推荐的物品
- **奖励**: 真实用户反馈（点击、购买等）
- **算法**: Conservative Q-Learning (CQL)
- **数据集**: 模拟电商推荐日志

**快速运行**:
```bash
cd implementations/offline_rl
python train_cql.py --dataset logs/ecommerce_logs.pkl
python policy_evaluation.py --method cql --baseline random
```

### 4. 多目标推荐优化
**路径**: `implementations/multi_objective/`
- **场景**: 同时优化点击率、留存率、收入
- **算法**: Multi-Objective DDPG
- **权重调整**: 帕累托前沿探索

**快速运行**:
```bash
cd implementations/multi_objective
python train_mo_ddpg.py --objectives click,retention,revenue
```

## 🛠️ 环境配置

### 依赖安装
```bash
pip install -r requirements.txt
```

### 主要依赖
- Python 3.8+
- PyTorch / TensorFlow
- Gym / Gymnasium
- NumPy
- Matplotlib

## 📖 学习资源

### 推荐书籍
- **《Reinforcement Learning: An Introduction》** - Sutton & Barto（经典RL教材）
- **《Deep Reinforcement Learning》** - Aske Plaat（深度RL入门）
- **《Recommender Systems Handbook》** - Ricci et al.（推荐系统综述）

### 推荐课程
- **David Silver's RL Course** (UCL) - RL基础理论
- **CS285: Deep Reinforcement Learning** (UC Berkeley) - 深度RL
- **RecSys Tutorial: RL for Recommendation** - 推荐系统会议教程

### 重要论文（推荐系统+RL）

#### 基础论文
1. **DRN**: Deep Reinforcement Learning for News Recommendation (2018)
   - 首个将DQN应用于新闻推荐的工作
2. **Deep Reinforcement Learning for Page-wise Recommendations** (RecSys 2018)
   - 整页推荐的RL建模
3. **Top-K Off-Policy Correction for Recommender System** (WSDM 2019)
   - 离线RL在推荐中的应用

#### 进阶论文
4. **SlateQ**: Slate Optimization via Q-Learning (KDD 2019)
   - 同时推荐多个物品的RL方法
5. **Generative Adversarial User Model for RL in Recommendation** (ICML 2019)
   - 用户模拟器构建
6. **Model-Based RL for Sequential Recommendation** (SIGIR 2020)
   - 基于模型的推荐RL

#### 最新进展
7. **Large Language Models for Recommendation with RL** (2023)
   - LLM + RL的生成式推荐
8. **Multi-Objective RL for Long-term User Engagement** (2024)
   - 多目标优化最新方法

### 开源项目参考
- **RecoGym** - 推荐系统的RL环境模拟器
- **RecBole** - 推荐算法库（包含部分RL方法）
- **TorchRL** - Meta的RL库（可用于推荐）
- **RLlib** - Ray的可扩展RL框架

## 🎯 实践项目

### 1. 新闻推荐系统（入门级）
- **目标**: 优化用户的新闻点击率和阅读时长
- **技术**: DQN + Experience Replay
- **数据**: MIND数据集
- **难度**: ⭐⭐☆☆☆

### 2. 短视频推荐（中级）
- **目标**: 最大化用户观看时长和平台留存
- **技术**: Actor-Critic (A2C)
- **挑战**: 连续状态空间、延迟奖励
- **难度**: ⭐⭐⭐☆☆

### 3. 电商推荐（中高级）
- **目标**: 平衡点击率和购买转化率
- **技术**: Offline RL (CQL)
- **挑战**: 从历史日志学习、避免过拟合
- **难度**: ⭐⭐⭐⭐☆

### 4. 多目标推荐优化（高级）
- **目标**: 同时优化点击、留存、收入
- **技术**: Multi-Objective DDPG
- **挑战**: 目标权衡、帕累托前沿
- **难度**: ⭐⭐⭐⭐⭐

### 5. 生成式推荐（前沿）
- **目标**: 结合LLM生成个性化内容并推荐
- **技术**: PPO + 预训练模型
- **挑战**: 大模型调优、prompt工程
- **难度**: ⭐⭐⭐⭐⭐

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone git@github.com:konyellin-cyber/RL.git
cd RL
```

### 2. 安装依赖
```bash
# 安装基础依赖
pip install -r requirements.txt

# （可选）安装GPU版本PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. 运行第一个推荐系统示例
```bash
# 运行DQN新闻推荐
cd implementations/dqn_recommendation
python train.py

# 查看训练结果
tensorboard --logdir logs/
```

### 4. 理论学习路径
```bash
# 建议按顺序阅读
1. theory/foundations/01_basics.md              # RL基础概念
2. theory/foundations/02_mdp_detailed.md        # MDP在推荐中的应用
3. theory/foundations/03_bellman_equations_detailed.md  # 贝尔曼方程详解
```

## 📊 学习进度

**开始日期**: 2026-02-15

**理论部分**:
- [x] MDP基础与推荐系统建模
- [x] 贝尔曼方程详解
- [ ] 策略梯度方法原理
- [ ] Actor-Critic架构设计
- [ ] 离线强化学习理论

**实现部分**:
- [ ] DQN新闻推荐系统
- [ ] A2C短视频推荐
- [ ] Offline RL电商推荐
- [ ] 多目标推荐优化
- [ ] LLM + RL生成式推荐

**实验记录**:
- [ ] 对比不同RL算法在推荐中的表现
- [ ] 多目标权衡实验
- [ ] 探索策略对比（ε-greedy vs UCB vs Thompson Sampling）

## 🤝 贡献

欢迎提出建议和改进！特别欢迎：
- 📝 推荐系统相关的RL论文笔记
- 💻 新的算法实现和优化
- 🐛 Bug修复和代码改进
- 📊 实验结果和性能对比

## 🔗 相关资源

- [RecSys会议论文集](https://recsys.acm.org/)
- [强化学习中文社区](https://www.zhihu.com/topic/19846282)
- [推荐系统实践](https://github.com/topics/recommender-systems)

## 📄 License

MIT License

---

**持续更新中...** 💪

*如有疑问或建议，欢迎提Issue或PR！*
