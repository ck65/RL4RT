# Reinforcement_Learning_Algorithm_Collection

# 算法文件
- Logger 设定
- 随机种子的设定
- 环境实例化
- 为计算图创建 placeholder
- 通过 actor_critic 函数传递算法函数作为参数构建actor-critic计算图
- 实例化经验缓存
- 为算法特定的损失函数和诊断建立计算图
- 配置训练参数
- 构建 TF Session 并初始化参数
- 通过 logger 设置模型保存
- 定义运行算法主循环需要的函数（例如核心更新函数，获取动作函数，测试智能体函数等，取决于具体的算法）
- 运行算法主循环
- 让智能体在环境中开始运行
- 根据算法的主要方程式，周期性更新参数
- 记录核心性能指标并保存智能体

# 项目结构
```text
my_rl_project/
|-- agents/
|   |-- dqn_agent.py        # 深度 Q 学习 (DQN) 算法
|   |-- a2c_agent.py        # Advantage Actor-Critic (A2C) 算法
|   |-- ppo_agent.py        # Proximal Policy Optimization (PPO) 算法
|   |-- ddpg_agent.py       # Deep Deterministic Policy Gradient (DDPG) 算法
|   |-- sac_agent.py        # Soft Actor-Critic (SAC) 算法
|   |-- ...                 # 其他算法的实现
|
|-- environments/
|   |-- my_env.py           # 自定义环境实现
|   |-- another_env.py      # 另一个自定义环境实现
|   |-- ...
|
|-- utils/
|   |-- replay_buffer.py    # 经验回放缓存实现
|   |-- exploration.py      # 探索策略实现
|   |-- neural_networks.py  # 神经网络模型实现
|   |-- ...
|
|-- train.py                # 训练脚本
|-- evaluate.py             # 评估脚本
|-- hyperparameters.yaml    # 超参数配置文件
|-- requirements.txt        # 项目依赖
|-- README.md               # 项目说明文档
```