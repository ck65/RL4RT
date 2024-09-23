#!/usr/bin/python3.9

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# -*- coding: utf-8 -*-
"""CLI to run the baseline Deep Q-learning and Random agents
   on a sample CyberBattle gym environment and plot the respective
   cummulative rewards in the terminal.

Example usage:

    python -m run --training_episode_count 50  --iteration_count 9000 --rewardplot_width 80  --chain_size=20 --ownership_goal 1.0

"""
import logging
import sys

import asciichartpy
import cyberbattle._env.cyberbattle_env as cyberbattle_env
import cyberbattle.agents.baseline.agent_ddql as ddqla
import cyberbattle.agents.baseline.agent_dql as dqla
import cyberbattle.agents.baseline.agent_dueling_ddql as dueling_ddqla
import cyberbattle.agents.baseline.agent_dueling_dql as dueling_dqla
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.learner as learner
import cyberbattle.agents.baseline.plotting as p
import gym
import torch
from cyberbattle.agents.baseline.agent_wrapper import Verbosity

chain_size = 4
ownership_goal = 0.2
reward_goal = 2180
training_episode_count = 5
iteration_count = 100
run_random_agent = True
eval_episode_count = 3
rewardplot_width = 80

logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")

print(f"torch cuda available={torch.cuda.is_available()}")

cyberbattlechain = gym.make('CyberBattleChain-v0',
                            size=chain_size,
                            attacker_goal=cyberbattle_env.AttackerGoal(
                                own_atleast_percent=ownership_goal,
                                reward=reward_goal))

ep = w.EnvironmentBounds.of_identifiers(
    maximum_total_credentials=22,
    maximum_node_count=22,
    identifiers=cyberbattlechain.identifiers
)

all_runs = []

# Run Deep Q-learning
dqn_learning_run = learner.epsilon_greedy_search(
    cyberbattle_gym_env=cyberbattlechain,
    environment_properties=ep,
    learner=dqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=10,
        batch_size=512,
        learning_rate=0.01),  # torch default is 1e-2
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    render=True,
    # epsilon_multdecay=0.75,  # 0.999,
    epsilon_exponential_decay=5000,  # 10000
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    title="DQL"
)

all_runs.append(dqn_learning_run)


# Run Due Deep Q-learning
ddqn = learner.epsilon_greedy_search(
    cyberbattle_gym_env=cyberbattlechain,
    environment_properties=ep,
    learner=ddqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=10,
        batch_size=512,
        learning_rate=0.01),  # torch default is 1e-2
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    render=True,
    # epsilon_multdecay=0.75,  # 0.999,
    epsilon_exponential_decay=5000,  # 10000
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    title="DDQL"
)

all_runs.append(ddqn)

dueling_dqn = learner.epsilon_greedy_search(
    cyberbattle_gym_env=cyberbattlechain,
    environment_properties=ep,
    learner=dueling_dqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=10,
        batch_size=512,
        learning_rate=0.01),  # torch default is 1e-2
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    render=True,
    # epsilon_multdecay=0.75,  # 0.999,
    epsilon_exponential_decay=5000,  # 10000
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    title="Dueling DQL"
)

all_runs.append(dueling_dqn)

dueling_ddqn = learner.epsilon_greedy_search(
    cyberbattle_gym_env=cyberbattlechain,
    environment_properties=ep,
    learner=dueling_ddqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=10,
        batch_size=512,
        learning_rate=0.01),  # torch default is 1e-2
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    render=True,
    # epsilon_multdecay=0.75,  # 0.999,
    epsilon_exponential_decay=5000,  # 10000
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    title="Dueling DDQL"
)

all_runs.append(dueling_ddqn)

colors = [asciichartpy.red, asciichartpy.green, asciichartpy.yellow, asciichartpy.blue]

if __name__ == '__main__':
    print("Episode duration -- DQN=Red, Random=Green")
    print(asciichartpy.plot(p.episodes_lengths_for_all_runs(all_runs), {'height': 30, 'colors': colors}))

    print("Cumulative rewards -- DQN=Red, DoubleDQN=Green, DuelingDQN=Yellow, DuelingDoubleDQN=Blue")
    c = p.averaged_cummulative_rewards(all_runs, rewardplot_width)
    print(asciichartpy.plot(c, {'height': 10, 'colors': colors}))
