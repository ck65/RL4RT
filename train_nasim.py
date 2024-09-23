import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gym
import nasim
import wandb

from agents.DQNAgent import DQNAgent
from agents.DueDQNAgent import DueDQNAgent
from agents.DoubleDQNAgent import DoubleDQNAgent
from agents.D3QNAgent import D3QNAgent
from agents.PER_D3QNAgent import PERD3QNAgent
from agents.PPOAgent import PPOAgent
from utils.config import GetHyperparameters

HyperparametersFile = "./hyperparameters.yaml"



def train(name, DQNParams):
    env = nasim.make_benchmark("tiny")
    # env = gym.make("CartPole-v0")
    wandb.init(project="NASimEmu",name="DQN-tiny", config=DQNParams)
    print(f"====DQN===={name}====")
    dqn_agent = DQNAgent(env = env,
                         name=name,
                         learning_rate=DQNParams["lr"],
                         buffer_size=DQNParams["buffer_size"],
                         batch_size=DQNParams["batch_size"],
                         gamma=DQNParams["gamma"],
                         target_update_freq=DQNParams["target_update_freq"],
                         exploration_fraction=DQNParams["exploration_fraction"],
                         exploration_initial_eps=DQNParams["exploration_initial_eps"],
                         exploration_final_eps=DQNParams["exploration_final_eps"],
                         experiment_dir=DQNParams["experiment_dir"],
                         verbose=1,
                         layers=DQNParams["layers"]
                         )
    dqn_agent.train(episodes=DQNParams["episodes"])
    wandb.finish()
    # print("====DoubleDQN===={name}====")
    # env = nasim.make_benchmark("tiny")
    # wandb.init(project="NASimEmu", name="DoubleDQN-tiny", config=DQNParams)
    # dqn_agent = DoubleDQNAgent(env = env,
    #                           name=name,
    #                      learning_rate=DQNParams["lr"],
    #                      buffer_size=DQNParams["buffer_size"],
    #                      batch_size=DQNParams["batch_size"],
    #                      gamma=DQNParams["gamma"],
    #                      target_update_freq=DQNParams["target_update_freq"],
    #                      exploration_fraction=DQNParams["exploration_fraction"],
    #                      exploration_initial_eps=DQNParams["exploration_initial_eps"],
    #                      exploration_final_eps=DQNParams["exploration_final_eps"],
    #                      experiment_dir=DQNParams["experiment_dir"],
    #                      verbose=1,
    #                      layers=DQNParams["layers"]
    #                            )
    # dqn_agent.train(episodes=DQNParams["episodes"])
    # wandb.finish()
    # print("====DueDQN===={name}====")
    # env = nasim.make_benchmark("tiny")
    # wandb.init(project="NASimEmu", name="DueDQN-tiny", config=DQNParams)
    # dqn_agent = DueDQNAgent(env = env,
    #                           name=name,
    #                      learning_rate=DQNParams["lr"],
    #                      buffer_size=DQNParams["buffer_size"],
    #                      batch_size=DQNParams["batch_size"],
    #                      gamma=DQNParams["gamma"],
    #                      target_update_freq=DQNParams["target_update_freq"],
    #                      exploration_fraction=DQNParams["exploration_fraction"],
    #                      exploration_initial_eps=DQNParams["exploration_initial_eps"],
    #                      exploration_final_eps=DQNParams["exploration_final_eps"],
    #                      experiment_dir=DQNParams["experiment_dir"],
    #                      verbose=1,
    #                      layers=DQNParams["layers"]
    #                         )
    # dqn_agent.train(episodes=DQNParams["episodes"])
    # wandb.finish()
    # print("====D3QN===={name}====")
    # env = nasim.make_benchmark("tiny")
    # wandb.init(project="NASimEmu", name="D3DQN-tiny", config=DQNParams)
    # dqn_agent = D3QNAgent(env = env,
    #                       name=name,
    #                      learning_rate=DQNParams["lr"],
    #                      buffer_size=DQNParams["buffer_size"],
    #                      batch_size=DQNParams["batch_size"],
    #                      gamma=DQNParams["gamma"],
    #                      target_update_freq=DQNParams["target_update_freq"],
    #                      exploration_fraction=DQNParams["exploration_fraction"],
    #                      exploration_initial_eps=DQNParams["exploration_initial_eps"],
    #                      exploration_final_eps=DQNParams["exploration_final_eps"],
    #                      experiment_dir=DQNParams["experiment_dir"],
    #                      verbose=1,
    #                      layers=DQNParams["layers"]
    #                       )
    # dqn_agent.train(episodes=DQNParams["episodes"])
    # wandb.finish()
    # print("====PERD3QN===={name}====")
    # env = nasim.make_benchmark("tiny")
    # wandb.init(project="NASimEmu", name="PERD3DQN-tiny", config=DQNParams)
    # dqn_agent = PERD3QNAgent(env = env,
    #                           name=name,
    #                      learning_rate=DQNParams["lr"],
    #                      buffer_size=DQNParams["buffer_size"],
    #                      batch_size=DQNParams["batch_size"],
    #                      gamma=DQNParams["gamma"],
    #                      target_update_freq=DQNParams["target_update_freq"],
    #                      exploration_fraction=DQNParams["exploration_fraction"],
    #                      exploration_initial_eps=DQNParams["exploration_initial_eps"],
    #                      exploration_final_eps=DQNParams["exploration_final_eps"],
    #                      experiment_dir=DQNParams["experiment_dir"],
    #                      verbose=1,
    #                      layers=DQNParams["layers"]
    #                          )
    # dqn_agent.train(episodes=DQNParams["episodes"])
    # wandb.finish()


if __name__ == '__main__':
    try:
        hyperparameters = GetHyperparameters(HyperparametersFile)
        env_names = hyperparameters["envs"]["env_name"]
        DQNParams = hyperparameters["DQN"]
    except Exception as e:
        raise f"Error: {e}"
    for name in env_names:
        print(f"{name} training starting......")
        train(name, DQNParams)
        print(f"{name} training finished!")