from agents.EPDQNAgent import *
from agents.EPDQNAgent import _create_action_lists

if __name__ == '__main__':
    wandb.init(project="NASimEmu", name="INVDeepQNetwork")
    env = nasim.make_benchmark("tiny", fully_obs=False, flat_actions=False, flat_obs=False)
    # print(env.observation_space.shape)
    # env = nasim.make_benchmark("tiny",fully_obs=False,flat_obs=False)
    _, _, action_list = _create_action_lists(env=env)
    agent = Agent(env, len(action_list))
    agent.train(env)
    # print(agent.train(env, episodes=1000))