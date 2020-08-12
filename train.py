import os.sys
import numpy as np
import argparse
import copy
import gym
from gym import wrappers
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from IPython import display
from JSAnimation.IPython_display import display_animation
from IPython.display import HTML

from models.SARSA import SARSAAgent
from models.Q_Learning import QLearningAgent
from models.DQN import DQNAgent
from models.REINFORCE import REINFORCEAgent
from models.Actor_Critic import ActorCriticAgent
from utils.helpers import bins, discretize_state

parser = argparse.ArgumentParser(description="cartpole")
parser.add_argument("--algorithm", "-algo", type=str, help="the name of a model", default="SARSA")
parser.add_argument("--num_episode", type=int, help="the number of episodes for training", default=1200)
parser.add_argument("--penalty", type=int, help="the penalty for ending episodes in the middle", default=10)
parser.add_argument("--num_discretize", type=int, help="the number of state space divisions (for Q table)", default=6)
parser.add_argument("--memory_size", type=int, help="the size of replay buffer", default=50000)
parser.add_argument("--initial_memory_size", type=int, help="# of random transitions to store first in replay buffer", default=500)
parser.add_argument("--use_baseline", action="store_true", help="use RINFORCE with baseline")
parser.add_argument("--suffix", type=str, help="the suffix of a model name")
opt = parser.parse_args()

opt.name = opt.algorithm
if opt.suffix:
    opt.name += "_%s"%opt.suffix
os.makedirs("results", exist_ok=True)
# for log
episode_rewards = []
num_average_epidodes = 10

def define_agent(algorithm, state_space, action_space, env):
    if algorithm=="SARSA":
        agent = SARSAAgent(opt, state_space, action_space)
        opt.is_discrete_state = True
    elif algorithm == "Qlearning":
        agent = QLearningAgent(opt, state_space, action_space)
        opt.is_discrete_state = True
    elif algorithm == "DQN":
        agent = DQNAgent(opt, state_space, action_space, memory_size=opt.memory_size)
        agent.init_replay_buffer(env) # put the data from the first random action in the replay buffer
        opt.is_discrete_state = False
    elif algorithm == "REINFORCE":
        agent = REINFORCEAgent(opt, state_space, action_space, use_baseline=opt.use_baseline)
        opt.is_discrete_state = False
    elif algorithm == "ActorCritic":
        agent = ActorCriticAgent(opt, state_space, action_space)
        opt.is_discrete_state = False
    else:
        raise NotImplementedError()
    return agent
    
def train(opt):
    env = gym.make('CartPole-v0')
    opt.max_steps = env.spec.max_episode_steps  # maximum number of steps in an episode
    agent = define_agent(opt.algorithm, state_space=env.observation_space.shape[0], action_space=env.action_space.n, env=env)
    print("train an agent:", opt.name)
    for episode in range(opt.num_episode):
        observation = env.reset() # env returns observations of continuous values in 4 dimensions
        episode_reward = agent.train_per_one_episode(episode, observation, env)
        episode_rewards.append(episode_reward)
        if episode % 100 == 0:
            print("Episode %d finished | Episode reward %f" % (episode, episode_reward))
                
    # display the moving average of cumulative rewards during the course of the study
    moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes)/num_average_epidodes, mode='valid')
    plt.plot(np.arange(len(moving_average)),moving_average)
    plt.title('%s: average rewards in %d episodes' % (opt.name, num_average_epidodes))
    plt.xlabel('episode')
    plt.ylabel('rewards')
    #plt.show()
    plt.savefig("results/%s_reward.png" % opt.name)
    plt.close()

    env.close()
    return agent

def test(opt, agent):
    # Testing (visualization) of the final policy
    env = gym.make('CartPole-v0')
    frames = []
    for episode in range(5):
        observation = env.reset()
        if opt.is_discrete_state:
            state = discretize_state(observation, opt.num_discretize)
        else:
            state = observation
        frames.append(env.render(mode='rgb_array'))
        done = False
        while not done:
            action = agent.get_greedy_action(state)
            next_observation, reward, done, _ = env.step(action)
            frames.append(env.render(mode='rgb_array'))
            if opt.is_discrete_state:
                state = discretize_state(next_observation, opt.num_discretize)
            else:
                state = next_observation
    env.close()

    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        patch.set_data(frames[i])
        
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    HTML(anim.to_jshtml())

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    anim.save("results/%s_%s.mp4" % (opt.name, 'CartPole-v0'), writer=writer)
    

if __name__ == "__main__":
    trained_agent = train(opt)
    test(opt, trained_agent)