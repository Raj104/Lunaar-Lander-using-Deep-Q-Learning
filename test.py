import torch
import gym
import numpy as np
from train import QNetwork
from gym.utils.save_video import save_video
import matplotlib.pyplot as plt

# Load the trained model
state_size = 8  # Assuming the state size of LunarLander-v2
action_size = 4  # Assuming the action size of LunarLander-v2
q_network = QNetwork(state_size, action_size)
q_network.load_state_dict(torch.load('lunar_lander_dqn.pth'))
q_network.eval()

render_mode = 'human' # 'human' or 'rgb_array_list'
video = False #if render_mode == 'human' else True

# Test the trained model
env = gym.make('LunarLander-v2', render_mode=render_mode)
state = env.reset()
state = state[0]
total_reward = 0
avg = []
count = 0

# maximum number of steps per episode
max_steps = 500

for i in range(100):
    state = env.reset()
    state = state[0]
    total_reward = 0
    time_step = 0

    while True:
        with torch.no_grad():
            q_values = q_network(torch.FloatTensor(state))
            action = q_values.argmax().item()

        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        total_reward += reward

        time_step += 1
        if time_step > max_steps:
            done = True

        if done:
            if total_reward > 300 and count < 5 and video:
                save_video(env.render(), './videos', fps=30, name_prefix='test', episode_index=i)
                env.reset()
            break
    print("Episode: {}, total_reward: {}".format(i, total_reward))
    avg.append(total_reward)
env.close()

print("Average reward over 100 episodes: {}".format(np.mean(avg)))

# plot graph of rewards vs episode
plt.plot(avg)
plt.title('Rewards over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()