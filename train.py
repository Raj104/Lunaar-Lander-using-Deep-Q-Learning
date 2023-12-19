import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from gym.utils.save_video import save_video

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.vstack(states), actions, rewards, np.vstack(next_states), dones
    
    def __len__(self):
        return len(self.buffer)

# Training function
def train_dqn(q_network, target_network, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    current_q_values = q_network(states).gather(1, actions.unsqueeze(1))
    next_q_values = target_network(next_states).max(1)[0].detach()
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == '__main__':

    render_mode = 'human' # 'human' or 'rgb_array_list'
    video = False #if render_mode == 'human' else True

    vid_count = 5 # number of videos to save
    episode_trigger_arr = [0, 100, 300,600,1200,1800] # episodes to save videos for

    # Hyperparameters
    env = gym.make('LunarLander-v2', render_mode=render_mode)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    learning_rate = 0.001
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    target_update_frequency = 10
    batch_size = 64
    capacity = 10000
    num_episodes = 2000

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Q-networks and optimizer
    q_network = QNetwork(state_size, action_size).to(device)
    target_network = QNetwork(state_size, action_size).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    # Initialize experience replay buffer
    replay_buffer = ReplayBuffer(capacity)

    # Lists to store rewards for plotting
    all_rewards = []

    # maximum number of steps per episode
    max_steps = 500

    # Training loop
    epsilon = epsilon_start
    for episode in range(num_episodes):
        state = env.reset()
        state = state[0]
        total_reward = 0
        time_step = 0

        while True:
            
            # Epsilon-greedy exploration
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_network(torch.FloatTensor(state).to(device))
                    action = q_values.argmax().item()

            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Training step
            train_dqn(q_network, target_network, optimizer, replay_buffer, batch_size, gamma)

            time_step += 1
            if time_step > max_steps:
                done = True

            if done:
                if video:
                    save_video(env.render(), f'./videos', fps=30,episode_trigger=lambda ep: ep in episode_trigger_arr, episode_index=episode)
                break

        # Update target network every target_update_frequency episodes
        if episode % target_update_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Epsilon decay
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Store total reward for plotting
        all_rewards.append(total_reward)

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    # Save the trained model
    torch.save(q_network.state_dict(), 'lunar_lander_dqn.pth')

    # Plot the rewards
    plt.plot(all_rewards)
    plt.title('Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()