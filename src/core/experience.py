import numpy as np
import torch
from gym.spaces import Box, Discrete


class ExperienceReplay:
    def __init__(self, env, capacity):
        self.env = env
        self.capacity = capacity
        self.obs_dim = env.observation_space.shape
        self.action_dim = None
        self.n_possible_actions = None
        if isinstance(self.env.action_space, Discrete):
            self.action_dim = 1
            self.n_possible_actions = self.env.action_space.n
        elif isinstance(self.env.action_space, Box):
            self.action_dim = np.prod(env.action_space.shape)
        else:
            raise NotImplementedError
        self.obs = torch.empty((self.capacity, *self.obs_dim))
        self.actions = torch.empty((self.capacity, self.action_dim))
        self.episode = torch.empty((self.capacity, 1))
        self.next_obs = torch.empty((self.capacity, *self.obs_dim))
        self.counter = 0
        self.episode_number = 0
        self.is_full = False

    def reset(self):
        self.obs = torch.empty((self.capacity, *self.obs_dim))
        self.actions = torch.empty((self.capacity, self.action_dim))
        self.episode = torch.empty((self.capacity, 1))
        self.next_obs = torch.empty((self.capacity, *self.obs_dim))
        self.counter = 0
        self.episode_number = 0
        self.is_full = False

    def convert_to_tensor(self, array):
        return torch.from_numpy(np.array(array)).float()

    def add(self, obs, action, done, next_obs):
        try:
            obs = self.convert_to_tensor(obs)
        except:
            print(obs)
            raise
        next_obs = self.convert_to_tensor(next_obs)
        action = self.convert_to_tensor(action).reshape(1, self.action_dim)
        self.obs[self.counter] = obs
        self.actions[self.counter] = action
        self.episode[self.counter] = self.episode_number
        self.next_obs[self.counter] = next_obs
        self.counter = (self.counter + 1) % self.capacity
        if done:
            self.episode_number += 1
        if self.counter == 0:
            self.is_full = True

    def sample(self, batch_size, seq_len, device='cpu'):
        episodes_number, episodes_length = torch.unique(self.episode, return_counts=True)
        episodes_number = episodes_number[episodes_length > seq_len]
        assert len(episodes_number) > 0, 'Sequence length is too long and there are no episodes long enough.' \
                                         'The longest episode is {} steps long.'.format(episodes_length.max().item())
        sampled_episodes = np.random.choice(episodes_number, batch_size, replace=True)
        # sampled episodes indices:
        episodes_indices = [np.argwhere(self.episode==sampled_episode) for sampled_episode in sampled_episodes]
        obs_batch = torch.empty((batch_size, seq_len, *self.obs_dim))
        action_batch = torch.empty((batch_size, seq_len, self.action_dim))
        if isinstance(self.env.action_space, Discrete):
            action_batch = action_batch.long()
        elif isinstance(self.env.action_space, Box):
            action_batch = action_batch.float()
        else:
            raise NotImplementedError
        next_obs_batch = torch.empty((batch_size, seq_len, *self.obs_dim))
        for i in range(batch_size):
            episode_indices = episodes_indices[i][0]
            #idx until seq_len before the end of the episode:
            idx = episode_indices[: -seq_len]
            # random index from the remaining indices:
            idx = np.random.choice(idx, 1)[0]
            obs_batch[i], action_batch[i], next_obs_batch[i] = self.obs[idx: idx+seq_len], self.actions[idx: idx+seq_len], self.next_obs[idx: idx+seq_len]
        return obs_batch.to(device), action_batch.to(device), next_obs_batch.to(device)

    def __len__(self):
        if self.is_full:
            return self.capacity
        else:
            return self.counter


def collect_experience(env, replay_buffer):
    while len(replay_buffer) < replay_buffer.capacity:
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.add(obs, action, done, next_obs)
            obs = next_obs
            if replay_buffer.is_full:
                break
        print(len(replay_buffer) / replay_buffer.capacity * 100, '%')


if __name__ == '__main__':
    import gym
    import matplotlib.pyplot as plt
    env = gym.make('CartPole-v1')
    experience_capacity = 1000
    experience = ExperienceReplay(env, experience_capacity)
    collect_experience(env, experience)
    obs, action, next_obs = experience.sample(10, 20)
    # returns = experience.calculate_returns(gamma=0.9)
    # plt.plot(returns)
    # plt.show()
    print()


