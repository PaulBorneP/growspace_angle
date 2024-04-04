
import os
import glob
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import Categorical

import numpy as np


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


# class ActorCritic(nn.Module):
#     def __init__(self, env, device: str):
#         super(ActorCritic, self).__init__()
#         """ActorCritic network for PPO with discrete action space"""
#         self.device = device

#         self.actor = nn.Sequential(
#             nn.Linear(env.observation_space.prod(), 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#             nn.Linear(64, env.action_space.n),
#             nn.Softmax(dim=-1)
#         )

#         self.critic = nn.Sequential(
#             nn.Linear(env.observation_space.prod(), 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#             nn.Linear(64, 1)
#         )

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, envs, device: str):
        super(ActorCritic, self).__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),

        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.action_space.n), std=0.01),
            nn.Softmax(dim=-1)
        )
        self.device = device

    def forward(self):
        raise NotImplementedError

    def act(self, state: torch.Tensor, memory: Memory) -> int:
        """Sample action from policy"""
        state = state.to(self.device).reshape(1, -1)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        return action.item()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        """Evaluate logprobs and state value"""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def save(self, filename, directory):
        torch.save(self.state_dict(), '%s/%s_actor.pth' %
                   (directory, filename))

    def load(self, filename, directory):
        self.load_state_dict(torch.load('%s/%s_actor.pth' %
                             (directory, filename)))


class PPO_Simple:
    def __init__(self, env, lr_actor: float, lr_critic: float, betas: tuple, gamma: float, n_epochs: int, eps_clip: float, device: str) -> None:

        self.lr_actor = lr_actor
        self.critic = lr_critic
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.n_epochs = n_epochs

        self.policy = ActorCritic(env, device).to(device)
        self.old_policy = ActorCritic(env, device).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.device = device
        self.MSE_loss = nn.MSELoss()

    def update(self, memory: Memory):

        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()

        # Optimize policy for K epochs:

        for _ in range(self.n_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            value_loss = self.MSE_loss(state_values, rewards)
            action_loss = - \
                torch.min(ratios * advantages, torch.clamp(ratios,
                          1-self.eps_clip, 1+self.eps_clip))
            loss = action_loss + value_loss - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            return loss.mean(), value_loss.mean(), action_loss.mean(), dist_entropy.mean()

        # Copy new weights into old policy:
        self.old_policy.load_state_dict(self.policy.state_dict())

    def save(self, filename: str, directory: str) -> None:
        torch.save(self.policy.state_dict(), '%s/%s_actor_critic.pth' %
                   (directory, filename))

    def load(self, filename: str, directory: str) -> None:
        self.policy.load_state_dict(torch.load(
            '%s/%s_actor_critic.pth' % (directory, filename)))
        self.old_policy.load_state_dict(self.policy.state_dict())


if __name__ == "__main__":
    # test on cartpole
    import gym
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    n_latent_var = 64           # number of variables in hidden layer
    lr_actor = 0.02
    lr_critic = 0.02
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    n_epochs = 4                # update policy for n epochs
    eps_clip = 0.2              # clip parameter for PPO
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ppo = PPO_Simple(state_dim, action_dim, n_latent_var, lr_actor,
                     lr_critic, betas, gamma, n_epochs, eps_clip, device)

    # memory = Memory()
    # max_timesteps = 300
    # update_timestep = 2000
    # timestep = 0
    # running_reward = 0

    # # training loop
    # for i_episode in range(1000):
    #     state = env.reset()
    #     for t in range(max_timesteps):
    #         timestep += 1
    #         action = ppo.policy.act(torch.tensor(state, dtype=torch.float32).to(device), memory)
    #         state, reward, done, _ = env.step(action)
    #         memory.rewards.append(reward)
    #         memory.is_terminals.append(done)

    # #         # update if its time
    #         if timestep % update_timestep == 0:
    #             ppo.update(memory)
    #             memory.clear_memory()
    #             timestep = 0

    #         running_reward += reward
    #         if done:
    #             break

    #     if i_episode % 10 == 0:
    #         print(f"Episode {i_episode}, running_reward: {running_reward}")
    #         running_reward = 0

    # env.close()

    # # save model
    # ppo.save("ppo_cartpole", "./")
    # print("Model saved")

    # # load model
    ppo.load("ppo_cartpole", "./")
    print("Model loaded")
    memory = Memory()

    # show that model is working

    env = gym.make('CartPole-v1')
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = ppo.policy.act(torch.tensor(
            state, dtype=torch.float32).to(device), memory)
        state, reward, done, _ = env.step(action)
    env.close()

    print("Model is working")
