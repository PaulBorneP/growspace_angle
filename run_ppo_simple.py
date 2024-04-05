
import wandb
import torch
from torch import nn
from ppo_simple import PPO_Simple, Memory
import numpy as np
from envs import make_env
import gym
import growspace
from tqdm import tqdm


def main():
    env_name = "GrowSpaceEnv-Angular-v0"
    env = gym.make(env_name)
    lr_actor = 0.02
    lr_critic = 0.02
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    n_epochs = 4                # update policy for n epochs
    eps_clip = 0.2              # clip parameter for PPO
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ppo = PPO_Simple(env, lr_actor,
                     lr_critic, betas, gamma, n_epochs, eps_clip, device)

    wandb.init(project='growspace_angular', name="PPO_Simple_Angular",
               dir="/Users/newt/Desktop/CS/SDI/RL/projet/angular_growspace/wandb")

    memory = Memory()
    max_timesteps = 2500
    log_interval = 1
    timestep = 0
    running_reward = 0
    episode_rewards = []
    episode_length = []
    episode_success = []

    # training loop
    for j in tqdm(range(400)):
        state = env.reset()
        running_reward = 0
        timestep = 0
        for t in range(max_timesteps):
            timestep += 1
            total_num_steps = (j + 1) * max_timesteps
            action = ppo.policy.act(torch.tensor(
                state, dtype=torch.float32).to(device), memory)
            state, reward, done, info = env.step(action)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            running_reward += reward

            if done:
                episode_rewards.append(running_reward)
                episode_length.append(t)
                if info['success']:
                    episode_success.append(1)
                else:
                    episode_success.append(0)
                state = env.reset()
                running_reward = 0
                timestep = 0

        # update if its time
        loss, value_loss, action_loss, dist_entropy = ppo.update(
            memory)

        memory.clear_memory()

        if j % log_interval == 0 and len(episode_rewards) > 1:

            wandb.log({"Reward Min": np.min(episode_rewards)},
                      step=total_num_steps)
            wandb.log({"Summed Reward": np.sum(episode_rewards)},
                      step=total_num_steps)
            wandb.log({"Reward Mean": np.mean(episode_rewards)},
                      step=total_num_steps)
            wandb.log({"Reward Max": np.max(episode_rewards)},
                      step=total_num_steps)

            wandb.log({"Entropy": dist_entropy}, step=total_num_steps)
            wandb.log({"Value Loss": value_loss}, step=total_num_steps)
            wandb.log({"Action Loss": action_loss}, step=total_num_steps)

            episode_rewards.clear()
            episode_length.clear()
            episode_success.clear()

    env.close()


if __name__ == "__main__":
    main()

#  envs = make_vec_envs(env_name, seed, num_processes,
#                          gamma, log_dir, device, False, custom_gym)

#     actor_critic = Policy(
#         envs.observation_space.shape,
#         envs.action_space,
#         base_kwargs={'recurrent': recurrent_policy})
#     actor_critic.to(device)

#     agent = PPO(
#         actor_critic,
#         clip_param,
#         ppo_epoch,
#         num_mini_batch,
#         value_loss_coef,
#         entropy_coef,
#         lr=lr,
#         eps=eps,
#         max_grad_norm=max_grad_norm,
#         optimizer=optimizer,
#         momentum=momentum
#     )

#     rollouts = RolloutStorage(num_steps, num_processes,
#                               envs.observation_space.shape, envs.action_space,
#                               actor_critic.recurrent_hidden_state_size)

#     obs = envs.reset()
#     rollouts.obs[0].copy_(obs)
#     rollouts.to(device)

#     episode_rewards = []
#     episode_length = []
#     episode_success = []

#     num_updates = int(
#         num_env_steps) // num_steps // num_processes

#     for j in range(num_updates):

#         action_dist = np.zeros(envs.action_space.n)

#         total_num_steps = (j + 1) * num_processes * num_steps

#         if use_linear_lr_decay:
#             # decrease learning rate linearly
#             utils.update_linear_schedule(
#                 agent.optimizer, j, num_updates,
#                 agent.optimizer.lr if algo == "acktr" else lr)
#         # new_branches = []
#         for step in range(num_steps):
#             with torch.no_grad():
#                 value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
#                     rollouts.obs[step], rollouts.recurrent_hidden_states[step],
#                     rollouts.masks[step])

#             # Obser reward and next obs
#             obs, reward, done, infos = envs.step(action)

#             # if isinstance(action_space_type, Discrete):
#             action_dist[action] += 1

#             for info in infos:
#                 if 'episode' in info.keys():
#                     episode_rewards.append(info['episode']['r'])
#                     episode_length.append(info['episode']['l'])
#                     wandb.log(
#                         {"Episode_Reward": info['episode']['r']}, step=total_num_steps)

#                 if 'success' in info.keys():
#                     episode_success.append(info['success'])
#                     wandb.log(
#                         {"Episode_Success": info['success']}, step=total_num_steps)

#             # If done then clean the history of observations.
#             masks = torch.FloatTensor(
#                 [[0.0] if done_ else [1.0] for done_ in done])
#             bad_masks = torch.FloatTensor(
#                 [[0.0] if 'bad_transition' in info.keys() else [1.0]
#                  for info in infos])
#             rollouts.insert(obs, recurrent_hidden_states, action,
#                             action_log_prob, value, reward, masks, bad_masks)

#         with torch.no_grad():
#             next_value = actor_critic.get_value(
#                 rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
#                 rollouts.masks[-1]).detach()

#         rollouts.compute_returns(next_value, use_gae, gamma,
#                                  gae_lambda, use_proper_time_limits)

#         value_loss, action_loss, dist_entropy = agent.update(rollouts)

#         rollouts.after_update()

#         if j % log_interval == 0 and len(episode_rewards) > 1:

#             wandb.log({"Reward Min": np.min(episode_rewards)},
#                       step=total_num_steps)
#             wandb.log({"Summed Reward": np.sum(episode_rewards)},
#                       step=total_num_steps)
#             wandb.log({"Reward Mean": np.mean(episode_rewards)},
#                       step=total_num_steps)
#             wandb.log({"Reward Max": np.max(episode_rewards)},
#                       step=total_num_steps)
#             wandb.log({"Entropy": dist_entropy}, step=total_num_steps)
#             wandb.log({"Value Loss": value_loss}, step=total_num_steps)
#             wandb.log({"Action Loss": action_loss}, step=total_num_steps)

#             episode_rewards.clear()
#             episode_length.clear()
#             episode_success.clear()


# if __name__ == "__main__":
#     main()
