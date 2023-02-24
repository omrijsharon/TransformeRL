import torch
import torch.nn as nn
import gym
import matplotlib.pyplot as plt
import numpy as np

from core.experience import ExperienceReplay, collect_experience
from core.model import TransformerRL

# env is cartpole v1:
env = gym.make('CartPole-v1')
# env is lunarlander v2:
# env = gym.make('LunarLanderContinuous-v2')
experience_capacity = 10000
experience = ExperienceReplay(env, experience_capacity)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize the model
model = TransformerRL(state_space=env.observation_space.shape,
                      action_space=env.action_space,
                      d_model=64,
                      nhead=4,
                      num_layers=2,
                      dropout=0).to(device)

# Define the loss function and optimizer
loss_next_state_fn = nn.MSELoss()
loss_next_action_fn = nn.CrossEntropyLoss()
loss_coefficients = torch.tensor([1.0, 1.0]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

batch_size = 1024
num_iterations = 100
num_epochs = experience_capacity // batch_size
print(f'num_epochs: {num_epochs}')

loss_list_next_state = np.array([])
loss_list_next_action = np.array([])
loss_total_list = np.array([])

# a list with the plots names:
plots_names = ['Loss next state', 'Loss next action', 'Total loss']
fig, axs = plt.subplots(1, len(plots_names))
# set titles:
for i in range(len(plots_names)):
    axs[i].set_title(plots_names[i])

show_last_n_loss = 100
# Train the model
for i in range(num_iterations):
    # collect experience with random actions:
    experience.reset()
    collect_experience(env, experience)
    for j in range(num_epochs):
        #clear all plots:
        for k in range(len(plots_names)):
            axs[k].clear()
            axs[k].set_title(plots_names[k])
        # Get a batch of training examples
        state, action, next_state = experience.sample(batch_size=batch_size, device=device)

        # Zero the gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        predicted_next_state, predicted_next_action = model(state, action)

        # Compute the loss
        loss_next_state = loss_next_state_fn(predicted_next_state, next_state)
        loss_returns = mse_loss_with_std(predicted_returns_mean, predicted_returns_std, returns)
        loss_done = loss_done_fn(predicted_done, done)
        losses = torch.stack([loss_next_state, loss_returns, loss_done])
        loss = losses @ loss_coefficients
        # Backward pass
        loss.backward()
        optimizer.step()
        loss_list_next_state = np.append(loss_list_next_state, loss_next_state.cpu().detach().numpy())
        loss_list_returns = np.append(loss_list_returns, loss_returns.cpu().detach().numpy())
        list_returns_std = np.append(list_returns_std, predicted_returns_std.cpu().detach().mean().numpy())
        loss_list_done = np.append(loss_list_done, loss_done.cpu().detach().numpy())
        loss_list = np.append(loss_list, loss.cpu().detach().numpy())
        # plot in 4 different plots: loss_next_state, loss_returns, loss_done and loss
        axs[0].plot(loss_list_next_state[-show_last_n_loss:])
        axs[1].plot(loss_list_returns[-show_last_n_loss:])
        axs[2].plot(list_returns_std[-show_last_n_loss:])
        axs[3].plot(loss_list_done[-show_last_n_loss:])
        axs[4].plot(loss_list[-show_last_n_loss:])
        # update the plots:
        plt.pause(0.001)
plt.show()


