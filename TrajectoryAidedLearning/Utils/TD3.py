from os import stat
import numpy as np 
from matplotlib import pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MEMORY_SIZE = 100000


# hyper parameters
BATCH_SIZE = 100
GAMMA = 0.99
tau = 0.005
NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
POLICY_NOISE = 0.2


class SmartBufferTD3(object):
    def __init__(self, max_size=1000000, state_dim=14, act_dim=1):     
        self.max_size = max_size
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.ptr = 0

        self.states = np.empty((max_size, state_dim))
        self.actions = np.empty((max_size, act_dim))
        self.next_states = np.empty((max_size, state_dim))
        self.rewards = np.empty((max_size, 1))
        self.dones = np.empty((max_size, 1))

    def add(self, s, a, s_p, r, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.next_states[self.ptr] = s_p
        self.rewards[self.ptr] = r
        self.dones[self.ptr] = d

        self.ptr += 1
        
        if self.ptr == 99999: self.ptr = 0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.ptr-1, size=batch_size)
        states = np.empty((batch_size, self.state_dim))
        actions = np.empty((batch_size, self.act_dim))
        next_states = np.empty((batch_size, self.state_dim))
        rewards = np.empty((batch_size, 1))
        dones = np.empty((batch_size, 1))

        for i, j in enumerate(ind): 
            states[i] = self.states[j]
            actions[i] = self.actions[j]
            next_states[i] = self.next_states[j]
            rewards[i] = self.rewards[j]
            dones[i] = self.dones[j]

        return states, actions, next_states, rewards, dones

    def size(self):
        return self.ptr


class Actor(nn.Module):   
    def __init__(self, state_dim, action_dim, max_action, h_size):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, h_size)
        self.l2 = nn.Linear(h_size, h_size)
        self.l3 = nn.Linear(h_size, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x)) 
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h_size):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, h_size)
        self.l2 = nn.Linear(h_size, h_size)
        self.l3 = nn.Linear(h_size, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, h_size)
        self.l5 = nn.Linear(h_size, h_size)
        self.l6 = nn.Linear(h_size, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1



class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, name):
        self.name = name
        self.state_dim = state_dim
        self.max_action = max_action
        self.act_dim = action_dim

        self.actor = None
        self.actor_target = None
        self.actor_optimizer = None

        self.critic = None
        self.critic_target = None
        self.critic_optimizer = None

        self.replay_buffer = SmartBufferTD3(state_dim=state_dim, act_dim=action_dim)

    def create_agent(self, h_size):
        state_dim = self.state_dim
        action_dim = self.act_dim
        max_action = self.max_action
        self.actor = Actor(state_dim, action_dim, max_action, h_size)
        self.actor_target = Actor(state_dim, action_dim, max_action, h_size)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim, h_size)
        self.critic_target = Critic(state_dim, action_dim, h_size)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def select_action(self, state, noise=0.1):
        return self.act(state, noise=noise)

    def act(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1))

        action = self.actor(state).data.numpy().flatten()
        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=self.act_dim))
            
        return action.clip(-self.max_action, self.max_action)

    def get_critic_value(self, state, action):
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)

        current_Q1, current_Q2 = self.critic(state[None, :], action[None, :])
        ret = current_Q1.detach().item()

        return ret

    def train(self, iterations=2):
        if self.replay_buffer.size() < BATCH_SIZE * 5:
            return 0
        for it in range(iterations):
            # Sample replay buffer 
            x, u, y, r, d = self.replay_buffer.sample(BATCH_SIZE)
            state = torch.FloatTensor(x)
            action = torch.FloatTensor(u)
            next_state = torch.FloatTensor(y)
            done = torch.FloatTensor(1 - d)
            reward = torch.FloatTensor(r)

            # Select action according to policy and add clipped noise 
            noise = torch.FloatTensor(u).data.normal_(0, POLICY_NOISE)
            noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * GAMMA * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % POLICY_FREQUENCY == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        total_loss = actor_loss + critic_loss
        
        return total_loss

    def save(self, directory="./saves"):
        filename = self.name

        torch.save(self.actor, '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic, '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.actor_target, '%s/%s_actor_target.pth' % (directory, filename))
        torch.save(self.critic_target, '%s/%s_critic_target.pth' % (directory, filename))

    def load(self, directory="./saves"):
        filename = self.name
        self.actor = torch.load('%s/%s_actor.pth' % (directory, filename))
        self.critic = torch.load('%s/%s_critic.pth' % (directory, filename))
        self.actor_target = torch.load('%s/%s_actor_target.pth' % (directory, filename))
        self.critic_target = torch.load('%s/%s_critic_target.pth' % (directory, filename))

        print("Agent Loaded")

    def try_load(self, load=True, h_size=300, path=None):
        if load:
            try:
                self.load(path)
            except Exception as e:
                print(f"Exception: {e}")
                print(f"Unable to load model")
                pass
        else:
            print(f"Not loading - restarting training")
            self.create_agent(h_size)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
