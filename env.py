#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 01:05:18 2021

@author: josev
"""

import random
import gym
from DQN import DQN
import torch
from torchvision import transforms, utils
from replay_memory import experience_replay
import torch.nn.functional as F

from collections import namedtuple
from itertools import count
import random
import matplotlib
import matplotlib.pyplot as plt

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

EPISODES = 100
EPS = 1.0
BATCH_SIZE = 32
LR = 1e-3
EPS_ITER_MAX = 1e6
GAMMA = 0.99
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
TARGET_UPDATE = 50


def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def eps_decay(i): 
    return 0.9*EPS - (0.9*EPS*i/EPS_ITER_MAX) + 0.1*EPS

def main(env, policy_net, target_net, action_space):
    
    Transition = namedtuple("Transition", ('state', 'action', 'reward', 'next_state', "done"))
    rm = experience_replay()
    opt = torch.optim.RMSprop(policy_net.parameters(), lr=LR)
    
    
    preprocess = transforms.Compose([transforms.ToPILImage(),
                                    transforms.RandomGrayscale(p=1),
                                    transforms.Resize((84, 84)),
                                    transforms.ToTensor()])
    
    cnt = 0
    episodes_duration = []
    rewards_episode = []
    
    policy_net = policy_net.to(DEV)
    target_net = target_net.to(DEV)
    
    for episode in range(EPISODES):
        state = env.reset() # image
        state = preprocess(state).unsqueeze(0).to(DEV)
        reward_episode = 0
        for i in count():
            probs = policy_net(state)
            if eps_decay(cnt) > random.random():
                action = torch.tensor([random.randrange(0, 18)])
            else:
                action = torch.argmax(probs).view(1)
                
            next_state, reward, done, info = env.step(action.item())
            reward_episode += reward
            reward = torch.tensor([reward])
            next_state = preprocess(next_state).unsqueeze(0)
            cnt += 1
            rm.add_to_memory((state.to("cpu"), action.to("cpu"), reward, next_state, done))
            
            batch = rm.get_batch_for_replay(BATCH_SIZE)
            if batch is None: continue
            batch = Transition(*zip(*batch))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                                device=DEV)
            
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(DEV)
            
            state_batch = torch.cat(batch.state).to(DEV)
            action_batch = torch.cat(batch.action).to(DEV)
            reward_batch = torch.cat(batch.reward).to(DEV)
            
            
            d = policy_net(state_batch)
            state_action_value = policy_net(state_batch).gather(1, action_batch.reshape(-1, 1))
            
            
            next_states_values = torch.zeros(BATCH_SIZE, device=DEV)
            next_states_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
            
            y = (next_states_values*GAMMA) + reward_batch
            
            
            
            loss = F.smooth_l1_loss(state_action_value.squeeze(), y)
            
            opt.zero_grad()
            loss.backward()
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
                
                
            opt.step()
            if done:
                episodes_duration.append(i+1)
                rewards_episode.append(reward_episode)
                print(f"step: {cnt} || reward episode: {reward_episode}")
                #plot_durations(episodes_duration)
                break
                
        if episode%TARGET_UPDATE==0:
            target_net.load_state_dict(policy_net.state_dict())
            



if __name__ == "__main__":
    
    env = gym.make('BattleZone-v0')
    action_space = env.action_space.n # [0, 18] actions
    n_channel = 3
    state_dim = (210, 160, 3)
    policy_net = DQN(action_space, n_channel)
    target_net = DQN(action_space, n_channel)
    
    target_net.load_state_dict(policy_net.state_dict())
    
    main(env, policy_net, target_net, action_space)
    
    
    
    
    