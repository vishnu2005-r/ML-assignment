
import tensorflow as tf
import gym # openAi gym
from gym import envs
import numpy as np 
import pandas as pd

print(env.observation_space)
print()
env.env.s=29 #Random State
env.render()
env.env.s = 200 #Random State
env.render()
rew_tot=0
obs= env.reset()
env.render()
for _ in range(6):
    action = env.action_space.sample() #take step using random action from possible actions (actio_space)
    obs, rew, done, info = env.step(action) 
    rew_tot = rew_tot + rew
    env.render()
#Print the reward of these random action
print("Reward: %r" % rew_tot) 
     NUM_ACTIONS = env.action_space.n #assigning action
NUM_STATES = env.observation_space.n #assigning state
V = np.zeros([NUM_STATES]) # The Value for each state
Pi = np.zeros([NUM_STATES], dtype=int)  # policy
gamma = 0.9 # discount factor
significant_improvement = 0.01

def best_action_value(s):
    # to find the highest value action (max_a) in state s
    best_a = None
    best_value = float('-inf') # negative infinity

    # Adding a loop to run in all possible actions to find the best current action
    for a in range (0, NUM_ACTIONS):
        env.env.s = s
        s_new, rew, done, info = env.step(a) #take the action
        v = rew + gamma * V[s_new]
        if v > best_value:
            best_value = v
            best_a = a
    return best_a

iteration = 0 #intializing
while True:
    # biggest_change is referred to by the mathematical symbol delta in equations
    biggest_change = 0
    for s in range (0, NUM_STATES):
        old_v = V[s]
        action = best_action_value(s) #choosing an action with the highest future reward
        env.env.s = s # goto the state
        s_new, rew, done, info = env.step(action) #take the action
        V[s] = rew + gamma * V[s_new] #Update Value for the state using Bellman equation
        Pi[s] = action
        biggest_change = max(biggest_change, np.abs(old_v - V[s]))
    iteration += 1
    if biggest_change < significant_improvement:
        print (iteration,' iterations done')
        break
     NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n
Q = np.zeros([NUM_STATES, NUM_ACTIONS]) #Assiging
gamma = 0.9 # discount factor
alpha = 0.9 # learning rate
for episode in range(1,1001):
    done = False
    rew_tot = 0
    obs = env.reset()
    while done != True:
            action = np.argmax(Q[obs]) #choosing the action with the highest Q value 
            obs2, rew, done, info = env.step(action) #take the action
            Q[obs,action] += alpha * (rew + gamma * np.max(Q[obs2]) - Q[obs,action]) #Update Q-marix using Bellman equation
            rew_tot = rew_tot + rew
            obs = obs2   
    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode,rew_tot))
