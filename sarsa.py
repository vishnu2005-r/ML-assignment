import numpy as np
import matplotlib.pyplot as plt
import gym #library for reinforcement problems
from gym import spaces
import time
class GridEnvironment(gym.Env):
    metadata = { 'render.modes': [] }
    
    def _init_(self):
        self.observation_space = spaces.Discrete(16)
        self.action_space = spaces.Discrete(4)
        self.done = False
        
    def reset(self):
        
        self.done = False
        self.state = np.zeros((4,4))
        self.agent_pos = [0, 0]
        self.goal_pos = [3, 3]       
        self.danger1_pos = [2,0] #first danger position (-3)
        self.danger2_pos = [1,2] #second danger position (-4)
        self.gold1_pos = [1,0] #First positive reward position (+2)
        self.gold2_pos = [3,1] #Second positive reward position (+5)

        self.state[tuple(self.agent_pos)] = 1
        self.state[tuple(self.goal_pos)] = 0.8
        self.state[tuple(self.gold1_pos)] = 0.7
        self.state[tuple(self.gold2_pos)] = 0.7
        self.state[tuple(self.danger1_pos)] = 0.3
        self.state[tuple(self.danger2_pos)] = 0.3
        observation = self.state.flatten()
        return observation
    
    def step(self, action):

        self.state = np.random.choice(self.observation_space.n)
        if action == 0:
          self.agent_pos[0] += 1
        if action == 1:
          self.agent_pos[0] -= 1
        if action == 2:
          self.agent_pos[1] += 1
        if action == 3:
          self.agent_pos[1] -= 1
          
        self.agent_pos = np.clip(self.agent_pos, 0, 3)
        self.state = np.zeros((4,4))
        self.state[tuple(self.agent_pos)] = 1
        self.state[tuple(self.goal_pos)] = 0.8
        self.state[tuple(self.gold1_pos)] = 0.7
        self.state[tuple(self.gold2_pos)] = 0.7
        self.state[tuple(self.danger1_pos)] = 0.3
        self.state[tuple(self.danger2_pos)] = 0.3
        observation = self.state.flatten()
        
        reward = 0
        if (self.agent_pos == self.goal_pos).all():
          reward = 20 #Target position: Given max reward if it reaches target
          self.done = True

        elif (self.agent_pos == self.danger1_pos).all():
            reward = -3 #A negative reward -3 if it enters 1st danger position

        elif (self.agent_pos == self.danger2_pos).all():
            reward = -4 #A negative reward of -4 if it enters 2nd danger position

        elif (self.agent_pos == self.gold1_pos).all():
            reward = 2 #A reward of +2 at [1,1]

        elif (self.agent_pos == self.gold2_pos).all():
            reward = 5 #A reward of +5 at [2,2]
        
        return self.agent_pos, reward, self.done
        
    def render(self):
        plt.imshow(self.state)

#SARSA
def sarsa(discount_factor = 0.95, timesteps = 15, episodes = 1000, evaluation_results = False):
  env = GridEnvironment()
  obs = env.reset()#resets the environment to its initial configuration

  #Intialize parameters
  learning_rate = 0.15 #alpha
  discount_factor = discount_factor #how much weightage to put on future rewards
  det_epsilon = 0.99 # For all states in deterministic environment p(s', r/s, a) = {0, 1}: Either action taken or No action taken


  #Intial state
  current_state = 0 #s1
  action_val = [0,1,2,3]

  #Q table representing 16 rows: one for each state (i.e., 0,1,2,...15) -> (i.e., s1, s2, s3,....s16) and 4 columns: one for each action (i.e., 0,1,2,3) -> (down,up,right,left)
  # (0-15, 0-3) remember the dimension is one less
  q_table = np.zeros((16,4))

  #mapping next_state co-ordinates to q_table co-ordinates
  states = {(0,0): 0, (0,1): 1, (0,2): 2, (0,3): 3,
                (1,0): 4, (1,1): 5, (1,2): 6, (1,3): 7,
                (2,0): 8, (2,1): 9, (2,2): 10, (2,3): 11,
                (3,0): 12, (3,1): 13, (3,2): 14, (3,3): 15} #16 states

  #Empty lists to store values
  optimal = []
  reward_values = []
  total_timesteps = []
  epsilon_values = []
  eva_rewards = []


  done = False #signifies if agent reached terminal or not 
  total_episodes = episodes
  eva_episodes = 10
  avg_timesteps = 0
  epsilon = 1 #multiply by 0.995 for each episode(#after 30 iterations# or terminal state reached)
  decay_factor = (0.01/1)(1/total_episodes)
  target = np.array([3,3])

  #For evaluation results
  if evaluation_results: 
     total_episodes += eva_episodes 
     print("Evaluation Results")

  for episode in range(1, total_episodes+1):
    
    obs = env.reset() #resets the environment
    current_state = 0 
    total_rewards = 0
    timestep = 0


    #e - greedy algorithm to choose s and a
    rand_num = np.random.random()
    if epsilon > rand_num:
      action = np.random.choice(action_val)
    else:
      action = np.argmax(q_table[current_state]) #action in current state s with max_q value

    while timestep < timesteps: #(i.e., considering untill the terminal is reached or 15 timesteps are completed)
    
      rand_num = np.random.random()
      if det_epsilon > rand_num: #Choosing an action in deterministic environment
          
          next_state_pos, reward, done = env.step(action)
          next_state = states[tuple(next_state_pos)]

          if reward == 20:
            reward += 100

          if np.linalg.norm(target - np.array(next_state_pos)) <= 1:
            reward = reward + 5  #before +1, 5 is good

          #e - greedy algorithm to choose next_action for next_state
          rand_num = np.random.random()
          if epsilon > rand_num:
            next_action = np.random.choice(action_val)
          else:
            next_action = np.argmax(q_table[next_state]) #action in next state s' with max_q value
          
          #q-value update function for SARSA
          q_table[current_state][action] = q_table[current_state][action] + learning_rate*(reward + discount_factor*q_table[next_state][next_action] - q_table[current_state][action])

          if episode == total_episodes:
            optimal.append(current_state+1)

          total_rewards += reward #Captured all the rewards in each episode
          timestep += 1 #Number of timesteps in each episode

          current_state = next_state #next_state is assigned to current_state
          action = next_action
          
          if done == True: #If terminal or target state reached then stop the episode
            done = False
            break        
          
    
    #Results after each episode
    avg_timesteps += timestep #Capturing all timesteps for all 100 episodes
    total_timesteps.append(avg_timesteps)

    reward_values.append(total_rewards) #Append rewards in every episode
    epsilon_values.append(epsilon) #Append epsilon values in every episode


    if epsilon > 0.01: #keeping epsilon in [0.01 - 1] range as if it falls below 0.01 it will exploit more: choosing best actions. We want our agent to explore a bit: choosing random actions
        epsilon = epsilon*decay_factor
    else:
        epsilon = 0.01


    if (episode % 100) == 0 and evaluation_results == False: #printing results for every 100 episodes
      print("Episode: {}, Rewards: {}, Average timesteps taken: {}, epsilon: {}".format(episode, total_rewards, avg_timesteps//100, epsilon))
      avg_timesteps = 0

    #evaluation results
    if evaluation_results:
      if episode > total_episodes - eva_episodes:
         eva_rewards.append(reward)

    #printing the optimal path in last episode
    if episode == total_episodes:
          print("Optimal Path: ")
          for i in optimal:
            print(i,"->", end = " ")
          print(next_state+1)

  #Final Q - Table
  print("Q Table: \n", q_table)

  #Plotting the results
  #x, y co-ordinates
  x = [episode for episode in range(total_episodes)]
  yr = reward_values
  ye = epsilon_values

  yr_eva = eva_rewards
  x_eva = [episode for episode in range(eva_episodes)]
 

  if evaluation_results:
      #episodes vs rewards
      plt.plot(x_eva,yr_eva)
      plt.title("Rewards per episode")
      plt.xlabel('Episodes')
      plt.ylabel('Rewards')

  else:
      #episodes vs epsilon
      #Plots showing episodes vs epsilon, episodes vs rewards
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
      ax1.plot(x, ye)
      ax1.set_title("Epsilon decay")

      #episodes vs rewards
      ax2.plot(x,yr)
      ax2.set_title("Rewards per episode")
sarsa(evaluation_results=True) #During evaluation the agent doesn't learn it just picks the best next action (i.e. greedy action)
