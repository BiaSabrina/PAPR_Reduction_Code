import gym
import numpy as np
from gym import spaces
import sionna as sn
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_agents.trajectories import time_step as ts

# Inicialization:
    
NUM_BITS_PER_SYMBOL = 4 # 16-QAM
N = 32  # Número de subportadoras OFDM
batch_size = 1000  # gera 1000 vezes N.
Np = 2
BLOCK_LENGTH = (N-Np)*NUM_BITS_PER_SYMBOL 

def create_matrix(batch_size, N):
    matrix = np.zeros((batch_size, N), dtype=int)
    for i in range(batch_size):
        matrix[i] = np.arange(0, N, 1)
    return matrix

def create_pilot(batch_size, Np):
    matrix = np.zeros((batch_size, Np), dtype=int)
    for i in range(batch_size):
        matrix[i] = np.array([1,N-2])
    return matrix

def pilot_value(batch_size, Np):
    matrix = np.zeros((batch_size, Np), dtype=int)
    for i in range(batch_size):
        matrix[i] = np.array([1,1])
    return matrix

# Bits and Mapper:
    
constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
constellation.show()
binary_source = sn.utils.BinarySource()
mapper = sn.mapping.Mapper(constellation = constellation)

bits = binary_source([batch_size, BLOCK_LENGTH])
print('bits:',bits)
x = mapper(bits)

#%% Creating the dataCarriers and pilotCarriers:
    
allCarriers = create_matrix(batch_size, N)
pilotCarriers = create_pilot(batch_size, Np)
dataCarriers = np.delete(allCarriers, pilotCarriers,axis=1)

#%% Allocate bits to dataCarriers and pilotCarriers:
    
symbol = np.zeros((batch_size, N))  # the overall N subcarriers
symbol[:, pilotCarriers] = pilot_value(batch_size, Np)  # assign values to pilots
symbol[np.arange(batch_size)[:, None], dataCarriers] = x  # assign values to datacarriers
#symbol[:, dataCarriers] = bits

#%% Simbolo OFDM:

OFDM_time = np.sqrt(N)*np.fft.ifft(symbol)


#%% Find PAPR:
    
idx = np.arange(0, batch_size)
PAPR = np.zeros(len(idx))

for i in idx:
    var = np.var(OFDM_time[i])
    print('var:',var)
    peakValue = np.max(abs(OFDM_time[i])**2)
    PAPR[i] = peakValue/var
    
PAPR_dB = 10*np.log10(PAPR)

#%% Find CCDF

PAPR_Total = len(PAPR_dB)

ma = max(PAPR_dB)
mi = min(PAPR_dB)

eixo_x = np.arange(mi, ma, 0.1)

y = []

for j in eixo_x:
    A = len(np.where(PAPR_dB > j)[0])/PAPR_Total
    y.append(A) #Adicionar A na lista y.
    
CCDF = y
 

#%% Reinforcement Learning Method:
    
def pilot_value_change(batch_size, Np, Pilot_1, Pilot_2):
    matrix = np.zeros((batch_size, Np), dtype=int)
    for i in range(batch_size):
        matrix[i] = np.array([Pilot_1, Pilot_2])
    return matrix

class PAPR_Reduction_Env(gym.Env):
    
    def __init__(self):
        
        super().__init__()
        # Define the action space based on the q_table's number of actions
        self.action_space = spaces.Discrete(21)  # Replace NUM_ACTIONS with the actual number of actions
        #self._episode_ended = False  # Initialize the attribute
        
    def _Bits(self, batch_size, BLOCK_LENGTH):
        binary_source = sn.utils.BinarySource()
        return binary_source([batch_size, BLOCK_LENGTH])

    def _Modulation(self, bits):
        constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
        mapper = sn.mapping.Mapper(constellation = constellation)
        return mapper(bits)
    
    def _symbol(self, x, Pilot_1, Pilot_2):
        allCarriers = create_matrix(batch_size, N)
        pilotCarriers = create_pilot(batch_size, Np)
        dataCarriers = np.delete(allCarriers, pilotCarriers,axis=1)       
        # Allocate bits to dataCarriers and pilotCarriers:            
        symbol = np.zeros((batch_size, N))  # the overall N subcarriers
        symbol[:, pilotCarriers] = pilot_value_change(batch_size, Np, Pilot_1, Pilot_2)  # assign values to pilots
        symbol[np.arange(batch_size)[:, None], dataCarriers] = x  # assign values to datacarriers
        return symbol
    
    def _IFFT(self, symbol):
        OFDM_time = np.sqrt(N) * np.fft.ifft(symbol)
        return OFDM_time

    
    def Step_Action(self, PAPR_dB_ref, PAPR_dB_red, state):
        reward = 0    
        PAPR_dB_red_nov = np.zeros(len(PAPR_dB_red))
        E = 1
        
        step = np.sqrt(E) / 10
        
        for i in range(10):
            Pilot_1 = -np.sqrt(E) + i * step
            Pilot_2 = Pilot_1 + step
            print(f'Pilot_1: {Pilot_1}, Pilot_2: {Pilot_2}')
            for idx in range(0, batch_size):
                if PAPR_dB_ref < PAPR_dB_red[idx]:
                    reward = -1
                    info = env._IFFT(env._symbol(env._Modulation(env._Bits(batch_size, BLOCK_LENGTH)), Pilot_1, Pilot_2))
                    PAPR_dB_red = env._PAPR(info)
                    print('reward:', reward)
                    print('Pmax_red:', PAPR_dB_red[idx])
                    print('Pmax_ref:', PAPR_dB_ref)
                    print('PAPR_dB_red_nov:', PAPR_dB_red_nov)
                    
                else:
                    reward = 1              
                    PAPR_dB_red_nov[idx] = PAPR_dB_red[idx] 
                    print('PAPR_dB_red_nov:', PAPR_dB_red_nov)
                    print('reward:', reward)
                    print('Pmax_ref:', PAPR_dB_ref)
                    
                next_state = state + 1
                print('next_state:', next_state)
                print('idx:', idx)
        return next_state, reward, PAPR_dB_red_nov

    
    def _PAPR(self, info):
        idy = np.arange(0, batch_size)
        PAPR_red = np.zeros(len(idy))
    
        for i in idy:
            var_red = np.var(info[i])
            peakValue_red = np.max(abs(info[i])**2)
            PAPR_red[i] = peakValue_red / var_red
    
        PAPR_dB_red = 10 * np.log10(PAPR_red)
        return PAPR_dB_red


    def _CCDF(self, PAPR_dB_red):
        PAPR_Total_red = len(PAPR_dB_red)
        ma_red = max(PAPR_dB_red)
        mi_red = min(PAPR_dB_red)
        eixo_x_red = np.arange(mi_red, ma_red, 0.1)
        y_red = []
        for jj in eixo_x_red:
            A_red = len(np.where(PAPR_dB_red > jj)[0])/PAPR_Total_red
            y_red.append(A_red) #Adicionar A na lista y.       
        CCDF_red = y_red
        return CCDF_red
    
class QLearningAgent:
    def __init__(self, env, learning_rate=0.8, discount_factor=0.95, exploration_prob=1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.num_states = 2
        self.num_actions = 2  # Number of actions in the action space
        self.q_table = np.zeros((self.num_states, self.num_actions))

#    def _discretize_state(self, state):
#        normalized_state = (state - 1001) / 1001
#        return int(np.clip(np.floor(normalized_state * self.num_states), 0, self.num_states - 1))


    def get_action(self, state):
        state_row = state
        if np.random.uniform(0, 1) < self.exploration_prob:
            # Choose a random action with probability of exploration_prob
            return np.random.choice(self.num_actions)
        else:
            # Choose the best action from the Q-table
            print('ENTREI')
            return np.argmax(self.q_table[state_row, :])

    def update_q_table(self, state, action, next_state, reward):
        
        self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
                                           self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[next_state, :]))
    

        print('state:', state)
        print('next_state:', next_state)
        print('q-table:', self.q_table)
        
    def train(self, num_episodes):
        rewards = []
        max_env_steps = 1
        epsilon_min = 0.01
        epsilon_decay = 0.999
        
        # Initialize Pmax_red once per episode
        info = env._IFFT(env._symbol(env._Modulation(env._Bits(batch_size, BLOCK_LENGTH)), -1, 1))
        PAPR_dB_red = env._PAPR(info)
        PAPR_dB_ref = ma - 3
        state = 0
        for episode in range(num_episodes):
            total_reward = 0
            
            for time in range(max_env_steps):
                action = self.get_action(state)
                next_state, reward, PAPR_dB_red_nov = env.Step_Action(PAPR_dB_ref, PAPR_dB_red, state)
                          
                # Update the environment state before updating the Q-table
                state = next_state
                print('state:', state)
    
                # Update the Q-table
                self.update_q_table(state, action, next_state, reward)
                
                
                total_reward += reward

            rewards.append(total_reward)
            
            if self.exploration_prob > epsilon_min:
                self.exploration_prob *= epsilon_decay
            print('exploration:', self.exploration_prob)
            print('episodes:', num_episodes)
        return rewards, PAPR_dB_red_nov
        

env = PAPR_Reduction_Env()
agent = QLearningAgent(env)
rewards, PAPR_dB_red_nov = agent.train(num_episodes=2)
print("Training rewards:", rewards)

CCDF_red = env._CCDF(PAPR_dB_red_nov)

# Plot the figure
plt.figure(figsize=(10, 8))
plt.plot(eixo_x, CCDF,'x-', c=f'C0', linewidth=2.5,label="Original Signal")
plt.plot(np.arange(min(PAPR_dB_red_nov), max(PAPR_dB_red_nov), 0.1), CCDF_red,'o-', c=f'C1', linewidth=2.5,label="RL Signal")
plt.legend(loc="lower left")
plt.xlabel('PAPR (dB)')
plt.ylabel('CCDF')
plt.title('PAPR x CCDF')
plt.yscale('log') 
plt.ylim([1e-2, 1])
plt.grid(True)
plt.show() 


