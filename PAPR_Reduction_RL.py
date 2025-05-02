import gym
import numpy as np
from gym import spaces
import sionna as sn
import matplotlib.pyplot as plt
from tensorflow.keras import Model
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
'''
def pilot_value(batch_size, Np):
    matrix = np.zeros((batch_size, Np), dtype=int)
    for i in range(batch_size):
        matrix[i] = np.array([0.3,-0.3])
    return matrix

def Bits(batch_size, BLOCK_LENGTH):
    binary_source = sn.utils.BinarySource()
    return binary_source([batch_size, BLOCK_LENGTH])

def Modulation(bits):
    constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
    mapper = sn.mapping.Mapper(constellation = constellation)
    return mapper(bits)

def symbol(x):
    allCarriers = create_matrix(batch_size, N)
    pilotCarriers = create_pilot(batch_size, Np)
    dataCarriers = np.delete(allCarriers, pilotCarriers,axis=1)
    
    # Allocate bits to dataCarriers and pilotCarriers:
        
    symbol = np.zeros((batch_size, N))  # the overall N subcarriers
    symbol[:, pilotCarriers] = pilot_value(batch_size, Np)  # assign values to pilots
    symbol[np.arange(batch_size)[:, None], dataCarriers] = x  # assign values to datacarriers
    return symbol

def IFFT(self, symbol):
    OFDM_time = np.sqrt(N)*np.fft.ifft(symbol)
    Pmax_ref = np.max(abs(OFDM_time)**2)
    return OFDM_time, Pmax_ref

def PAPR(OFDM_time):
    idx = np.arange(0, 1000)
    PAPR = np.zeros(len(idx))

    for i in idx:
        var = np.var(OFDM_time[i])
        peakValue = np.max(abs(OFDM_time[i])**2)
        PAPR[i] = peakValue/var 
        PAPR_dB = 10*np.log10(PAPR)
    return PAPR_dB

def CCDF(PAPR_dB):
    PAPR_Total = len(PAPR_dB)
    ma = max(PAPR_dB)
    mi = min(PAPR_dB)
    eixo_x = np.arange(mi, ma, 0.1)
    y = []
    for j in eixo_x:
        A = len(np.where(PAPR_dB > j)[0])/PAPR_Total
        y.append(A) #Adicionar A na lista y.       
    CCDF = y
    return CCDF

bits = Bits(batch_size, BLOCK_LENGTH)
modulated_bits = Modulation(bits)
ofdm_time_signal = IFFT(symbol(modulated_bits))
papr_dB = PAPR(ofdm_time_signal)
ccdf = CCDF(papr_dB)
'''
#%% Training:
    
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
        self._episode_ended = False  # Initialize the attribute
        
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
        Pmax_red = np.max(abs(OFDM_time) ** 2)
        return OFDM_time, Pmax_red

    
    def Step_Action(self, Pmax_ref, Pmax_red):
        
       if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self._reset()
       
       if Pmax_red > Pmax_ref:
           reward = -1
           Pilot_1 = np.random.uniform(-1, 1)
           Pilot_2 = np.random.uniform(-1, 1)
           while Pilot_1 or Pilot_2 == 0:
               Pilot_1 = np.random.uniform(-1, 1)
               Pilot_2 = np.random.uniform(-1, 1)
           return Pilot_1, Pilot_2, reward
       else:
           reward = 1
           Pmax_ref = Pmax_red
           return Pmax_ref, reward
    
    def _reset(self):
       return ts.restart(np.array([0], dtype=np.int32))

    
    def _PAPR(self, OFDM_time):
        idx = np.arange(0, 1000)
        PAPR = np.zeros(len(idx))

        for i in idx:
            var = np.var(OFDM_time[i])
            peakValue = np.max(abs(OFDM_time[i])**2)
            PAPR[i] = peakValue/var 
            PAPR_dB = 10*np.log10(PAPR)
        return PAPR_dB

    def _CCDF(self, PAPR_dB):
        PAPR_Total = len(PAPR_dB)
        ma = max(PAPR_dB)
        mi = min(PAPR_dB)
        eixo_x = np.arange(mi, ma, 0.1)
        y = []
        for j in eixo_x:
            A = len(np.where(PAPR_dB > j)[0])/PAPR_Total
            y.append(A) #Adicionar A na lista y.       
        CCDF = y
        return CCDF

env = PAPR_Reduction_Env()
        
possible_actions = [1, 2]

def exploration_policy(state):
    return np.random.choice(possible_actions[state])

np.random.seed(42)

Q_values = np.full((3, 5), -np.inf)  # Change the size of the second axis to match the number of possible actions
for state in range(3):  # Iterate through the states
    for action in possible_actions:  # Iterate through the possible actions
        Q_values[state][action - 1] = 0  # Subtract 1 from action to account for 0-based indexing in NumPy


alpha0 = 0.05  # initial learning rate
decay = 0.005  # learning rate decay
gamma = 0.90  # discount factor
state = 0  # initial state
history2 = []  # Not shown in the book

papr_values = []
Pmax_ref = 0
Pmax_red = 0
Pilot_1 = 1
Pilot_2 = -1

for iteration in range(10):
    done = False
    history2.append(Q_values.copy())  # Not shown
    action = exploration_policy(state)
    if action in possible_actions:
        action -= 1  # Adjust action to 0-based indexing for Q_values array
    if action not in range(2):
        # Handle invalid actions
        continue
    reward, done = env.Step_Action(Pmax_ref, Pmax_red)
    if state == 'terminal':
        # Handle episode termination
        break
    if state == 'non-terminal':
        # Perform the IFFT calculation for non-terminal state
        bits = env._Bits(batch_size, BLOCK_LENGTH)
        modulated_bits = env._Modulation(bits)
        PAPR_dB = env._PAPR(env._IFFT(env._symbol(modulated_bits, Pilot_1, Pilot_2)))
        papr_values.extend(PAPR_dB)

# Calculate CCDF from the collected PAPR values
papr_values = np.array(papr_values)
eixo_x, ccdf_values = env._CCDF(papr_values)

#%%  Plot PAPR x CCDF

fig = plt.figure(figsize=(10, 8))
#plt.plot(np.arange(min(papr_dB), max(papr_dB), 0.1), ccdf,'x-', c=f'C0', linewidth=2.5,label="Original Signal")
plt.plot(eixo_x, ccdf_values,'o-.', c=f'C1', linewidth=2.5,label="RL Signal")
plt.legend(loc="lower left")
plt.xlabel('PAPR (dB)')
plt.ylabel('CCDF')
plt.yscale('log')  
plt.grid()
plt.ylim([1e-2, 1])