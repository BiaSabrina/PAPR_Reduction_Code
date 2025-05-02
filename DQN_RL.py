import gym
import numpy as np
from collections import deque
from gym import spaces
import sionna as sn
import matplotlib.pyplot as plt
import keras
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
        matrix[i] = np.array([-1,-1])
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
Pmax_ref = np.max(abs(OFDM_time) ** 2)


#%% Find PAPR:
    
idx = np.arange(0, 1000)
PAPR = np.zeros(len(idx))

for i in idx:
    var = np.var(OFDM_time[i])
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
        OFDM_time = np.sqrt(N) * tf.signal.ifft(tf.complex(np.real(symbol),np.imag(symbol)))
        return OFDM_time

    
    def Step_Action(self, action):
        
        if action == 1:  # Compare the maximum Pmax_red with Pmax_ref
            reward = -10
            Pilot_1 = np.random.uniform(-1, 1) #1 e -1 é para garantir a vaga.
            Pilot_2 = np.random.uniform(-1, 1)
            while Pilot_1 == 0 or Pilot_2 == 0:  # Change the logical operator to 'or'
                Pilot_1 = np.random.uniform(-1, 1)
                Pilot_2 = np.random.uniform(-1, 1)
            #self._episode_ended = True  # Episode ended due to PAPR constraint violation
            print('reward:', reward)
            print('Pilot_1:', Pilot_1)
            print('Pilot_2:', Pilot_2)
            info = env._IFFT(env._symbol(env._Modulation(env._Bits(batch_size, BLOCK_LENGTH)), Pilot_1, Pilot_2))
            done = True
            return next_state, reward, done, info
        else:
            Pilot_1 = np.random.uniform(-1, 1)
            Pilot_2 = np.random.uniform(-1, 1)
            reward = 10
            print('reward:', reward)
            print('Pilot_1:', Pilot_1)
            print('Pilot_2:', Pilot_2)
            info = env._IFFT(env._symbol(env._Modulation(env._Bits(batch_size, BLOCK_LENGTH)), Pilot_1, Pilot_2))
            done = True
            return next_state, reward, done, info

    
    def _reset(self):
       return ts.restart(np.array([0], dtype=np.int32))

    
    def _PAPR(self, OFDM_time):
        idx = np.arange(0, 1000)
        PAPR = np.zeros(len(idx))

        for i in idx:
            #var = np.var(OFDM_time[i])
            peakValue = np.max(abs(OFDM_time[i])**2)
            PAPR[i] = peakValue/0.6
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
    def papr(y_true,y_pred):
        x = tf.square(tf.abs(y_pred))
        return 10*tf.experimental.numpy.log10(tf.reduce_mean(tf.reduce_max(x, axis = 1) / tf.reduce_mean(x, axis = 1)))


env = PAPR_Reduction_Env()

input_shape = [4] # == env.observation_space.shape
n_outputs = 2 # == env.action_space.n
state = 0
next_state = 0
model = keras.models.Sequential([
 keras.layers.Dense(32, activation="elu", input_shape=input_shape),
 keras.layers.Dense(32, activation="elu"),
 keras.layers.Dense(n_outputs)
])

def epsilon_greedy_policy(state, epsilon=0):
 if np.random.rand() < epsilon:
    return np.random.randint(2)
 else:
     print('state:',state)
     print('model',model) 
     Q_values = model.predict([state])
     return np.argmax(Q_values[0])

#Instead of training the DQN based only on the latest experiences, we will store all
#experiences in a replay buffer(or replay memory):
    
replay_buffer = deque(maxlen=2000)

def sample_experiences(batch_size):
 indices = np.random.randint(len(replay_buffer), size=batch_size)
 batch = [replay_buffer[index] for index in indices]
 states, actions, rewards, next_states, dones = [
 np.array([experience[field_index] for experience in batch])
 for field_index in range(5)]
 return states, actions, rewards, next_states, dones

def play_one_step(env, state, epsilon):
 state = 0
 action = epsilon_greedy_policy(state, epsilon)
 next_state, reward, done, info = env.Step_Action(action)
 replay_buffer.append((state, action, reward, next_state, done))
 return next_state, reward, done, info

batch_size = 32
discount_factor = 0.95
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error

def training_step(batch_size):
 experiences = sample_experiences(batch_size)
 states, actions, rewards, next_states, dones = experiences
 next_Q_values = model.predict(next_states)
 max_next_Q_values = np.max(next_Q_values, axis=1)
 target_Q_values = (rewards + (1 - dones) * discount_factor * max_next_Q_values)
 mask = tf.one_hot(actions, n_outputs)
 with tf.GradientTape() as tape:
     all_Q_values = model(states)
     Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
     loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
 grads = tape.gradient(loss, model.trainable_variables)
 optimizer.apply_gradients(zip(grads, model.trainable_variables))
 
for episode in range(600):
    obs = env.reset()
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break
    if episode > 50:
       training_step(batch_size)

PAPR_dB_red = env._PAPR(info)
CCDF_red = env._CCDF(PAPR_dB_red)

# Plot the figure
plt.figure(figsize=(10, 8))
plt.plot(eixo_x, CCDF,'x-', c=f'C0', linewidth=2.5,label="Original Signal")
plt.plot(np.arange(min(PAPR_dB_red), max(PAPR_dB_red), 0.1), CCDF_red,'o-', c=f'C1', linewidth=2.5,label="RL Signal")
plt.legend(loc="lower left")
plt.xlabel('PAPR (dB)')
plt.ylabel('CCDF')
plt.title('PAPR x CCDF')
plt.yscale('log') 
plt.ylim([1e-2, 1])
plt.grid(True)
plt.show()  
