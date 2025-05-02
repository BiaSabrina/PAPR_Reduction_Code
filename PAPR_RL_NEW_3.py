import gym
import random
import numpy as np
from gym import spaces
import sionna as sn
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_agents.trajectories import time_step as ts

# Inicialization:
    
NUM_BITS_PER_SYMBOL = 4 # 16-QAM
N = 2048  # Número de subportadoras OFDM
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

OFDM_time_ref = np.sqrt(N)*np.fft.ifft(symbol)
Pmax_ref = np.max(abs(OFDM_time_ref) ** 2)


#%% Find PAPR:
    
idx = np.arange(0, 1000)
PAPR = np.zeros(len(idx))

for i in idx:
    var = np.var(OFDM_time_ref[i])
    peakValue = np.max(abs(OFDM_time_ref[i])**2)
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
        self.Pmax_ref = np.max(abs(OFDM_time_ref) ** 2)
       # Define the observation space based on your environment's state characteristics
        # Example: if your state is a vector of length s_size, you can define the observation space as follows
        # Modify the above line according to your observation space characteristics
        
       
        
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

    
    def Step_Action(self, action):
        
        if action == 1: # Pref < Pred
            reward = -1
            Pilot_1 = np.random.uniform(-1, 1) 
            Pilot_2 = np.random.uniform(-1, 1)
            while Pilot_1 == 0 or Pilot_2 == 0:  # Change the logical operator to 'or'
                Pilot_1 = np.random.uniform(-1, 1)
                Pilot_2 = np.random.uniform(-1, 1)
            #self._episode_ended = True  # Episode ended due to PAPR constraint violation
            print('reward:', reward)
            print('Pilot_1:', Pilot_1)
            print('Pilot_2:', Pilot_2)
            info = env._IFFT(env._symbol(env._Modulation(env._Bits(batch_size, BLOCK_LENGTH)), Pilot_1, Pilot_2))
            next_state = state + 1
            done = True
            return next_state, reward, done, info
        
        if action == 2: # Pref > Pred
            Pilot_1 = np.random.uniform(-1, 1) 
            Pilot_2 = np.random.uniform(-1, 1)
            reward = 1
            print('reward:', reward)
            print('Pilot_1:', Pilot_1)
            print('Pilot_2:', Pilot_2)
            info = env._IFFT(env._symbol(env._Modulation(env._Bits(batch_size, BLOCK_LENGTH)), Pilot_1, Pilot_2))
            next_state = state + 1
            done = True
            return next_state, reward, done, info
        

    
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
    
# Create the custom environment
env = PAPR_Reduction_Env()

# Inicialização com a tabela de valores Q
q_table = np.zeros([2, 10])

# Hiperparâmetros
alpha = 0.1   # taxa de aprendizagem
gamma = 0.6   # fator de desconto
epsilon = 0.1  # chance de escolha aleatória  

# Total geral de ações executadas e penalidades recebidas durante a aprendizagem
epochs, penalties = 0,0

for i in range(1, 100): # Vai rodar 100000 diferentes versões do problema
    state = 0 
    next_state = 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.Step_Action(state) # Escolhe ação aleatoriamente
        else:
            action = np.argmax(q_table[state]) # Escolhe ação com base no que já aprendeu

        next_state, reward, done, info = env.Step_Action(action) # Aplica a ação
        
        old_value = q_table[action, state]  # Valor da ação escolhida no estado atual
        next_max = np.max(q_table[state]) # Melhor valor no próximo estado
        
        # Atualize o valor Q usando a fórmula principal do Q-Learning
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[action, state] = new_value
        print('q_table:', q_table)

        if reward == -1:  # Contabiliza as punições por pegar ou deixar no lugar errado
            penalties += 1
            
        state = next_state # Muda de estado
        epochs += 1
        print('interacoes:', i)
        

print("Total de ações executadas: {}".format(epochs))
print("Total de penalizações recebidas: {}".format(penalties))

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
