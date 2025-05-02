import numpy as np
import sionna as sn
import matplotlib.pyplot as plt
import gym, time, csv
from scipy import special
from tensorflow.keras import Model
from scipy.stats import norm, poisson, cauchy, expon
from scipy import stats

NUM_BITS_PER_SYMBOL = 4 
M = 2**NUM_BITS_PER_SYMBOL
N = 32  # Número de subportadoras OFDM
batch_size = 1  # gera 1000 vezes N.
Ntx=10000 #número de símbolos OFDM que serão gerados
Np = 2
BLOCK_LENGTH = (N-Np)*NUM_BITS_PER_SYMBOL

#E = 10 # alterar para a expressão genérica.
Eo = 1

E = (2/3)*(M-1)*Eo 

call_count_RL = 0 

#%% Matrizes para alocar as pilotos e carriers:
    
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

def pilot_value_change(batch_size, Np, Pilot_5, Pilot_6):
    matrix = np.zeros((batch_size, Np), dtype=complex)
    for i in range(batch_size):
        matrix[i] = np.array([Pilot_5, Pilot_6])
    return matrix

#%% Geração de bits e modulação:
    
def Bits(batch_size, BLOCK_LENGTH):
    binary_source = sn.utils.BinarySource()
    return binary_source([batch_size, BLOCK_LENGTH])

def Modulation(bits):
    constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
    mapper = sn.mapping.Mapper(constellation = constellation)
    constellation.show()
    return mapper(bits)

#%% Alocação de pilotos e sinais modulados no simbolo que será transmitido:
    
def symbol(x, Pilot_5, Pilot_6):
    allCarriers = create_matrix(int(np.size(x)/(N-Np)), N)
    pilotCarriers = create_pilot(int(np.size(x)/(N-Np)), Np)
    dataCarriers = np.delete(allCarriers, pilotCarriers,axis=1)            
    symbol = np.zeros((int(np.size(x)/(N-Np)), N), dtype=complex)  # the overall N subcarriers
    symbol[:, pilotCarriers] = pilot_value_change(int(np.size(x)/(N-Np)), Np, Pilot_5, Pilot_6)  # assign values to pilots
    symbol[np.arange(int(np.size(x)/(N-Np)))[:, None], dataCarriers] = x # assign values to datacarriers
    return symbol

#%% Cálculo da IFFT:
    
def IFFT(symbol):
    global call_count_RL
    call_count_RL += 1
    OFDM_ti = np.sqrt(N) * np.fft.ifft(symbol).astype('complex64') 
    return OFDM_ti

#%% Cálculo da PAPR e CCDF:
    
def PAPR(info):
    idy = np.arange(0, info.shape[0]) #O(1)
    PAPR_red = np.zeros(len(idy)) #O(1)  
    for i in idy: #O(M)
        var_red = np.mean(abs(info[i])**2) #O(NN)
        peakValue_red = np.max(abs(info[i])**2) #O(NN)
        PAPR_red[i] = peakValue_red / var_red

    PAPR_dB_red = 10 * np.log10(PAPR_red) #O(M)
    return PAPR_dB_red

def CCDF(PAPR_final):
    PAPR_Total_red = PAPR_final.size #O(1)
    mi = min(PAPR_final)
    ma = max(PAPR_final)
    eixo_x_red = np.arange(mi, ma, 0.01) #O(1)
    y_red = []
    for jj in eixo_x_red: #O(Q)
        A_red = len(np.where(PAPR_final > jj)[0])/PAPR_Total_red #O(log2(M))
        y_red.append(A_red) #O(Q)     
    CCDF_red = y_red
    return CCDF_red

bits = Bits(batch_size, BLOCK_LENGTH)
mapper = Modulation(bits)

#%% Reinforcement Learning Method:

call_count_RL = 0

start_timeRL = time.time()
   
class PAPR_Reduction_Env(gym.Env): #The class keyword is used to define a class in object-oriented programming (OOP).
    # gym.Env: OpenAI Gym's environment class.
    
    def __init__(self): # The self convention is used in Python to reference the instance of the class itself.
        
        super().__init__() # It means that the child class is invoking the constructor of its parent class. 
        
    def Step_Action(self, PAPR_dB_ref, PAPR_dB_red, state, ind_complex_RL, ic_RL, OFDM_time, symbol_Ori, reward, count, Pilot_3, Pilot_4):    
        
        if PAPR_dB_red < PAPR_dB_ref:                
            reward += 1
            count +=1
            
        while PAPR_dB_red > PAPR_dB_ref:       
            reward += -1
            Pilot_3 = np.sqrt(E)*(np.random.randn() + 1j*np.random.randn()) 
            Pilot_4 = np.sqrt(E)*(np.random.randn() + 1j*np.random.randn())
            symbol_Ori = symbol(mapper, Pilot_3, Pilot_4)
            OFDM_time = IFFT(symbol_Ori)
            PAPR_dB_red = PAPR(OFDM_time)
            ic_RL = ic_RL + 1
                                  
        if state == 3:
            next_state = 0
        else:
            next_state = min(state + 1, 3) 
            
        return next_state, reward, OFDM_time, PAPR_dB_red, symbol_Ori, ic_RL, count, Pilot_3, Pilot_4
    
class QLearningAgent:
    def __init__(self, env, learning_rate=0.6, discount_factor=0.8, exploration_prob=1):
        self.env = env # Environment
        self.learning_rate = learning_rate # Controls how quickly the agent updates Q values.
        self.discount_factor = discount_factor # Represents how important future reward is relative to immediate reward.
        self.exploration_prob = exploration_prob # Regulates the transition between the exploration and exploitation stages.
        self.num_states = 4    
        self.num_actions = 4 
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.successful_pilots = []
        
    
       
    def get_action(self, state, Pilot_3, Pilot_4):
        
        if np.random.uniform(0, 1) < self.exploration_prob:  # Exploration
            Pilot_3 = -np.sqrt(E)
            Pilot_4 = np.sqrt(E)
            action = np.random.choice(self.num_actions)
            return Pilot_3, Pilot_4, action
       
        else:  # Exploitation
            Pilot_3, Pilot_4 = self.successful_pilots[np.random.choice(range(len(self.successful_pilots)))]
            action = np.argmax(self.q_table[state, :])
            return Pilot_3, Pilot_4, action
 
    def update_q_table(self, state, action, next_state, reward): 
        
        self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
            self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[next_state, :]))
        
        print('Q-Table:', self.q_table)
        
    def train(self):
        
        global mapper  
        
        rewards = []
        epsilon_min = 0.001
        epsilon_decay = 0.999
        
        PAPR_dB_ref = 4
        
        OFDM_RL = np.zeros([Ntx,N], complex)
        symbol_RL = np.zeros([Ntx,N], complex)
        PAPR_RL = np.zeros((Ntx,1))
        ind_complex_RL= np.zeros((Ntx,1)) # Frequency of encountering negative rewards.
        state = 0
        count = 0     
        Pilot_3 = 0
        Pilot_4 = 0
        
        for nt_Rl in range(0,Ntx):                   
            ic_RL=1 # Assumes the nt_Rl position in the ind_complex_RL vector.       
            reward = 0 
            Pilot_3, Pilot_4, action = self.get_action(state, Pilot_3, Pilot_4) 
            symbol_Ori = symbol(mapper, Pilot_3, Pilot_4)
            OFDM_time = IFFT(symbol_Ori) 
            PAPR_dB = PAPR(OFDM_time) 
            PAPR_dB_red = PAPR_dB
            next_state, reward, OFDM_time, PAPR_dB_red, symbol_Ori, ic_RL, count, Pilot_3, Pilot_4 = env.Step_Action(PAPR_dB_ref, 
                                                  PAPR_dB_red, state, ind_complex_RL, ic_RL,
                                                  OFDM_time, symbol_Ori, reward, count, Pilot_3, Pilot_4)   
            self.successful_pilots.append((Pilot_3, Pilot_4))
                
            self.update_q_table(state, action, next_state, reward)   
            state = next_state
            rewards.append(reward)   
            
            if self.exploration_prob > epsilon_min: 
                self.exploration_prob *= epsilon_decay   
            print('exploration_prob:', self.exploration_prob)
            
            PAPR_RL[nt_Rl]=PAPR_dB_red
            OFDM_RL[nt_Rl]=OFDM_time
            symbol_RL[nt_Rl]=symbol_Ori
            ind_complex_RL[nt_Rl]=ic_RL
            
        successful_pilots = self.successful_pilots   
        return rewards, PAPR_RL, symbol_RL, OFDM_RL, ind_complex_RL, OFDM_time, count, successful_pilots  
        
env = PAPR_Reduction_Env()
agent = QLearningAgent(env)
rewards, PAPR_RL, symbol_RL, OFDM_RL, ind_complex_RL, OFDM_time, count, successful_pilots = agent.train()
_CCDF_red = CCDF(PAPR_RL)

#%% Histogram:

filtered_test = [value for value in ind_complex_RL if value != 1]

hist, edges = np.histogram(filtered_test, bins='auto')
x = (edges[:-1] + edges[1:]) / 2  # Midpoints of the bins
pdf_mean = expon.pdf(x, scale = np.mean(filtered_test))
pdf_std = expon.pdf(x, scale = np.std(filtered_test))

fig = plt.figure(figsize=(10, 8))
plt.plot(x, hist/len(filtered_test), label='Histogram', drawstyle='steps-mid', alpha=0.7)
plt.plot(x, pdf_mean / np.sum(pdf_mean), label='Exponential PDF (mean)')
plt.plot(x, pdf_std / np.sum(pdf_std), label='Exponential PDF (std)')
plt.legend()
plt.show()

#%% Complexity: 

filtered_test_new = np.array(filtered_test)

filtered_test_IFFT = filtered_test_new * N * np.log10(N)

hist_IFFT, edges_IFFT = np.histogram(filtered_test_IFFT, bins='auto')
x_IFFT = (edges_IFFT[:-1] + edges_IFFT[1:]) / 2  # Midpoints of the bins
pdf_mean_IFFT = expon.pdf(x_IFFT, scale = np.mean(filtered_test_IFFT))
pdf_std_IFFT = expon.pdf(x_IFFT, scale = np.std(filtered_test_IFFT))

fig = plt.figure(figsize=(10, 8))
plt.plot(x_IFFT, hist_IFFT/len(filtered_test_IFFT), label='Histogram', drawstyle='steps-mid', alpha=0.7)
plt.plot(x_IFFT, pdf_mean_IFFT / np.sum(pdf_mean_IFFT), label='Exponential PDF (mean)')
plt.plot(x_IFFT, pdf_std_IFFT / np.sum(pdf_std_IFFT), label='Exponential PDF (std)')
plt.legend()
plt.show()

#%% Both:
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))

# Plotting the first figure
axes[0].plot(x, hist/len(filtered_test), label='Histogram', drawstyle='steps-mid', alpha=0.7)
axes[0].plot(x, pdf_mean/np.sum(pdf_mean), label='Exponential PDF (mean)')
axes[0].plot(x, pdf_std/np.sum(pdf_std), label='Exponential PDF (std)')
axes[0].legend()
axes[0].set_title('Histogram and Exponential PDFs')
axes[1].plot(x_IFFT, hist_IFFT/len(filtered_test_IFFT), label='Histogram', drawstyle='steps-mid', alpha=0.7)
axes[1].plot(x_IFFT, pdf_mean_IFFT/np.sum(pdf_mean_IFFT), label='Exponential PDF (mean)')
axes[1].plot(x_IFFT, pdf_std_IFFT/np.sum(pdf_std_IFFT), label='Exponential PDF (std)')
axes[1].legend()
axes[1].set_title('Histogram and Exponential PDFs (IFFT)')
plt.show()

#%% Plot 

     
#bits = Bits(Ntx, BLOCK_LENGTH)
#mapper = Modulation(bits)
#symbol_Ori_ = symbol(mapper, -np.sqrt(E), np.sqrt(E))
#OFDM_time_ = IFFT(symbol_Ori_)
#_PAPR_dB = PAPR(OFDM_time_)
#_CCDF = CCDF(_PAPR_dB)   
       
fig = plt.figure(figsize=(10, 8))
#plt.semilogy(np.arange(min(_PAPR_dB), max(_PAPR_dB), 0.01), _CCDF,'-', c=f'C0',label="Original Signal")
plt.semilogy(np.arange(min(PAPR_RL), max(PAPR_RL), 0.01), _CCDF_red,'o', c=f'C1',label="GA Signal")
plt.xlabel('PAPR (dB)', fontsize=17, fontweight='bold')
plt.ylabel('CCDF', fontsize=17, fontweight='bold')
plt.legend(loc="lower left")
plt.ylim([1e-3, 1])
plt.grid()
plt.show()
        
#np.savetxt('/Users/bianc/OneDrive/Documentos/symbol_Ori.csv', symbol_Ori_, delimiter=';', fmt='%.4f')   
np.savetxt('/Users/bianc/OneDrive/Documentos/symbol_RL.csv', symbol_RL, delimiter=';', fmt='%.4f')
np.savetxt('/Users/bianc/OneDrive/Documentos/ind_complex_RL.csv', ind_complex_RL, delimiter=';', fmt='%.4f')
