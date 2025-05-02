import numpy as np
import sionna as sn
import matplotlib.pyplot as plt
import gym, time, csv
from scipy import special
from tensorflow.keras import Model

NUM_BITS_PER_SYMBOL = 4 
N = 32  # Número de subportadoras OFDM
batch_size = 1000  # gera 1000 vezes N.
Np = 2
BLOCK_LENGTH = (N-Np)*NUM_BITS_PER_SYMBOL

call_count_GA = 0 
call_count_rl = 0

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
    return mapper(bits)

#%% Alocação de pilotos e sinais modulados no simbolo que será transmitido:
    
def symbol(x, Pilot_5, Pilot_6):
    allCarriers = create_matrix(batch_size, N)
    pilotCarriers = create_pilot(batch_size, Np)
    dataCarriers = np.delete(allCarriers, pilotCarriers,axis=1)            
    symbol = np.zeros((int(np.size(x)/(N-Np)), N), dtype=complex)  # the overall N subcarriers
    symbol[:, pilotCarriers] = pilot_value_change(int(np.size(x)/(N-Np)), Np, Pilot_5, Pilot_6)  # assign values to pilots
    symbol[np.arange(int(np.size(x)/(N-Np)))[:, None], dataCarriers] = x # assign values to datacarriers
    return symbol

#%% Cálculo da IFFT:
    
def IFFT(symbol):
    global call_count_rl
    global call_count_GA
    call_count_rl += 1
    call_count_GA += 1
    OFDM_time = np.sqrt(N) * np.fft.ifft(symbol).astype('complex64')    
    return OFDM_time

#%% Cálculo da PAPR e CCDF:
    
def PAPR(info):
    idy = np.arange(0, batch_size) #O(1)
    PAPR_red = np.zeros(len(idy)) #O(1)  
    for i in idy: #O(M)
        var_red = np.mean(abs(info[i])**2) #O(NN)
        peakValue_red = np.max(abs(info[i])**2) #O(NN)
        PAPR_red[i] = peakValue_red / var_red

    PAPR_dB_red = 10 * np.log10(PAPR_red) #O(M)
    return PAPR_dB_red

def CCDF(PAPR_dB_red):
    PAPR_Total_red = len(PAPR_dB_red) #O(1)
    ma_red = max(PAPR_dB_red) #O(R)
    mi_red = min(PAPR_dB_red) #O(R)
    eixo_x_red = np.arange(mi_red, ma_red, 0.01) #O(1)
    y_red = []
    for jj in eixo_x_red: #O(Q)
        A_red = len(np.where(PAPR_dB_red > jj)[0])/PAPR_Total_red #O(log2(M))
        y_red.append(A_red) #O(Q)     
    CCDF_red = y_red
    return CCDF_red

bits = Bits(batch_size, BLOCK_LENGTH)
mapper = Modulation(bits)
symbol_Ori = symbol(mapper, -1, 1)
OFDM_time = IFFT(symbol_Ori)
PAPR_dB = PAPR(OFDM_time)
_CCDF = CCDF(PAPR_dB) 

#%% Reinforcement Learning Method:

start_timeRL = time.time()

call_count_rl = 0 
   
class PAPR_Reduction_Env(gym.Env):
    
    def __init__(self):
        
        super().__init__()
    
    def Step_Action(self, PAPR_dB_ref, _PAPR_dB_red, state, PAPR_dB_red_nov, Pilot_3, Pilot_4, symbol_RL):                      
        total_reward = 0
        reward = 0 
        for idx in range(0, batch_size): # O(M)      
            if PAPR_dB_ref > _PAPR_dB_red[idx]:  # O(1)
                reward = 1
                PAPR_dB_red_nov[idx] = _PAPR_dB_red[idx] 
                
            if PAPR_dB_red_nov[idx] == 0: # O(1)
                reward = -1
                symbol_RL[idx] = symbol(mapper[idx], Pilot_3, Pilot_4)
            total_reward += reward
        
        OFDM_time_RL = IFFT(symbol_RL)  #Nlog2(N)       
        _PAPR_dB_red = PAPR(OFDM_time_RL) # O(M*N**2)​
        next_state = min(state + 1, 3) #O(I)
        return next_state, PAPR_dB_red_nov, total_reward, symbol_RL, OFDM_time_RL, _PAPR_dB_red
    
class QLearningAgent:
    def __init__(self, env, learning_rate=0.8, discount_factor=0.95, exploration_prob=1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.num_states = 4    
        self.num_actions = 4 
        self.q_table = np.zeros((self.num_states, self.num_actions))

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_prob: #O(1)
            return np.random.choice(self.num_actions) #O(1)
        else:
            return np.argmax(self.q_table[state, :]) #O(V)
        
    def update_q_table(self, state, action, next_state, reward):     
        
        self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
                                           self.learning_rate * (reward + self.discount_factor * 
                                                                 np.max(self.q_table[next_state, :]))        
        #O(V), where V is the max complexity.
        
    def train(self, done):
        global mapper       
        rewards = []
        epsilon_min = 0.01
        epsilon_decay = 0.9
        
        _PAPR_dB_red = PAPR_dB
        PAPR_dB_ref = 7
        PAPR_dB_red_nov = np.zeros(len(_PAPR_dB_red)) # O(1)
        
        Pilot_3_values = []
        Pilot_4_values = []
        
        symbol_RL = symbol(mapper, -1, 1)
        
        while not done:
            state = 0
            for Y in range(4): # O(A), where A is the size.     
                Pilot_3 = np.random.randn() # O(1)
                Pilot_4 = np.random.randn() # O(1)
             
                while Pilot_3 == 0 or Pilot_4 == 0: # O(1)
                    Pilot_3 = np.random.randn() # O(1)
                    Pilot_4 = np.random.randn() # O(1)
                    
                Pilot_3_values.append(Pilot_3) #O(P) 
                Pilot_4_values.append(Pilot_4) #O(P)  
                      
                action = self.get_action(state) # O(V)
                next_state, PAPR_dB_red_nov, reward, symbol_RL, OFDM_time_RL, _PAPR_dB_red = env.Step_Action(PAPR_dB_ref, 
                                                      _PAPR_dB_red, state, PAPR_dB_red_nov, Pilot_3, Pilot_4, symbol_RL)
                print('PAPR_dB_red_nov:',PAPR_dB_red_nov)
                self.update_q_table(state, action, next_state, reward) #O(V)    
                state = next_state
                rewards.append(reward) #O(U)              
            if self.exploration_prob > epsilon_min: # O(1)
                self.exploration_prob *= epsilon_decay
            if np.all(PAPR_dB_red_nov != 0): # O(M)
                done = True
        return rewards, PAPR_dB_red_nov, PAPR_dB, Pilot_3_values, Pilot_4_values, symbol_RL, OFDM_time_RL
        
env = PAPR_Reduction_Env()
agent = QLearningAgent(env)
rewards, PAPR_dB_red_nov, PAPR_dB, Pilot_3_values, Pilot_4_values, symbol_RL, OFDM_time_RL = agent.train(done = False)
_CCDF_red = CCDF(PAPR_dB_red_nov)

#%% Complexidade do RL:

Comp_RL = call_count_rl
detection_timesRL = []
compRL_2 = []

M = batch_size
A = 4

for n in np.arange(0, Comp_RL):
    comp_value_RL_1 = (M**2)*A*(1+(N**2)) + n*(A*M*N*np.log2(N))
    compRL_2.append(comp_value_RL_1)        
    elapsed_timeRL = (time.time() - start_timeRL)
    detection_timesRL.append(elapsed_timeRL) 
    
#detection_timesRL = detection_timesRL[:len(compRL_2)]  
#compRL_2 = compRL_2[:len(n)] 
        
#%% GA Method:

start_timeGA = time.time()

call_count_GA = 0

symbol_GA = symbol(mapper, -1, 1)
   
PAPR_dB_ref = 7
_PAPR_dB = PAPR_dB
PAPR_dB_red = np.zeros(len(_PAPR_dB)) # O(1)

Pilot_1_values = []
Pilot_2_values = []

done =  False

while not done:
    
    Pilot_1 = np.random.randn() # O(1)
    Pilot_2 = np.random.randn() # O(1)
 
    while Pilot_1 == 0 or Pilot_2 == 0: # O(1)
        Pilot_1 = np.random.randn() # O(1)
        Pilot_2 = np.random.randn() # O(1)
        
    Pilot_1_values.append(Pilot_1) #O(P)
    Pilot_2_values.append(Pilot_2) #O(P)
    
    for idy in range(0, batch_size): # O(M)      
        if PAPR_dB_ref > _PAPR_dB[idy]: # O(1)     
            PAPR_dB_red[idy] = _PAPR_dB[idy] 
            print('PAPR_dB_red:', PAPR_dB_red)
            
        if PAPR_dB_red[idy] == 0:   # O(1) 
            symbol_GA[idy] = symbol(mapper[idy], Pilot_1, Pilot_2)
            
    OFDM_time_GA = IFFT(symbol_GA) # O(Nlog2(N))
    _PAPR_dB = PAPR(OFDM_time_GA) # O(MN**2)
            
    if np.all(PAPR_dB_red != 0): # O(M)
        done = True
            
CCDF_red = CCDF(PAPR_dB_red) 
  

#%% Complexidade do GA:
 
Comp_GA = call_count_GA
detection_timesGA = []
compGA_2 = []

for nn in np.arange(0, Comp_GA):
    comp_value_GA_1 =  (M**2)*(1+(N**2)) + nn*(M*N*np.log2(N))
    compGA_2.append(comp_value_GA_1)        
    elapsed_timeGA = (time.time() - start_timeGA)
    detection_timesGA.append(elapsed_timeGA)

#detection_timesGA = detection_timesGA[:len(compGA_2)]    
#compGA_2 = compGA_2[:len(nn)]

#%% Plot da complexidade:
    
#Mesma figura, porém um método do lado do outro:
    
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), sharex=True)
ax2.plot(np.arange(0, Comp_RL), compRL_2, 'blue', label='RL', marker='o')
ax1.plot(np.arange(0, Comp_GA), compGA_2, 'orange', label='GA', marker='x')
ax1.set_xlabel('IFFT Iterations (n)', fontsize=17, fontweight='bold')
ax1.set_ylabel('Complexity', fontsize=17, fontweight='bold')
ax2.set_xlabel('IFFT Iterations (n)', fontsize=17, fontweight='bold')
ax2.grid(True)
ax1.grid(True)
ax1.legend()
ax2.legend()
plt.show()

#Ambos os métodos na mesma figura, um sobreposto do outro:
    
fig, ax = plt.subplots(figsize=(10, 8), sharex=True)
ax.plot(np.arange(0, Comp_RL), compRL_2, 'blue', label='RL', marker='o')
ax.plot(np.arange(0, Comp_GA), compGA_2, 'orange',label='GA', marker='x')
ax.set_xlabel('IFFT Iterations (n)', fontsize=17, fontweight='bold')
ax.set_ylabel('Complexity', fontsize=17, fontweight='bold')
ax.grid(True)
ax.legend()
plt.show()

#%% 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), sharex=True)
ax2.plot(np.arange(0, Comp_RL), detection_timesRL, 'blue', label='RL', marker='o')
ax1.plot(np.arange(0, Comp_GA), detection_timesGA, 'orange', label='GA', marker='x')
ax1.set_xlabel('IFFT Iterations (n)', fontsize=17, fontweight='bold')
ax1.set_ylabel('Execution Time (s)', fontsize=17, fontweight='bold')
ax2.set_xlabel('IFFT Iterations (n)', fontsize=17, fontweight='bold')
ax2.grid(True)
ax1.grid(True)
ax1.legend()
ax2.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10, 8), sharex=True)
ax.plot(np.arange(0, Comp_RL), detection_timesRL, 'blue', label='RL', marker='o')
ax.plot(np.arange(0, Comp_GA), detection_timesGA, 'orange',label='GA', marker='x')
ax.set_xlabel('IFFT Iterations (n)', fontsize=17, fontweight='bold')
ax.set_ylabel('Execution Time (s)', fontsize=17, fontweight='bold')
ax.grid(True)
ax.legend()
plt.show()

#%% Plot da PAPR:
    
fig = plt.figure(figsize=(10, 8))
plt.semilogy(np.arange(min(PAPR_dB), max(PAPR_dB), 0.01), _CCDF,'-', c=f'C0',label="Original Signal")
plt.plot(np.arange(min(PAPR_dB_red), max(PAPR_dB_red), 0.01), CCDF_red,'o', c=f'C1',label="GA Signal")
plt.plot(np.arange(min(PAPR_dB_red_nov), max(PAPR_dB_red_nov), 0.01), _CCDF_red,'x', c=f'C2',label="RL Signal")
plt.xlabel('PAPR (dB)', fontsize=17, fontweight='bold')
plt.ylabel('CCDF', fontsize=17, fontweight='bold')
plt.legend(loc="lower left")
plt.ylim([1e-3, 1])
plt.grid()
plt.show()

#%% Valores máximo e mínimo das pilotos de saída de cada método:
    
max_Pilot_1 = max(Pilot_1_values)
min_Pilot_1 = min(Pilot_1_values)

max_Pilot_2 = max(Pilot_2_values)
min_Pilot_2 = min(Pilot_2_values)

max_Pilot_3 = max(Pilot_3_values)
min_Pilot_3 = min(Pilot_3_values)

max_Pilot_4 = max(Pilot_4_values)
min_Pilot_4 = min(Pilot_4_values)

# Salve os valores em um arquivo CSV
with open('valores.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Escreva os cabeçalhos das colunas
    writer.writerow(['Variável', 'Valores', 'Máximo', 'Mínimo'])
    
    # Escreva os valores máximo e mínimo para cada variável
    writer.writerow(['Pilot_1', Pilot_1_values, max_Pilot_1, min_Pilot_1])
    writer.writerow(['Pilot_2', Pilot_2_values, max_Pilot_2, min_Pilot_2])
    writer.writerow(['Pilot_3', Pilot_3_values, max_Pilot_3, min_Pilot_3])
    writer.writerow(['Pilot_4', Pilot_4_values, max_Pilot_4, min_Pilot_4])

print("Valores salvos com sucesso no arquivo 'valores.csv'.")


#%% Valores de saída símbolo, IFFT e PAPR para o GA, RL e Sinal original:
 
np.savetxt('/Users/bianc/OneDrive/Documentos/symbol_Ori.csv', symbol_Ori, delimiter=';', fmt='%.4f')
np.savetxt('/Users/bianc/OneDrive/Documentos/symbol_GA.csv', symbol_GA, delimiter=';', fmt='%.4f')
np.savetxt('/Users/bianc/OneDrive/Documentos/symbol_RL.csv', symbol_RL, delimiter=';', fmt='%.4f')

np.savetxt('/Users/bianc/OneDrive/Documentos/OFDM_time.csv', OFDM_time, delimiter=';', fmt='%.4f')
np.savetxt('/Users/bianc/OneDrive/Documentos/OFDM_time_GA.csv', OFDM_time_GA, delimiter=';', fmt='%.4f')
np.savetxt('/Users/bianc/OneDrive/Documentos/OFDM_time_RL.csv', OFDM_time_RL, delimiter=';', fmt='%.4f')

np.savetxt('/Users/bianc/OneDrive/Documentos/PAPR_Original.csv', PAPR_dB, delimiter=';', fmt='%.4f')
np.savetxt('/Users/bianc/OneDrive/Documentos/PAPR_GA.csv', PAPR_dB_red, delimiter=';', fmt='%.4f')
np.savetxt('/Users/bianc/OneDrive/Documentos/PAPR_RL.csv', PAPR_dB_red_nov, delimiter=';', fmt='%.4f')
    
#%% Comparação de Sinais OFDM na saída de cada método:

R = 100e6

tempo_ms = np.linspace(0, 1/R*(N-1), N)
tempo_us = tempo_ms * 1e6

fig, ax = plt.subplots(figsize=(15, 10), sharex=True)

#dices_diff = np.where((OFDM_time[0, :] != OFDM_time_GA[0, :]) & (OFDM_time[0, :] != OFDM_time_RL[0, :]) & (OFDM_time_GA[0, :] != OFDM_time_RL[0, :]))


# Plot stem plots in the first subplot (ax1)
ax.stem(tempo_us, OFDM_time[25, :], 'black', label="Original Signal", markerfmt='ko')
ax.stem(tempo_us, OFDM_time_GA[25, :], 'red', label="GA Signal", markerfmt='ro')
ax.stem(tempo_us, OFDM_time_RL[25, :], 'blue', label="RL Signal", markerfmt='o', linefmt='--')
ax.set_ylabel('Amplitude', fontsize=17, fontweight='bold')
ax.legend(loc="upper right")
ax.grid(True)
# Adjust layout
plt.tight_layout()
plt.show()

#%% Generate SNR x BER:

class UncodedSystemAWGN(Model): 
    def __init__(self, num_bits_per_symbol, block_length,Subcarriers):

        super().__init__() 

        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = BLOCK_LENGTH
        self.N = Subcarriers
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
        
    def __call__(self, batch_size, ebno_db):
        
        global OFDM_time        
        #global symbol_Ori
        global bits
        
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        pilotCarriers = create_pilot(batch_size, Np)
        # Channel
        h = np.array([1])
        self.H = np.fft.fft(h,self.N)        
        self.OFDM_RX_FD = OFDM_time
        y = self.awgn_channel([self.OFDM_RX_FD, no]) # no = potência do ruído
        y= (np.sqrt(1/N)*np.fft.fft(y)).astype('complex64')
        OFDM_demod = np.delete(y, pilotCarriers,axis=1)
        llr = self.demapper([OFDM_demod,no])
        
        return bits, llr

class UncodedSystemAWGN_GA(Model): 
    def __init__(self, num_bits_per_symbol, block_length,Subcarriers):

        super().__init__() 

        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = BLOCK_LENGTH
        self.N = Subcarriers
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
        
    def __call__(self, batch_size, ebno_db):

        global OFDM_time_GA     
        global bits
        
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        pilotCarriers = create_pilot(batch_size, Np)        
        # Channel
        h = np.array([1])        
        self.H = np.fft.fft(h,self.N)        
        self.OFDM_RX_FD = OFDM_time_GA
        y = self.awgn_channel([self.OFDM_RX_FD, no]) # no = potência do ruído        
        y= (np.sqrt(1/N)*np.fft.fft(y)).astype('complex64')
        OFDM_demod = np.delete(y, pilotCarriers,axis=1)
        llr_GA = self.demapper([OFDM_demod,no])
        
        return bits, llr_GA
   
class UncodedSystemAWGN_RL(Model): 
    def __init__(self, num_bits_per_symbol, block_length,Subcarriers):

        super().__init__() 

        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = BLOCK_LENGTH
        self.N = Subcarriers
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
        
    def __call__(self, batch_size, ebno_db):

        global OFDM_time_RL
        #global symbol_GA
        global bits
        
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        pilotCarriers = create_pilot(batch_size, Np)
        # Channel
        h = np.array([1])        
        self.H = np.fft.fft(h,self.N)   
        self.OFDM_RX_FD = OFDM_time_RL
        y = self.awgn_channel([self.OFDM_RX_FD, no]) # no = potência do ruído
        y= (np.sqrt(1/N)*np.fft.fft(y)).astype('complex64')
        OFDM_demod = np.delete(y, pilotCarriers,axis=1)
        llr_RL = self.demapper([OFDM_demod,no])
        
        return bits, llr_RL
    
#%%
  
# Instanciando o modelo
model_uncoded_awgn = UncodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, Subcarriers=N)
model_uncoded_awgn_GA = UncodedSystemAWGN_GA(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, Subcarriers=N)
model_uncoded_awgn_RL = UncodedSystemAWGN_RL(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, Subcarriers=N)

#%%

# Sionna provides a utility to easily compute and plot the bit error rate (BER).
SNR = np.arange(0, 10)

EBN0_DB_MIN = min(SNR) # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = max(SNR) # Maximum value of Eb/N0 [dB] for simulations

# Original Simulation:

ber_plots = sn.utils.PlotBER()
ber_plots.simulate(model_uncoded_awgn,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=batch_size,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM = np.array(ber_plots.ber).ravel()

# GA Simulation:
    
ber_plots_GA = sn.utils.PlotBER()
ber_plots_GA.simulate(model_uncoded_awgn_GA,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=batch_size,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM_GA = np.array(ber_plots_GA.ber).ravel()

# RL Simulation:

ber_plots_RL = sn.utils.PlotBER()
ber_plots_RL.simulate(model_uncoded_awgn_RL,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=batch_size,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM_RL = np.array(ber_plots_RL.ber).ravel()


#%% Theoretical:

M = 2**(NUM_BITS_PER_SYMBOL)
L = np.sqrt(M)
mu = 4 * (L - 1) / L  # Número médio de vizinhos
E = 3 / (L ** 2 - 1) # Fator de ajuste da constelação
    
ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)
BER_THEO = np.zeros((len(ebno_dbs)))

i = 0
for idx in ebno_dbs:
    BER_THEO[i] = (mu/(2*N*NUM_BITS_PER_SYMBOL))*np.sum(special.erfc(np.sqrt(np.abs(model_uncoded_awgn.H) **2
                                                            *E*NUM_BITS_PER_SYMBOL*10**(idx/10)) / np.sqrt(2)))
    
    i = i+1

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(ebno_dbs, BER_SIM, 'o', markersize=5, c='C1', label='Original Signal')
ax.plot(ebno_dbs, BER_SIM_GA, 'x', markersize=5, c='C2', label='GA Signal')
ax.plot(ebno_dbs, BER_SIM_RL, '*', markersize=5, c='C3', label='RL Signal')
ax.plot(ebno_dbs, BER_THEO, '-', c='C0', label='Theoretical', linewidth=2)
ax.set_title('BER vs. SNR in AWGN Channel', fontsize=16, fontweight='bold')
ax.set_ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')
ax.set_xlabel('Eb/N0 (dB)', fontsize=14, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.legend(loc="upper right", fontsize=12)
ax.set_xlim([EBN0_DB_MIN, EBN0_DB_MAX])
ax.set_ylim([1e-3, 1])
ax.set_yscale('log')
plt.show()
