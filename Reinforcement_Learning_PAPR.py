import numpy as np
import sionna as sn
import matplotlib.pyplot as plt
import time
from scipy import special
from tensorflow.keras import Model


#%% Cria o ambiente:

cont = 0

class env_Transmission:
    def __init__(self):
        self.NUM_BITS_PER_SYMBOL = int(input("Number of Bits per Symbol: "))
        self.Np = int(input("Pilots Number: "))
        self.N = int(input("Carriers Number: "))
        self.M = 2 ** self.NUM_BITS_PER_SYMBOL
        self.Ntx = self.M**(self.N - self.Np)
        self.BLOCK_LENGTH = (self.N - self.Np) * self.NUM_BITS_PER_SYMBOL
        self.BLOCK_LENGTH_Ori = (self.N) * self.NUM_BITS_PER_SYMBOL
        self.Eo = 1
        self.E = (2 / 3) * (self.M - 1) * self.Eo
        self.constellation = sn.mapping.Constellation("qam", self.NUM_BITS_PER_SYMBOL)
        self.constellation.show()

    @staticmethod
    def create_matrix(batch, N):
        matrix = np.zeros((batch, N), dtype=int)
        for i in range(batch):
            matrix[i] = np.arange(0, N, 1)
        return matrix

    @staticmethod
    def create_pilot(batch, N, Np):
        allocation_position = N // Np
        matrix = np.zeros((batch, Np), dtype=int)
        for i in range(batch):
            matrix[i] = np.arange(0, Np * allocation_position, allocation_position)
        return matrix

    @staticmethod
    def pilot_value_change(batch, Np, pilots):
        matrix = np.zeros((batch, Np), dtype=complex)
        for i in range(batch):
            for j in range(Np):
                matrix[i, j] = pilots[j]
        return matrix

    def Bits(self, batch):
        binary_source = sn.utils.BinarySource()
        return binary_source([batch, self.BLOCK_LENGTH])
    
    def BitsOri(self, batch):
        binary_source = sn.utils.BinarySource()
        return binary_source([batch, self.BLOCK_LENGTH_Ori])

    def Modulation(self, bits):
        constellation = sn.mapping.Constellation("qam", self.NUM_BITS_PER_SYMBOL, normalize=True)
        mapper = sn.mapping.Mapper(constellation=constellation)
        return mapper(bits)

    def symbol(self, x_, pilots):
        allCarriers = self.create_matrix(int(np.size(x_) / (self.N - self.Np)), self.N)
        pilotCarriers = self.create_pilot(int(np.size(x_) / (self.N - self.Np)), self.N, self.Np)
        dataCarriers = np.delete(allCarriers, pilotCarriers, axis=1)
        symbol = np.zeros((int(np.size(x_) / (self.N - self.Np)), self.N), dtype=complex)
        pilots_values = self.pilot_value_change(int(np.size(x_) / (self.N - self.Np)), self.Np, pilots)
        symbol[:, pilotCarriers] = pilots_values
        symbol[np.arange(int(np.size(x_) / (self.N - self.Np)))[:, None], dataCarriers] = x_
        return symbol

    @staticmethod
    def FFT(symbol_):
        return 1 / (np.sqrt(symbol_.shape[1])) * np.fft.fft(symbol_).astype('complex64')

    @staticmethod
    def IFFT(symbol):
        return np.sqrt(symbol.shape[1]) * np.fft.ifft(symbol).astype('complex64')

    @staticmethod
    def PAPR(info):
        PAPR_red = np.max(abs(info) ** 2, axis=1) / np.mean(abs(info) ** 2, axis=1)
        PAPR_dB_red = 10 * np.log10(PAPR_red)
        return PAPR_dB_red

    @staticmethod
    def CCDF(PAPR_final):
        PAPR_Total_red = PAPR_final.size
        mi, ma = min(PAPR_final), max(PAPR_final)
        eixo_x_red = np.arange(mi, ma, 0.00001)
        CCDF_red = [len(np.where(PAPR_final > jj)[0]) / PAPR_Total_red for jj in eixo_x_red]
        return CCDF_red
     
    def PAPR_Original(self):
        bits = self.Bits(self.Ntx)
        mod = self.Modulation(bits)
        bits_Ori = self.BitsOri(self.Ntx)
        mod_ori = self.Modulation(bits_Ori)
        IFFT_Output = self.IFFT(mod_ori)
        PAPR_dB = self.PAPR(IFFT_Output)
        _CCDF = self.CCDF(PAPR_dB)
        return bits, bits_Ori, mod, IFFT_Output, PAPR_dB, _CCDF
    
    def papr_ccdf(self, mod_signal, action, Pilots):
        global cont 
        #sign_patterns = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        #pilot_allocation = (abs(np.real(pilots)) * sign_patterns[action][0]) + 1j * (abs(np.imag(pilots)) * sign_patterns[action][1])
        pilot_allocation = Pilots[action]
        Mod_Pil = self.symbol(mod_signal, pilot_allocation)
        OFDM_time = self.IFFT(Mod_Pil)
        _PAPR_dB = self.PAPR(OFDM_time)
        _CCDF = self.CCDF(_PAPR_dB)
        cont = cont + 1
        return _PAPR_dB, _CCDF, OFDM_time, Mod_Pil
    
    def step(self, reward, Papr_test, PAPR_ref, state, done, PAPR_anterior):
        """Função step para atualizar estado e calcular recompensa."""
        
        if Papr_test < PAPR_ref:
            reward = 1 
            if Papr_test < PAPR_anterior:
               reward = 5
        else:
            reward = -10

        next_state = state + 1
        if next_state == n_states:
            next_state = 0
            done = True
        
        return next_state, reward, done, Papr_test
    
    def reset(self):
        state = 0
        return state

# Execução do ambiente:
env = env_Transmission()

n_states = env.M**(env.N - env.Np)
n_actions = env.M**(env.Np)

# Hiperparâmetros
alpha = 0.9  # Taxa de aprendizado
gamma = 0.9  # Fator de desconto
epsilon = 1.0  # Exploração inicial
epsilon_min = 0.01
epsilon_decay = 0.999

n_episodes = 1000

OFDM_final = np.zeros((env.Ntx,env.N),dtype=complex)
symbol_final = np.zeros((env.Ntx,env.N),dtype=complex)
PAPR_final = np.zeros((env.Ntx,1))

#real_part = np.linspace(1, np.sqrt(env.E), n_actions)
#imag_part = np.linspace(-np.sqrt(env.E), np.sqrt(env.E), n_actions)

#Pilots = np.array([real_part + 1j * imag_part] * env.Np).T
#Pilots = np.array([np.random.choice([1, -1], size=n_actions) + 1j*np.random.choice([1, -1], size=n_actions)] * env.Np).T

n_states = env.M**(env.N - env.Np)
n_actions = env.M**(env.Np)


#real = np.random.choice([1, -1], size=(n_actions, env.Np))
#imag = np.random.choice([1, -1], size=(n_actions, env.Np))
Pilots = np.sqrt(env.E) * (np.random.randn(n_states, n_actions) + 1j * np.random.randn(n_states, n_actions))

bits, bits_Ori, mod, IFFT_Output, PAPR_dB, CCDF = env.PAPR_Original() # PAPR e CCDF sem redução nenhuma.

# Inicializa a Q-Table
q_table = np.zeros((n_states, n_actions))
reward = 0
list_reward = []

st = time.time()

All_PAPR = np.zeros((env.Ntx,1))
All_Pilots = np.zeros((env.Ntx,env.Np))
All_OFDM = np.zeros((env.Ntx,env.N),dtype=complex)
All_symbol = np.zeros((env.Ntx,env.N),dtype=complex)

# Treinamento
for episode in range(n_episodes):
    
    state = 0
    #state = np.random.choice([n_states])  # Começa em um estado aleatório
    
    done = False    
    tested_actions = set()  # Armazena ações testadas
       
    while not done: 
            Mod = mod[state]
            # Escolha da ação: Exploração vs Exploração
            if np.random.rand() < epsilon:
                action = np.random.choice(n_actions)  # Exploration
            else:
                action = np.argmax(q_table[state, :])  # Exploitation
            
            # Calcula a PAPR, CCDF, OFDM e símbolo para essa posição
            PAPR_test, CCDF_test, OFDM_time, symbol = env.papr_ccdf(Mod, action, Pilots)
            
            PAPR_ref = PAPR_dB[state]
            
            # Executa a ação
            next_state, reward, done, PAPR_test = env.step(reward, PAPR_test, PAPR_ref, state, done, PAPR_final[state])
    
            # Atualiza a Q-Table
            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
            
            if reward == 5:
                pass
            else:
                PAPR_final[state] = PAPR_test
                OFDM_final[state] = OFDM_time
                symbol_final[state] = symbol
                
            state = next_state
            
            list_reward.append(reward)
    
            print('reward:', reward)
            
            
                

    # Decaimento do epsilon após cada episódio
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print('episode:', episode)

    
    print('Epsilon:', epsilon)
    
CCDF_Final = env.CCDF(PAPR_final)
print("Q-Table Treinada:\n", q_table)

et = time.time()
elapsed_time = (et - st)/60
print('Execution time:', elapsed_time, 'minutes')

#%%

fig, ax = plt.subplots(figsize=(10, 8))    
ax.semilogy(np.arange(min(PAPR_dB), max(PAPR_dB), 0.00001), CCDF, '-', color='C2', label=f'PAPR Original', linewidth=2.5)
ax.semilogy(np.arange(min(PAPR_final), max(PAPR_final), 0.00001), CCDF_Final, '^', color='C1', label=f'Reinforcement Learning', linewidth=2.5)
ax.set_xlabel('PAPR (dB)', fontsize=17, fontweight='bold')
ax.set_ylabel('CCDF', fontsize=17, fontweight='bold')
ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.7, color='gray')
ax.yaxis.grid(True, which='minor', linestyle='--', alpha=0.5, color='gray')
ax.grid(axis='both', linestyle='--', alpha=0.7, color='gray')
ax.set_facecolor('white')
ax.legend(loc='upper right', fontsize=17, bbox_to_anchor=(1.0, 1.0), frameon=True, facecolor='white', edgecolor='black')
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_ylim([1e-3, 1])
plt.savefig('PAPR.pdf', bbox_inches='tight', dpi=300)
plt.show()


#%% SNR x BER

#%% BER Original:
    
demapper = sn.mapping.Demapper("app", constellation=env.constellation)
awgn_channel = sn.channel.AWGN()

ebno_db = 10

no = sn.utils.ebnodb2no(ebno_db,
                        num_bits_per_symbol=env.NUM_BITS_PER_SYMBOL,
                        coderate=1.0)
# Channel     
#Received_signal = IFFT_Output
#y = awgn_channel([Received_signal, no]) # no = potência do ruído
#y_= env.FFT(y)      
#llr = demapper([y_,no])     
    
def BER_Original(bits_Ori):
    def model(batch_size, ebno_db):
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=env.NUM_BITS_PER_SYMBOL,
                                coderate=1.0)
        
        y = awgn_channel([IFFT_Output, no])
        y_ = env.FFT(y)
        llr = demapper([y_, no])
        
        return bits_Ori, llr  # Isso que o PlotBER espera
    return model

model_uncoded_awgn = BER_Original(bits_Ori)


SNR = np.arange(0, 15)

EBN0_DB_MIN = min(SNR) # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = max(SNR) # Maximum value of Eb/N0 [dB] for simulations

# Original Simulation:

ber_plots = sn.utils.PlotBER()
ber_plots.simulate(model_uncoded_awgn,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size = env.Ntx,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=1000, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM = np.array(ber_plots.ber).ravel()

#%% BER Q_Learning:
    
def BER_Q_Learning(bits):
    def modell(batch_size, ebno_db):
        # Converte Eb/N0 para potência do ruído
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=env.NUM_BITS_PER_SYMBOL,
                                coderate=1.0)
        
        # Pilotos e canal
        pilotCarriers = env.create_pilot(env.Ntx, env.N, env.Np)
        OFDM_RX_FD_Pil = OFDM_final  # sua entrada de canal
        y_Pil = awgn_channel([OFDM_RX_FD_Pil, no * (env.N / (env.N - env.Np))])
        
        # FFT e remoção dos pilotos
        y_Pil_fft = env.FFT(y_Pil)
        y_without_pilots = np.delete(y_Pil_fft, pilotCarriers, axis=1)

        # Demapeamento
        llr_pil = demapper([y_without_pilots, no])
        
        return bits, llr_pil  # usado no cálculo de BER
    return modell

model_uncoded_awgn_pilots = BER_Q_Learning(bits)

ber_plots_Pil = sn.utils.PlotBER()
ber_plots_Pil.simulate(model_uncoded_awgn_pilots,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size = env.Ntx,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=False,
                  max_mc_iter=1000, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM_Pil = np.array(ber_plots_Pil.ber).ravel()

    
#%% Theoretical:

M = 2**(env.NUM_BITS_PER_SYMBOL)
L = np.sqrt(M)
mu = 4 * (L - 1) / L  # Número médio de vizinhos
Es = 3 / (L ** 2 - 1) # Fator de ajuste da constelação
    
ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)
BER_THEO = np.zeros((len(ebno_dbs)))
BER_THEO_des = np.zeros((len(ebno_dbs)))

i = 0
for idx in ebno_dbs:
    BER_THEO_des[i] = (mu/(2*np.log2(env.M)))*special.erfc(np.sqrt(((env.N-env.Np)/env.N)*Es*env.NUM_BITS_PER_SYMBOL*10**(idx/10))/np.sqrt(2))
    BER_THEO[i] = (mu/(2*np.log2(env.M)))*special.erfc(np.sqrt(Es*env.NUM_BITS_PER_SYMBOL*10**(idx/10))/np.sqrt(2))
    i = i+1

fig, ax = plt.subplots(figsize=(10, 8)) 
ax.plot(ebno_dbs, BER_THEO, '-', label=f'Original Theory')
ax.plot(ebno_dbs, BER_THEO_des, '--', color='C7', label='Displaced Theory', linewidth=2)
ax.plot(ebno_dbs, BER_SIM, '*', color='C2', label='Simulation Original', linewidth=2) 
ax.plot(ebno_dbs, BER_SIM_Pil, '^', label=f'Q-Learning') 
ax.set_ylabel('Bit Error Rate (BER)', fontsize=16, fontweight='bold')
ax.set_xlabel('Eb/N0 (dB)', fontsize=16, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=17)
ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.7, color='gray')
ax.yaxis.grid(True, which='minor', linestyle='--', alpha=0.5, color='gray')
ax.grid(axis='both', linestyle='--', alpha=0.7, color='gray')
ax.set_facecolor('white')
ax.legend(loc='upper right', fontsize=17, bbox_to_anchor=(1.0, 1.0), frameon=True, facecolor='white', edgecolor='black')
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_xlim([EBN0_DB_MIN, EBN0_DB_MAX])
ax.set_ylim([1e-5, 1])
ax.set_yscale('log')
plt.savefig('BER.pdf', bbox_inches='tight', dpi=300)
plt.show()
