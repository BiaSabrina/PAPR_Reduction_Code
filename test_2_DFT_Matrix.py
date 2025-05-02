import numpy as np
import sionna as sn
import matplotlib.pyplot as plt
from scipy import special
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow import keras
import time
#from keras import regularizers
from keras.optimizers import Adam
#import tensorflow.experimental.numpy as tnp
#from keras.layers import Dense, Dropout, BatchNormalization, Activation, Input , Lambda, Flatten, Layer, LeakyReLU
from keras.layers import Dense, Flatten
from tensorflow.keras import Sequential
#from mpl_toolkits.mplot3d import Axes3D


NUM_BITS_PER_SYMBOL = int(input("Number of Bits per Symbol: "))
Np = int(input("Pilots Number: "))
N = int(input("Carriers Number: "))

M = 2**NUM_BITS_PER_SYMBOL # Ordem da modulação QAM.

Ntx=1
#batch = Ntx  # 1 símbolo por rodada.
BLOCK_LENGTH = (N)*NUM_BITS_PER_SYMBOL

Eo = 1 # Menor energia de símbolo da constelação.

E = (2/3)*(M-1)*Eo # Energia de símbolo da constelação.

constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL) 
constellation.show()

cont = 0

#%% Matrizes para alocar as pilotos e carriers:
    
def create_matrix(batch, N):    
    matrix = np.zeros((batch, N), dtype=int)
    for i in range(batch):
        matrix[i] = np.arange(0, N, 1)
    return matrix

def create_pilot(batch, Np):
    allocation_position = N // Np
    matrix = np.zeros((batch, Np), dtype=int)
    for i in range(batch):
        matrix[i] = np.arange(0, Np * allocation_position, allocation_position)
    return matrix

def pilot_value_change(batch, Np, pilots):
    matrix = np.zeros((batch, Np), dtype=complex)
    for i in range(batch):
        for j in range(Np): 
            matrix[i, j] = pilots[j]
    return matrix

#%% Geração de bits e modulação:
    
def Bits(batch, BL):
    binary_source = sn.utils.BinarySource()
    return binary_source([batch, BL])

def Modulation(bits):
    constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL, normalize = True)
    mapper = sn.mapping.Mapper(constellation = constellation)
    return mapper(bits)

#%% 
    
def symbol(x_, pilots):
    allCarriers = create_matrix(int(np.size(x_) / (N - Np)), N)
    pilotCarriers = create_pilot(int(np.size(x_) / (N - Np)), Np)
    dataCarriers = np.delete(allCarriers, pilotCarriers, axis=1)     
    symbol = np.zeros((int(np.size(x_) / (N - Np)), N), dtype=complex)
    pilots_values = pilot_value_change(int(np.size(x_) / (N - Np)), Np, pilots)
    symbol[:, pilotCarriers] = pilots_values
    symbol[np.arange(int(np.size(x_) / (N - Np)))[:, None], dataCarriers] = x_
    return symbol

def symbol_Ori(x):
    allCarriers = create_matrix(int(np.size(x)/(N)), N)           
    symbols = np.zeros((int(np.size(x)/(N)), N), dtype=complex)  # the overall N subcarriers
    symbols[np.arange(int(np.size(x)/(N)))[:, None], allCarriers] = x # assign values to datacarriers
    return symbols

def FFT(symbol_):
    OFDM_time = 1/(np.sqrt(N)) * np.fft.fft(symbol_).astype('complex64')    
    return OFDM_time
    
def IFFT(symbol):
    OFDM_time__ = np.sqrt(N) * np.fft.ifft(symbol).astype('complex64')   
    return OFDM_time__

def FFTOri(symbol_):
    OFDM_time = 1/(np.sqrt(N-Np)) * np.fft.fft(symbol_).astype('complex64')    
    return OFDM_time
    
def IFFTOri(symbol):
    OFDM_time__ = np.sqrt(N-Np) * np.fft.ifft(symbol).astype('complex64')   
    return OFDM_time__

#%% Cálculo da PAPR e CCDF:
    
def PAPR(info):
    idy = np.arange(0, info.shape[0]) 
    PAPR_red = np.zeros(len(idy)) 
    for i in idy: 
        var_red = np.mean(abs(info[i])**2) 
        peakValue_red = np.max(abs(info[i])**2) 
        PAPR_red[i] = peakValue_red / var_red

    PAPR_dB_red = 10 * np.log10(PAPR_red) 
    return PAPR_dB_red

def PAPR_DFT(info):
    var_red = np.mean(abs(info)**2)  # Potência média do sinal inteiro
    peakValue_red = np.max(abs(info)**2)  # Pico do sinal
    PAPR_dB_red = 10 * np.log10(peakValue_red / var_red)  # Retorna um escalar
    return PAPR_dB_red

def CCDF(PAPR_final):
    PAPR_Total_red = PAPR_final.size 
    mi = min(PAPR_final)
    ma = max(PAPR_final)
    eixo_x_red = np.arange(mi, ma, 0.0001) 
    y_red = []
    for jj in eixo_x_red:
        A_red = len(np.where(PAPR_final > jj)[0])/PAPR_Total_red
        y_red.append(A_red)    
    CCDF_red = y_red
    return CCDF_red

#%%

bits = Bits(Ntx, BLOCK_LENGTH)
mod = Modulation(bits)
#symbol__ = symbol_Ori(mod)
OFDM__ = IFFT(mod)
_PAPR_dB = PAPR(OFDM__)
_CCDF = CCDF(_PAPR_dB)    

#%% DFT


#for iii in range(Ntx):
    
# Partition OFDM Symbol
P1_DFT = np.concatenate([mod[:, 0:16]])
P2_DFT = np.concatenate([mod[:, 16:32]])
P3_DFT = np.concatenate([mod[:, 32:48]])
P4_DFT = np.concatenate([mod[:, 48:64]])

Pt1 = (np.fft.fft(P1_DFT))
Pt2 = (np.fft.fft(P2_DFT))
Pt3 = (np.fft.fft(P3_DFT))
Pt4 = (np.fft.fft(P4_DFT))

F_DFT = np.concatenate((Pt1, Pt2, Pt3, Pt4), axis=1)
OFDM_DFT = np.fft.ifft(F_DFT)

PAPR_dB_DFT = PAPR(OFDM_DFT)
    
CCDF_final_DFT = CCDF(PAPR_dB_DFT)    

#print("Original Symbol:", symbol__)
print("Transmitted OFDM:", OFDM_DFT)

fig, ax = plt.subplots(figsize=(10, 8))    
ax.semilogy(np.arange(min(_PAPR_dB), max(_PAPR_dB), 0.0001), _CCDF, '-', color='C2', label=f'PAPR Original', linewidth=2.5)
ax.semilogy(np.arange(min(PAPR_dB_DFT), max(PAPR_dB_DFT), 0.0001), CCDF_final_DFT, '<', color='C4', label='DFT Spread', linewidth=2.5)
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

class UncodedSystemAWGN_DFT(Model): 
    def __init__(self, num_bits_per_symbol, block_length,Subcarriers):

        super().__init__() 

        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = BLOCK_LENGTH
        self.N = Subcarriers
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol, normalize = True)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
        
    def __call__(self, batch_size, ebno_db):
        
        h = np.array([1])
        self.H = np.fft.fft(h,self.N)  
        
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        self.OFDM_RX_FD_DFT = OFDM_DFT
        # Channel     
        _y_DFT = self.awgn_channel([self.OFDM_RX_FD_DFT, no]) # no = potência do ruído        
    
        FFT_DFT = np.fft.fft(_y_DFT)
        
        #ofdm_symbol_DFT_reconstructed = np.zeros((Ntx,N), dtype=complex)
        
        #for iiii in range(Ntx):
            
        P1_DFT_rec = np.concatenate([FFT_DFT[:,0:16]])
        P2_DFT_rec = np.concatenate([FFT_DFT[:,16:32]])
        P3_DFT_rec = np.concatenate([FFT_DFT[:,32:48]])
        P4_DFT_rec = np.concatenate([FFT_DFT[:,48:64]])
        
        Pt1_rec = (np.fft.ifft(P1_DFT_rec))
        Pt2_rec = (np.fft.ifft(P2_DFT_rec))
        Pt3_rec = (np.fft.ifft(P3_DFT_rec))
        Pt4_rec = (np.fft.ifft(P4_DFT_rec)) 
        
        # Reconstruct the OFDM Symbol
        ofdm_symbol_DFT_reconstructed = np.concatenate((Pt1_rec, Pt2_rec, Pt3_rec, Pt4_rec), axis=1)
        
        
        ofdm_symbol_DFT_reconstructed = tf.cast(ofdm_symbol_DFT_reconstructed, tf.complex64)
        no = tf.cast(no, tf.float32)
        
        print("Received OFDM:", _y_DFT)
        print("Reconstructed Symbol:", ofdm_symbol_DFT_reconstructed)

        llr_DFT = self.demapper([ofdm_symbol_DFT_reconstructed, no])
    
        return bits, llr_DFT

model_uncoded_awgn_DFT = UncodedSystemAWGN_DFT(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, 
                                               Subcarriers=N)

SNR = np.arange(0, 15)

EBN0_DB_MIN = min(SNR) # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = max(SNR) # Maximum value of Eb/N0 [dB] for simulations

# DFT Simulation:
    
ber_plots_DFT = sn.utils.PlotBER()
ber_plots_DFT.simulate(model_uncoded_awgn_DFT,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size = Ntx,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=1000, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM_DFT = np.array(ber_plots_DFT.ber).ravel()
    
#%% Theoretical:

M = 2**(NUM_BITS_PER_SYMBOL)
L = np.sqrt(M)
mu = 4 * (L - 1) / L  # Número médio de vizinhos
Es = 3 / (L ** 2 - 1) # Fator de ajuste da constelação
    
ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)
BER_THEO = np.zeros((len(ebno_dbs)))
BER_THEO_des = np.zeros((len(ebno_dbs)))

i = 0
for idx in ebno_dbs:   
    #BER_THEO_des[i] = (mu/(2*np.log2(M)))*special.erfc(np.sqrt(((N-Np)/N)*Es*NUM_BITS_PER_SYMBOL*10**(idx/10))/np.sqrt(2))
    BER_THEO[i] = (mu/(2*np.log2(M)))*special.erfc(np.sqrt(Es*NUM_BITS_PER_SYMBOL*10**(idx/10))/np.sqrt(2))
    i = i+1

fig, ax = plt.subplots(figsize=(10, 8)) 
ax.plot(ebno_dbs, BER_THEO, '-', label=f'Original Theoretical')
#ax.plot(ebno_dbs, BER_THEO_des, '-', color='C7', label='MCSA Theoretical', linewidth=2) 
ax.plot(ebno_dbs, BER_SIM_DFT, '<', markersize=5, color='C4', label='DFT Spread')    
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

