import numpy as np
import sionna as sn
import matplotlib.pyplot as plt
import gym, time, csv
from scipy import special
from tensorflow.keras import Model

NUM_BITS_PER_SYMBOL = 4 
N = 32  # Número de subportadoras OFDM
batch_size = 10  # gera 1000 vezes N.
Np = 2
BLOCK_LENGTH = (N-Np)*NUM_BITS_PER_SYMBOL

E = 10 # alterar para a expressão genérica.

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


PAPR_dB_ref = 6
_PAPR_dB = PAPR_dB
PAPR_dB_red = PAPR_dB
flag = False

Pilot_1_values = []
Pilot_2_values = []

while np.sum(PAPR_dB_red) > 0:
       
    Pilot_1 = np.sqrt(E)*(np.random.randn() + 1j*np.random.randn())
    Pilot_2 = np.sqrt(E)*(np.random.randn() + 1j*np.random.randn())
 
    while Pilot_1 == 0 or Pilot_2 == 0:
        Pilot_1 = np.sqrt(E)*(np.random.randn() + 1j*np.random.randn())
        Pilot_2 = np.sqrt(E)*(np.random.randn() + 1j*np.random.randn())
        
    Pilot_1_values.append(Pilot_1) 
    Pilot_2_values.append(Pilot_2) 
    
    for idy in range(0, batch_size):       
        if PAPR_dB_ref > _PAPR_dB[idy]:      
            PAPR_dB_red[idy] = 0 
            print('PAPR_dB_red:', PAPR_dB_red)
            
        if PAPR_dB_red[idy] != 0:    
            symbol_Ori[idy] = symbol(mapper[idy], Pilot_1, Pilot_2)
            #print('symbol_Ori:',symbol_Ori)
    
    OFDM_time = IFFT(symbol_Ori)
    _PAPR_dB = PAPR(OFDM_time)
            


#%% Plot 

np.savetxt('/Users/bianc/OneDrive/Documentos/symbol_Ori.csv', symbol_Ori, delimiter=';', fmt='%.4f')        


'''
CCDF = CCDF(PAPR_dB)   
CCDF_red = CCDF(PAPR_dB_red)
       
fig = plt.figure(figsize=(10, 8))
plt.semilogy(np.arange(min(PAPR_dB), max(PAPR_dB), 0.01), CCDF,'-', c=f'C0',label="Original Signal")
plt.plot(np.arange(min(PAPR_dB_red), max(PAPR_dB_red), 0.01), CCDF_red,'o', c=f'C1',label="GA Signal")
plt.xlabel('PAPR (dB)', fontsize=17, fontweight='bold')
plt.ylabel('CCDF', fontsize=17, fontweight='bold')
plt.legend(loc="lower left")
plt.ylim([1e-3, 1])
plt.grid()
plt.show()
'''        