import numpy as np
import sionna as sn
import matplotlib.pyplot as plt
import gym, time, csv
from scipy import special
import tensorflow as tf
from scipy.stats import norm, poisson, cauchy, expon
from scipy import stats
from tensorflow.keras import Model
import matplotlib.mlab as mlab
from matplotlib.mlab import psd
from numpy.fft import fftshift
from scipy.signal import correlate
from scipy import signal
import random


NUM_BITS_PER_SYMBOL = 6 
M = 2**NUM_BITS_PER_SYMBOL
N = 64 # Número de subportadoras OFDM
batch = 1  # gera 1000 vezes N.
Ntx=600 #número de símbolos OFDM que serão gerados
Np = 2
BLOCK_LENGTH = (N-Np)*NUM_BITS_PER_SYMBOL

Eo = 1

E = (2/3)*(M-1)*Eo 

constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
constellation.show()

#%% Matrizes para alocar as pilotos e carriers:
    
def create_matrix(batch, N):
    matrix = np.zeros((batch, N), dtype=int)
    for i in range(batch):
        matrix[i] = np.arange(0, N, 1)
    return matrix

def create_pilot(batch, Np):
    matrix = np.zeros((batch, Np), dtype=int)
    for i in range(batch):
        matrix[i] = np.array([1,N-2])
    return matrix

def pilot_value_change(batch, Np, Pilot_5, Pilot_6):
    matrix = np.zeros((batch, Np), dtype=complex)
    for i in range(batch):
        matrix[i] = np.array([Pilot_5, Pilot_6])
    return matrix

#%% Geração de bits e modulação:
    
def Bits(batch, BLOCK_LENGTH):
    binary_source = sn.utils.BinarySource()
    return binary_source([batch, BLOCK_LENGTH])

def Modulation(bits):
    constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
    mapper = sn.mapping.Mapper(constellation = constellation)
    return mapper(bits)

#%% 
    
def symbol(x, Pilot_5, Pilot_6):
    allCarriers = create_matrix(int(np.size(x)/(N-Np)), N)
    pilotCarriers = create_pilot(int(np.size(x)/(N-Np)), Np)
    dataCarriers = np.delete(allCarriers, pilotCarriers,axis=1)            
    symbol = np.zeros((int(np.size(x)/(N-Np)), N), dtype=complex)  # the overall N subcarriers
    symbol[:, pilotCarriers] = pilot_value_change(int(np.size(x)/(N-Np)), Np, Pilot_5, Pilot_6)  # assign values to pilots
    symbol[np.arange(int(np.size(x)/(N-Np)))[:, None], dataCarriers] = x # assign values to datacarriers
    return symbol

def FFT(symbol_):
    OFDM_time = 1/(np.sqrt(N)) * np.fft.fft(symbol_).astype('complex64')    
    return OFDM_time
    
def IFFT(symbol):
    OFDM_time_ = np.sqrt(N) * np.fft.ifft(symbol).astype('complex64')   
    return OFDM_time_

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


PAPR_dB_ref = 7 
bits_final = np.zeros((Ntx,BLOCK_LENGTH))
mapper_final = np.zeros((Ntx,N-Np),dtype=complex)
OFDM_final = np.zeros((Ntx,N),dtype=complex)
symbol_final = np.zeros((Ntx,N),dtype=complex)
PAPR_final = np.zeros((Ntx,1))
ind_complex= np.zeros((Ntx,1))

successful_pilots = []
desvio_padrao = np.zeros((Ntx,1))

for nt in range(0,Ntx):
    ic=1 
    
    bitss = Bits(batch, BLOCK_LENGTH) 
    bits_final[nt] = bitss
    mapper = Modulation(bitss)
    mapper_final[nt] = mapper    
    desvio_padrao[nt] = np.std(mapper)
     
    indices_of_repeated = np.where(desvio_padrao[nt] == desvio_padrao[:nt])[0] 
    
    if np.size(indices_of_repeated) != 0:        
        print('entrei:')
        for_end = False
        for repeated_index in indices_of_repeated: 
            Pilot_1, Pilot_2 = successful_pilots[repeated_index]
            symbol_Ori = symbol(mapper, Pilot_1, Pilot_2)
            OFDM_timee = IFFT(symbol_Ori) 
            PAPR_dB_red = PAPR(OFDM_timee)
            if PAPR_dB_red < PAPR_dB_ref or np.max(indices_of_repeated):
                print('PAPR_dB_red:', PAPR_dB_red)
                break
    else:
        Pilot_1 = -np.sqrt(E)
        Pilot_2 = np.sqrt(E)
        symbol_Ori = symbol(mapper, Pilot_1, Pilot_2)
        OFDM_timee = IFFT(symbol_Ori) 
        PAPR_dB = PAPR(OFDM_timee)
        PAPR_dB_red = PAPR_dB 
        print('não entrei:')

    while PAPR_dB_red > PAPR_dB_ref:      
        Pilot_1 = np.sqrt(E)*(np.random.randn() + 1j*np.random.randn()) 
        Pilot_2 = np.sqrt(E)*(np.random.randn() + 1j*np.random.randn())   
        symbol_Ori = symbol(mapper, Pilot_1, Pilot_2)
        OFDM_timee = IFFT(symbol_Ori) 
        PAPR_dB_red = PAPR(OFDM_timee) 
        ic=ic+1
        print('entrei3')
    successful_pilots.append((Pilot_1, Pilot_2))
    print('entrei4')
    PAPR_final[nt]=PAPR_dB_red
    OFDM_final[nt]=OFDM_timee
    symbol_final[nt]=symbol_Ori
    ind_complex[nt]=ic

#%% Original Signal:

symbol_Ori_ = symbol(mapper_final, -np.sqrt(E), np.sqrt(E))
OFDM_time_ = IFFT(symbol_Ori_)
_PAPR_dB = PAPR(OFDM_time_)
_CCDF = CCDF(_PAPR_dB)    
CCDF_red = CCDF(PAPR_final)


#%% GA Implementation:

symbol_1 = symbol_Ori_[:100, :].copy()
symbol_2 = symbol_Ori_[100:200, :].copy()
symbol_3 = symbol_Ori_[200:300, :].copy()
symbol_4 = symbol_Ori_[300:400, :].copy()
symbol_5 = symbol_Ori_[400:500, :].copy()
symbol_6 = symbol_Ori_[500:600, :].copy()

OFDM_GA_1 = IFFT(symbol_1)
OFDM_GA_2 = IFFT(symbol_2)
OFDM_GA_3 = IFFT(symbol_3)
OFDM_GA_4 = IFFT(symbol_4)
OFDM_GA_5 = IFFT(symbol_5)
OFDM_GA_6 = IFFT(symbol_6)

PAPR_1 = PAPR(OFDM_GA_1)
PAPR_2 = PAPR(OFDM_GA_2)
PAPR_3 = PAPR(OFDM_GA_3)
PAPR_4 = PAPR(OFDM_GA_4)
PAPR_5 = PAPR(OFDM_GA_5)
PAPR_6 = PAPR(OFDM_GA_6)

ii = 1
L = 5

while(np.all(PAPR_1) < 5 and np.all(PAPR_2) < 5 and np.all(PAPR_3) < 5 and np.all(PAPR_4) < 5 and np.all(PAPR_5) < 5):

        # Fitness Function:            
        apt1 = 0
        apt2 = 0
        apt3 = 0
        apt4 = 0
        apt5 = 0
        
        for jj in range(0,100):
            if PAPR_1[jj] > 5:
                apt1 += 10 
            if PAPR_2[jj] > 5:
                apt2 += 10
            if PAPR_3[jj] > 5:
                apt3 += 10 
            if PAPR_4[jj] > 5:
                apt4 += 10 
            if PAPR_5[jj] > 5:
                apt5 += 10 
        Apt = [apt1, apt2, apt3, apt4, apt5]    
        # Tournament Selection:
        for jj in range(0,L):
            pos1 = np.random.randint(0,L)
            pos2 = np.random.randint(0,L)    
            [ind, valor] = min[Apt[pos1], Apt[pos2]]
            pai[jj] = symbol_1[ind]
    
        # Crossover (Uniform):
            
        
        
        # Mutation (10%):


