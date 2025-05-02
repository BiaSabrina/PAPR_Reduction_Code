import numpy as np
import sionna as sn
import matplotlib.pyplot as plt
from scipy import special
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow import keras
import time
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Input , Lambda, Flatten, Layer
import torch

NUM_BITS_PER_SYMBOL = int(input("Number Bits per Symbol: "))
Np = int(input("Number of Pilots: "))
N = int(input("Number of Carriers: "))
cp_length = int(input("Length of CP: "))

M = 2**NUM_BITS_PER_SYMBOL
batch = 1  # gera 1000 vezes N.
Ntx=100 #número de símbolos OFDM que serão gerados
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
    #allocation_position = N // Np
    matrix = np.zeros((batch, Np), dtype=int)
    for i in range(batch):
        matrix[i] = np.arange(0, Np, 1)
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
    allCarriers = create_matrix(int(np.size(x)/(N-Np)), N-Np)           
    symbols = np.zeros((int(np.size(x)/(N-Np)), N-Np), dtype=complex)  # the overall N subcarriers
    symbols[np.arange(int(np.size(x)/(N-Np)))[:, None], allCarriers] = x # assign values to datacarriers
    return symbols

def FFT(symbol_):
    OFDM_time = 1/(np.sqrt(N)) * np.fft.fft(symbol_).astype('complex64')    
    return OFDM_time
    
def IFFT(symbol):
    OFDM_time_ = np.sqrt(N) * np.fft.ifft(symbol).astype('complex64')   
    return OFDM_time_

def IFFT_CP_(OFDM_time_no_CP):
    OFDM_data_with_cp = np.zeros((len(OFDM_time_no_CP), (N-Np)+cp_length), dtype=np.complex64)    
    for ii in range(0, len(OFDM_time_no_CP)-1):
        data = OFDM_time_no_CP[ii]
        CP = np.zeros(cp_length, dtype=np.complex64)
        CP[:] = data[-cp_length:]
        data_with_cp = np.concatenate((CP, data))
        OFDM_data_with_cp[ii] = data_with_cp        
    return OFDM_data_with_cp

def IFFT_CP(OFDM_time_no_CP):
    OFDM_data_with_cp = np.zeros((len(OFDM_time_no_CP), N+cp_length), dtype=np.complex64)    
    for ii in range(0, len(OFDM_time_no_CP)-1):
        data = OFDM_time_no_CP[ii]
        CP = np.zeros(cp_length, dtype=np.complex64)
        CP[:] = data[-cp_length:]
        data_with_cp = np.concatenate((CP, data))
        OFDM_data_with_cp[ii] = data_with_cp        
    return OFDM_data_with_cp

def Remove_CP(OFDM_awgn):
    OFDM_data_no_cp = np.zeros((len(OFDM_awgn), N), dtype=np.complex64)    
    for z in range(0, len(OFDM_awgn)-1):
        OFDM_data_no_cp[z] = OFDM_awgn[z][cp_length:]        
    return OFDM_data_no_cp

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

#%%

bits = Bits(Ntx, BLOCK_LENGTH)
mod = Modulation(bits)
symbol_Ori_ = symbol_Ori(mod)
OFDM_time_ = IFFT(symbol_Ori_)
OFDM_time_with_CP = IFFT_CP_(OFDM_time_)
_PAPR_dB = PAPR(OFDM_time_with_CP)
_CCDF = CCDF(_PAPR_dB)    

R = 100e6
tempo_ms = np.linspace(0,1/R*((N-Np)+cp_length-1),(N-Np)+cp_length)

tempo_us = tempo_ms * 1e6

fig, ax = plt.subplots(figsize=(15, 5))
plt.plot(tempo_us, np.real(OFDM_time_with_CP[0,:]),label='Real')
plt.plot(tempo_us, np.imag(OFDM_time_with_CP[0,:]), label='Imaginário')
ax.set_xlabel('Tempo (μs)', fontsize=17, fontweight='bold')
ax.set_ylabel('Amplitude', fontsize=17, fontweight='bold')
ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.7, color='gray')
ax.yaxis.grid(True, which='minor', linestyle='--', alpha=0.5, color='gray')
ax.grid(axis='both', linestyle='--', alpha=0.7, color='gray')
plt.axvspan(0, (1/R*(cp_length-1))/1e-6, color='gray', alpha=0.2, linestyle='dotted')
plt.axvspan(((1/R)*N)/1e-6, (1/R*(N+cp_length-1))/1e-6, color='gray', alpha=0.2, linestyle='dotted')
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
ax.tick_params(axis='both', which='major', labelsize=17)
plt.show()
    
#%%

symbol_Ori_pil = np.zeros((Ntx,N),dtype=complex)
Pilots_ori = np.zeros((Ntx,Np),dtype=complex)

for Pil_ori in range(0,Ntx):
    Pilots_ori[Pil_ori] = (np.sqrt(E) + 1j*np.sqrt(E))
    symbol_Ori_pil[Pil_ori, :Np] = Pilots_ori[Pil_ori]
    symbol_Ori_pil[Pil_ori, Np:] = symbol_Ori_[Pil_ori]


#%% 

PAPR_dB_ref = 6

bits_final_no = np.zeros((Ntx,BLOCK_LENGTH))
mapper_final_no = np.zeros((Ntx,N-Np),dtype=complex)
OFDM_final_no = np.zeros((Ntx,N),dtype=complex)
symbol_final_no = np.zeros((Ntx,N),dtype=complex)
PAPR_final_no = np.zeros((Ntx,1))
ind_complex_no = np.zeros((Ntx,1))
Pilots = []
successful_pilots = np.zeros((Ntx,Np),dtype=complex)
for Pil in range(0,Np):
    Pilots.append(np.sqrt(E)*(np.random.randn() + 1j*np.random.randn()))
Pilots = np.array(Pilots)

for nt in range(0,Ntx):
    ic_no=1     
    bitss_no = Bits(batch, BLOCK_LENGTH) 
    bits_final_no[nt] = bitss_no
    mapper_no = Modulation(bitss_no)
    mapper_final_no[nt] = mapper_no    
    Pilot_1_no = -np.sqrt(E)
    Pilot_2_no = np.sqrt(E)
    symbol_Ori_no = symbol(mapper_no, Pilots)
    OFDM_timee_no = IFFT(symbol_Ori_no) 
    PAPR_dB_no = PAPR(OFDM_timee_no)
    PAPR_dB_red_no = PAPR_dB_no 

    while PAPR_dB_red_no > PAPR_dB_ref:     
        Pilots = []        
        for Pil in range(0,Np):
            Pilots.append(np.sqrt(E)*(np.random.randn() + 1j*np.random.randn()))
        Pilots = np.array(Pilots)        
        symbol_Ori_no = symbol(mapper_no, Pilots)
        OFDM_timee_no = IFFT(symbol_Ori_no) 
        PAPR_dB_red_no = PAPR(OFDM_timee_no) 
        ic_no=ic_no+1
                   
    successful_pilots[nt] = Pilots
    PAPR_final_no[nt]=PAPR_dB_red_no
    OFDM_final_no[nt]=OFDM_timee_no
    symbol_final_no[nt]=symbol_Ori_no
    ind_complex_no[nt]=ic_no
    print('nt:', nt)
CCDF_red_no = CCDF(PAPR_final_no)

#%% MLP:

def create_matrix_training():    
    matrix = tf.zeros((batch, N), dtype=tf.int32)
    for i in range(batch):
        matrix = tf.tensor_scatter_nd_update(matrix, [[i, j] for j in range(N)], tf.range(0, N, 1))
    return matrix

def create_pilot_training():
   # allocation_position = N/Np
    matrix = tf.zeros((batch, Np), dtype=tf.int32)
    for ii in range(batch):
        matrix = tf.tensor_scatter_nd_update(matrix, [[ii, jj] for jj in range(Np)], tf.range(0, Np, 1))
    return matrix

def pilot_value_change_training(pil):
   # ma = tf.zeros((batch, Np), dtype=tf.complex64)
    #pil = tf.cast(tf.complex(pil[0,:], pil[1,:]), tf.complex64)
    #pil = tf.reshape(pil, (batch, Np)) 
    pil = pil[0,:]+1j*pil[1,:]
    print(pil)
    ma = pil
    return ma

def symbol_training(x_, pil):
    #pil = pil[:, 0] + 1j*pil[:, 1]
    print(pil)
    x_ = tf.reshape(tf.cast(x_, tf.complex64), (batch, N-Np))
    #x_ = tf.cast(x_, tf.complex64)
    print('x_:', x_)
    #all_carriers = create_matrix_training()
    #pilot_carriers = create_pilot_training()
    #mask = tf.ones(all_carriers.shape[1], dtype=bool)
    #mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(pilot_carriers, axis=-1), tf.fill(pilot_carriers.shape, False)) 
    symbol_ = tf.zeros((batch, N), dtype=tf.complex64) 
    pilots_values = pilot_value_change_training(pil)
    print(pilots_values)
    symbol_ = tf.concat([tf.expand_dims(pilots_values[:Np], axis=0), symbol_[:, Np:]], axis=1)
    print(symbol_)
    symbol_ = tf.concat([symbol_[:, :Np], tf.expand_dims(x_, axis=0)], axis=1)
    print(symbol_)
    return symbol_

def IFFT_training(tensor):
    #tensor = tf.reshape(tensor, (batch, Np))
    print('tensor:', tensor)
    sym = symbol_training(mod[nt_nn, N-Np], tensor)
    #sym_real = tf.dtypes.cast(sym[:, 0:N], tf.float32)  # Convertendo a parte real para float32
    #sym_imag = tf.dtypes.cast(sym[:, 0:N], tf.float32)  # Convertendo a parte imaginária para float32
    #sym_complex = tf.complex(sym_real, sym_imag)
    #sym_complex = tf.dtypes.cast(sym_complex, tf.complex64)  # Convertendo para complex64
    return np.sqrt(N)*tf.signal.ifft(sym)

def papr(y_true,y_pred):
    x = tf.square(tf.abs(y_pred))
    return 10*tf.experimental.numpy.log10(tf.reduce_mean(tf.reduce_max(x, axis = 1) / tf.reduce_mean(x, axis = 1)))

IFFT_pilots_NN = tf.zeros((Ntx, N), dtype=tf.complex64)
_PAPR_dB_Pil = tf.zeros((Ntx, N))
initializer = keras.initializers.glorot_normal(seed=25)

for nt_nn in range(0,Ntx):   
    inputs = Input(shape=(N,))
    
    # ENCODER
    x1 = Dense(100, kernel_initializer=initializer, bias_initializer='random_normal')(inputs)
    x2 = BatchNormalization()(x1)
    x3 = Activation('relu')(x2)
    
    x4 = Dense(200, kernel_initializer=initializer, bias_initializer='random_normal')(x3)
    x5 = BatchNormalization()(x4)
    x6 = Activation('relu')(x5)
    
    x7 = Dense(100, kernel_initializer=initializer, bias_initializer='random_normal')(x6)
    x8 = BatchNormalization()(x7)
    x9 = Activation('relu')(x8)
    
    x10 = Dense(100, kernel_initializer=initializer, bias_initializer='random_normal')(x6)
    x11 = BatchNormalization()(x10)
    x12 = Activation('relu')(x11)
    
    x13 = Dense(Np, kernel_initializer=initializer, bias_initializer='random_normal', activation='tanh')(x12)
    
    encoder = Lambda(IFFT_training, name='encoder') (x13) 
    
    model = keras.Model(inputs=inputs, outputs=encoder)
    
    dataset_original = np.reshape(np.c_[np.real(symbol_Ori_pil[nt_nn]), np.imag(symbol_Ori_pil[nt_nn])],(2,N))
    dataset_otimizado = np.reshape(np.c_[np.real(symbol_final_no[nt_nn]), np.imag(symbol_final_no[nt_nn])],(2,N))
    valset = np.reshape(np.c_[np.real(symbol_Ori_pil[nt_nn, :10]), np.imag(symbol_Ori_pil[nt_nn, :10])],(2,9))
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    opt = keras.optimizers.Adam(0.0001)
    model.compile(optimizer=opt, loss= {'encoder' : papr})
    print(model.summary())
    history = model.fit(dataset_original, dataset_otimizado, epochs=3, batch_size=Ntx, shuffle=True, callbacks=[callback], validation_data=valset)
    dataset = np.reshape(np.c_[np.real(symbol_Ori_pil[nt_nn]), np.imag(symbol_Ori_pil[nt_nn])],(2,N))
    st = time.time()
    RED = model.predict(dataset)
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    
    IFFT_pilots_NN[nt_nn] = IFFT_CP(RED)
    _PAPR_dB_Pil[nt_nn] = PAPR(IFFT_pilots_NN)
    
_CCDF_Pil = CCDF(_PAPR_dB_Pil)    

#%% MLP: 

'''
PAPR_dB_ref = 5    
bits_final_no = np.zeros((Ntx, BLOCK_LENGTH))
mapper_final_no = np.zeros((Ntx, N-Np), dtype=complex)
OFDM_final_no = np.zeros((Ntx, N), dtype=complex)
symbol_final_no = np.zeros((Ntx, N), dtype=complex)
PAPR_final_no = np.zeros((Ntx, 1))
ind_complex_no = np.zeros((Ntx, 1))
#X_train = np.zeros((Ntx, Np), dtype=complex)

Nt = 100

successful_pilots = []
adjusted_pilots = []
X_train = []
Y_train = []

for nt in range(0, Ntx):
    Sm_no = []
    ic_no = 1     
    bitss_no = Bits(batch, BLOCK_LENGTH) 
    bits_final_no[nt] = bitss_no
    mapper_no = Modulation(bitss_no)
    mapper_final_no[nt] = mapper_no    
    
    if ic_no < 10:
        symbol_Ori_no = symbol(mapper_no, Pilots)
    if ic_no > 10:
        symbol_Ori_no = symbol(mapper_no, adjusted_pilots)
        
    OFDM_timee_no = IFFT(symbol_Ori_no) 
    PAPR_dB_no = PAPR(OFDM_timee_no)
    PAPR_dB_red_no = PAPR_dB_no 
    
    if ic_no < 10:
        
        while PAPR_dB_red_no > PAPR_dB_ref: 
            Pilots = []
    
            for Pil in range(0,Np):
                Pilots.append(np.sqrt(E)*(np.random.randn() + 1j*np.random.randn()))
            Pilots = np.array(Pilots)
            
            X_train = Pilots      
            symbol_Ori_no = symbol(mapper_no, Pilots)
            OFDM_timee_no = IFFT(symbol_Ori_no) 
            PAPR_dB_red_no = PAPR(OFDM_timee_no) 
            Sm_no.append(PAPR_dB_red_no)
            ic_no += 1
            if ic_no >= Nt:
                PAPR_dB_red_no = min(Sm_no)
                break
    successful_pilots.append((Pilots))
    Y_train = successful_pilots
    if ic_no > 10:
        mlp_model = MLP(X_train, Y_train) 
        pilots_prediction = mlp_model.predict(X_train, Y_train)
        adjusted_pilots = pilots_prediction
    
    PAPR_final_no[nt] = PAPR_dB_red_no
    OFDM_final_no[nt] = OFDM_timee_no
    symbol_final_no[nt] = symbol_Ori_no
    ind_complex_no[nt] = ic_no
    print('nt:', nt)

CCDF_red_no = CCDF(PAPR_final_no)
'''

#%%  Plot PAPR x CCDF

fig, ax = plt.subplots(figsize=(10, 8))    
plt.semilogy(np.arange(min(_PAPR_dB), max(_PAPR_dB), 0.0001), _CCDF, label=f'PAPR Original', linewidth=2.5)
plt.semilogy(np.arange(min(_PAPR_dB_Pil), max(_PAPR_dB_Pil), 0.0001), _CCDF_Pil, label=f'PAPR with NN', linewidth=2.5)
plt.semilogy(np.arange(min(PAPR_final_no), max(PAPR_final_no), 0.0001), CCDF_red_no, label=f'MCSA', linewidth=2.5)
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
plt.show()

#%% 

R = 100e6
tempo_ms = np.linspace(0,1/R*(N+cp_length-1),N+cp_length)

tempo_us = tempo_ms * 1e6

fig, ax = plt.subplots(figsize=(15, 5))
plt.stem(tempo_us, np.real(OFDM_time_with_CP[0,:]), linefmt='b-', markerfmt='bo', basefmt=' ', label='Real')
plt.stem(tempo_us, np.imag(OFDM_time_with_CP[0,:]), linefmt='r-', markerfmt='ro', basefmt=' ', label='Imaginário')
ax.set_xlabel('Tempo (μs)', fontsize=17, fontweight='bold')
ax.set_ylabel('Amplitude', fontsize=17, fontweight='bold')
ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.7, color='gray')
ax.yaxis.grid(True, which='minor', linestyle='--', alpha=0.5, color='gray')
ax.grid(axis='both', linestyle='--', alpha=0.7, color='gray')
plt.axvspan(0, (1/R*(cp_length-1))/1e-6, color='gray', alpha=0.2, linestyle='dotted')
plt.axvspan((1/R*(N-1))/1e-6, (1/R*(N+cp_length-1))/1e-6, color='gray', alpha=0.2, linestyle='dotted')
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
ax.tick_params(axis='both', which='major', labelsize=17)
plt.show()

#%% SNR x BER

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
        
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        #pilotCarriers = create_pilot(batch_size, Np)
        # Channel     
        self.OFDM_RX_FD = OFDM_time_with_CP
        y = self.awgn_channel([self.OFDM_RX_FD, no]) # no = potência do ruído
        rem_CP = Remove_CP(y)
        y_= FFT(rem_CP)      
        #y_without_pilots = np.delete(y_, pilotCarriers,axis=1)
        llr = self.demapper([y_,no])     
        return bits, llr
    
class UncodedSystemAWGN_pilots(Model): 
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
        
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        pilotCarriers = create_pilot(batch_size, Np)
        # Channel     
        self.OFDM_RX_FD_Pil = IFFT_pilots_NN
        y_Pil = self.awgn_channel([self.OFDM_RX_FD_Pil, no*np.sqrt(N/(N-Np))]) # no = potência do ruído
        rem_CP_Pil = Remove_CP(y_Pil)
        y_Pil_fft= FFT(rem_CP_Pil)         
        y_without_pilots = np.delete(y_Pil_fft, pilotCarriers,axis=1)
        llr_pil = self.demapper([y_without_pilots,no])     
        return bits_final_no, llr_pil

class UncodedSystemAWGN_Without_Memory(Model): 
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
        
        
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        pilotCarriers = create_pilot(batch_size, Np)
        
        # Channel                
        self.OFDM_RX_FDWithout = OFDM_final_no
        y_Without = self.awgn_channel([self.OFDM_RX_FDWithout, no*np.sqrt(N/(N-Np))]) # no = potência do ruído
        y_Without= FFT(y_Without)
        OFDM_demod_Without = np.delete(y_Without, pilotCarriers,axis=1)
        print(OFDM_demod_Without.size)
        llr_Without = self.demapper([OFDM_demod_Without,no])
        self.Without = y_Without
        print(self.Without)
        return bits_final_no, llr_Without

model_uncoded_awgn = UncodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, 
                                       Subcarriers=N-Np)
model_uncoded_awgn_pilots = UncodedSystemAWGN_pilots(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, 
                                       Subcarriers=N)
model_uncoded_awgn_Without = UncodedSystemAWGN_Without_Memory(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, 
                                                              Subcarriers=N)

SNR = np.arange(0, 15)

EBN0_DB_MIN = min(SNR) # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = max(SNR) # Maximum value of Eb/N0 [dB] for simulations

# Original Simulation:

ber_plots = sn.utils.PlotBER()
ber_plots.simulate(model_uncoded_awgn,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size = Ntx,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM = np.array(ber_plots.ber).ravel()

# Simulation with Pilots:
    
ber_plots_Pil = sn.utils.PlotBER()
ber_plots_Pil.simulate(model_uncoded_awgn_pilots,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size = Ntx,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM_Pil = np.array(ber_plots_Pil.ber).ravel()

# Without Memory Simulation:
    
ber_plots_Without = sn.utils.PlotBER()
ber_plots_Without.simulate(model_uncoded_awgn_Without,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size = Ntx,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM_Without = np.array(ber_plots_Without.ber).ravel() 
    
L = np.sqrt(M)
mu = 4 * (L - 1) / L  # Número médio de vizinhos
Es = 3 / (L ** 2 - 1) # Fator de ajuste da constelação
    
ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)
BER_CORRECT = np.zeros((len(ebno_dbs)))
BER_THEO_des = np.zeros((len(ebno_dbs)))

i = 0
for idx in ebno_dbs:
   BER_CORRECT[i] = (mu/(2*np.log2(M)))*special.erfc(np.sqrt(Es*NUM_BITS_PER_SYMBOL*10**(idx/10))/np.sqrt(2))
   BER_THEO_des[i] = (mu/(2*np.log2(M)))*special.erfc(np.sqrt(Es*((N-Np)/N)*NUM_BITS_PER_SYMBOL*10**(idx/10))/np.sqrt(2))
   i = i+1

fig, ax = plt.subplots(figsize=(10, 8)) 
ax.plot(ebno_dbs, BER_CORRECT, '-', label=f'Original Theoretical')
ax.plot(ebno_dbs, BER_THEO_des, '-', color='C7', label='MCSA Theoretical', linewidth=2) 
ax.plot(ebno_dbs, BER_SIM, '^', label=f'Simulation Original') 
ax.plot(ebno_dbs, BER_SIM_Without, '^', markersize=5, color='C0', label='Simulation MCSA')
ax.plot(ebno_dbs, BER_SIM_Pil, 'o', label=f'Simulation NN')    
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
plt.show()