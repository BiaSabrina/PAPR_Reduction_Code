import numpy as np
import sionna as sn
import matplotlib.pyplot as plt
from scipy import special
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow import keras
import time
import tensorflow.experimental.numpy as tnp
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Input , Lambda, Flatten, Layer

NUM_BITS_PER_SYMBOL = int(input("Number of Bits per Symbol: "))
Np = int(input("Pilots Number: "))
N = int(input("Carriers Number: "))
#cp_length = int(input("CP Length: "))

cp_length = 8

M = 2**NUM_BITS_PER_SYMBOL # Ordem da modulação QAM.
batch = 1  # 1 símbolo por rodada.
Ntx=3000 # Número de símbolos OFDM que serão gerados.
BLOCK_LENGTH = (N-Np)*NUM_BITS_PER_SYMBOL

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
    allCarriers = create_matrix(int(np.size(x)/(N-Np)), N-Np)           
    symbols = np.zeros((int(np.size(x)/(N-Np)), N-Np), dtype=complex)  # the overall N subcarriers
    symbols[np.arange(int(np.size(x)/(N-Np)))[:, None], allCarriers] = x # assign values to datacarriers
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

def IFFT_CP_(OFDM_time_no_CP):
    OFDM_data_with_cp = np.zeros((len(OFDM_time_no_CP), (N-Np)+cp_length), dtype=np.complex64)    
    for ii in range(0, len(OFDM_time_no_CP)):
        data = OFDM_time_no_CP[ii]
        CP = np.zeros(cp_length, dtype=np.complex64)
        CP[:] = data[-cp_length:]
        data_with_cp = np.concatenate((CP, data))
        OFDM_data_with_cp[ii] = data_with_cp        
    return OFDM_data_with_cp

def IFFT_CP(OFDM_time_no_CP):
    OFDM_data_with_cp = np.zeros((len(OFDM_time_no_CP), N+cp_length), dtype=np.complex64)    
    for ii in range(0, len(OFDM_time_no_CP)):
        data = OFDM_time_no_CP[ii]
        CP = np.zeros(cp_length, dtype=np.complex64)
        CP[:] = data[-cp_length:]
        data_with_cp = np.concatenate((CP, data))
        OFDM_data_with_cp[ii] = data_with_cp        
    return OFDM_data_with_cp

def Remove_CP(OFDM_awgn):
    OFDM_data_no_cp = np.zeros((len(OFDM_awgn), N), dtype=np.complex64)    
    for z in range(0, len(OFDM_awgn)):
        OFDM_data_no_cp[z] = OFDM_awgn[z][cp_length:]        
    return OFDM_data_no_cp

def Remove_CP_(OFDM_awgn):
    OFDM_data_no_cp = np.zeros((len(OFDM_awgn), N-Np), dtype=np.complex64)    
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
symbol__ = symbol_Ori(mod)
OFDM__ = IFFTOri(symbol__)
Pilots_ori = []        
for Pil in range(0,Np):
    Pilots_ori.append(0)
Pilots_ori = np.array(Pilots_ori)        
symbol_Ori_ = symbol(mod, Pilots_ori)
OFDM_time_ = IFFTOri(symbol_Ori_)
#OFDM_time_with_CP = IFFT_CP_(OFDM_time_)
_PAPR_dB = PAPR(OFDM_time_)
_CCDF = CCDF(_PAPR_dB)    

R = 100e6
#tempo_ms = np.linspace(0,1/R*((N-Np)+cp_length-1),(N-Np)+cp_length)

tempo_ms = np.linspace(0,1/R*((N-Np)-1),(N-Np))

tempo_us = tempo_ms * 1e6
'''
fig, ax = plt.subplots(figsize=(15, 5))
plt.plot(tempo_us, np.real(OFDM_time_[0,:]),label='Real')
plt.plot(tempo_us, np.imag(OFDM_time_[0,:]), label='Imaginário')
ax.set_xlabel('Tempo (μs)', fontsize=17, fontweight='bold')
ax.set_ylabel('Amplitude', fontsize=17, fontweight='bold')
ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.7, color='gray')
ax.yaxis.grid(True, which='minor', linestyle='--', alpha=0.5, color='gray')
ax.grid(axis='both', linestyle='--', alpha=0.7, color='gray')
plt.axvspan(0, (1/R*(cp_length-1))/1e-6, color='gray', alpha=0.2, linestyle='dotted')
plt.axvspan(((1/R)*(N-Np))/1e-6, (1/R*((N-Np)+cp_length-1))/1e-6, color='gray', alpha=0.2, linestyle='dotted')
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
'''   
#%%
'''
symbol_Ori_pil = np.zeros((Ntx,N),dtype=complex)
Pilots_ori = np.zeros((Ntx,Np),dtype=complex)

for Pil_ori in range(0,Ntx):
    Pilots_ori[Pil_ori] = (np.sqrt(E) + 1j*np.sqrt(E))
    symbol_Ori_pil[Pil_ori, :Np] = Pilots_ori[Pil_ori]
    symbol_Ori_pil[Pil_ori, Np:] = symbol_Ori_[Pil_ori]

'''
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
    mapper_no = mod[nt]
    mapper_final_no[nt] = mapper_no    
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


#%% Não Supervisionado:
'''
def create_matrix_training():    
    matrix = tf.zeros((batch, N), dtype=tf.int32)
    for i in range(batch):
        matrix = tf.tensor_scatter_nd_update(matrix, [[i, j] for j in range(N)], tf.range(0, N, 1))
    return matrix

def create_pilot_training():
    allocation_position = N // Np
    matrix = tf.zeros((batch, Np), dtype=tf.int32)
    for i in range(batch):
        matrix = tf.tensor_scatter_nd_update(matrix, [[i, j] for j in range(Np)],range(0, Np * allocation_position, allocation_position))
    return matrix

def pilot_value_change_training(pil):
    pil = tf.complex(pil[0,:], pil[1,:])
    pil = tf.reshape(pil, (batch, Np)) 
    ma = pil
    return ma

def symbol_training(x_, pil):    
    all_Carriers = create_matrix(batch, N)
    pilot_Carriers = create_pilot(batch, Np)
    data_Carriers = np.delete(all_Carriers, pilot_Carriers, axis=1) 
    pilot_Carriers = tf.convert_to_tensor(pilot_Carriers)   
    symbol_ = np.zeros((batch, N), dtype='complex64')
    pilots_values_ = pilot_value_change_training(pil)
    symbol_[:, data_Carriers] = x_
    
    update_indices = tf.expand_dims(tf.range(batch), axis=-1)  # Indices das linhas
    update_indices = tf.tile(update_indices, [1, Np])           # Replicando para Np colunas
    update_indices = tf.stack([update_indices, pilot_Carriers], axis=-1)  # Índices completos
    symbol_ = tf.tensor_scatter_nd_update(symbol_, update_indices, pilots_values_)    
    return symbol_


def IFFT_training(tensor):
    sym = symbol_training(mod[nt_nn, :], tensor)
    return np.sqrt(N)*tf.signal.ifft(sym)

def papr(y_true,y_pred):
    x = tf.square(tf.abs(y_pred))
    return 10*tf.experimental.numpy.log10(tf.reduce_mean(tf.reduce_max(x, axis = 1) / tf.reduce_mean(x, axis = 1)))

IFFT_pilots_NN = np.zeros((Ntx,N),dtype=complex)
_PAPR_dB_Pil = np.zeros((Ntx,1))
initializer = keras.initializers.glorot_normal(seed=25)
lamba = 0.002
callback = tf.keras.callbacks.EarlyStopping(patience=100,min_delta=1e-9, verbose=1,restore_best_weights=True)

st = time.time()

for nt_nn in range(0,Ntx):   
    inputs = Input(shape=(N,))
    
    # ENCODER
    x1 = Dense(300, kernel_initializer=initializer, bias_initializer='random_normal')(inputs)
    x2 = BatchNormalization()(x1)
    x3 = Dropout(0.2)(x2)
    x3_ = Activation('ReLU')(x3)
    
    x4 = Dense(300, kernel_initializer=initializer, bias_initializer='random_normal')(x3_)
    x4_1 = BatchNormalization()(x4)
    x5 = Dropout(0.2)(x4_1)
    x6 = Activation('ReLU')(x5)
    
    x7 = Dense(150, kernel_initializer=initializer, bias_initializer='random_normal')(x6)
    x7_1 = BatchNormalization()(x7)
    x8 = Dropout(0.2)(x7_1)
    x9 = Activation('ReLU')(x8)
    
    x10 = Dense(2800, kernel_initializer=initializer, bias_initializer='random_normal')(x6)
    x10_1 = BatchNormalization()(x10)
    x11 = Dropout(0.2)(x10_1)
    x12 = Activation('relu')(x11)
    
    x13 = Dense(Np, kernel_initializer=initializer, bias_initializer='random_normal', activation='tanh')(x6)
    
    encoder = Lambda(IFFT_training, name='encoder') (x13) 
    
    model = keras.Model(inputs=inputs, outputs=encoder)
    
    dataset_original = np.reshape(np.c_[np.real(symbol_Ori_[nt_nn]), np.imag(symbol_Ori_[nt_nn])],(2,N))
    dataset_otimizado = np.reshape(np.c_[np.real(symbol_final_no[nt_nn]), np.imag(symbol_final_no[nt_nn])],(2,N))
    opt = keras.optimizers.Adam(0.0001)
    model.compile(optimizer=opt, loss= {'encoder' : papr}, loss_weights=[lamba, 1])
    #model.compile(optimizer=opt, loss_weights=[lamba, 1])
    #print(model.summary())
    
    history = model.fit(dataset_original, dataset_otimizado, epochs=1000, callbacks = [callback])
       
    dataset = dataset_original   
    RED = model.predict(dataset)
    
    print('nt_nn:', nt_nn)
    
    
    _PAPR_dB_Pil[nt_nn] = PAPR(RED)
    IFFT_pilots_NN[nt_nn] = RED
    #_PAPR_dB_Pil[nt_nn] = PAPR(IFFT_pilots_NN[nt_nn])
    
_CCDF_Pil = CCDF(_PAPR_dB_Pil)    

et = time.time()
elapsed_time = (et - st)/60
print('Execution time:', elapsed_time, 'minutes')
'''
#%% Não Supervisionado:
'''
def create_matrix_training():    
    matrix = tf.zeros((batch, N), dtype=tf.int32)
    for i in range(batch):
        matrix = tf.tensor_scatter_nd_update(matrix, [[i, j] for j in range(N)], tf.range(0, N, 1))
    return matrix

def create_pilot_training():
    allocation_position = N // Np
    matrix = tf.zeros((batch, Np), dtype=tf.int32)
    for i in range(batch):
        matrix = tf.tensor_scatter_nd_update(matrix, [[i, j] for j in range(Np)],range(0, Np * allocation_position, allocation_position))
    return matrix
'''

'''
def IFFT_training(tensor):
    sym = symbol_training(mapper_final_no[nt_nn, :], tensor)
    return np.sqrt(N)*tf.signal.ifft(sym)
'''
def papr(y_true,y_pred):
    x = tf.square(tf.abs(y_pred))
    return 10*tf.experimental.numpy.log10(tf.reduce_mean(tf.reduce_max(x, axis = 1) / tf.reduce_mean(x, axis = 1)))

def IFFT_training(tensor):
    sym = symbol_training(mapper_final_no[nt_nn, :], tensor)
    return np.sqrt(N)*tf.signal.ifft(sym)

'''
def papr(y_true, y_pred):
    y_true = tf.cast(y_true, tf.complex64)
    y_pred = tf.cast(y_pred, tf.complex64)
    
    # Calcular a diferença entre y_true e y_pred
    diff = y_true - y_pred
    
    # Calcular o quadrado da magnitude de diff
    x = tf.square(tf.abs(diff))
    
    # Calcular o PAPR
    papr_value = 10 * tf.experimental.numpy.log10(tf.reduce_mean(tf.reduce_max(x, axis=1) / tf.reduce_mean(x, axis=1)))
    
    return papr_value
'''
def pilot_value_change_training(pil):
    pil = tf.complex(pil[0,:], pil[1,:])
    pil = tf.reshape(pil, (batch, Np)) 
    ma = pil
    return ma

def symbol_training(x_, pil):    
    all_Carriers = create_matrix(batch, N)
    pilot_Carriers = create_pilot(batch, Np)
    data_Carriers = np.delete(all_Carriers, pilot_Carriers, axis=1) 
    pilot_Carriers = tf.convert_to_tensor(pilot_Carriers)   
    symbol_ = np.zeros((batch, N), dtype='complex64')
    pilots_values_ = pilot_value_change_training(pil)
    symbol_[:, data_Carriers] = x_
    
    update_indices = tf.expand_dims(tf.range(batch), axis=-1)  # Indices das linhas
    update_indices = tf.tile(update_indices, [1, Np])           # Replicando para Np colunas
    update_indices = tf.stack([update_indices, pilot_Carriers], axis=-1)  # Índices completos
    symbol_ = tf.tensor_scatter_nd_update(symbol_, update_indices, pilots_values_)    
    print(symbol_)
    return symbol_


IFFT_pilots_NN = np.zeros((Ntx,N),dtype=complex)
_PAPR_dB_Pil = np.zeros((Ntx,1))
initializer = keras.initializers.glorot_normal(seed=25)
lamba = 0.002
callback = tf.keras.callbacks.EarlyStopping(patience=100,min_delta=1e-9, verbose=1,restore_best_weights=True)

st = time.time()

for nt_nn in range(0,Ntx):   
    
    inputs = Input(shape=(N,))
    x = Dense(3000, activation='sigmoid', kernel_initializer=initializer)(inputs)
    
    x = Dense(3000, activation='sigmoid', kernel_initializer=initializer)(x)
    
    x = Dense(3000, activation='sigmoid', kernel_initializer=initializer)(x)
    
    x = Dense(Np, kernel_initializer=initializer)(x)
    
    encoder = Lambda(IFFT_training, name='encoder')(x)
    
    model = keras.Model(inputs=inputs, outputs=encoder)
    
    dataset_original = np.reshape(np.c_[np.real(symbol_Ori_[nt_nn]), np.imag(symbol_Ori_[nt_nn])],(2,N))
    #dataset_otimizado = np.reshape(np.c_[np.real(symbol_final_no[nt_nn]), np.imag(symbol_final_no[nt_nn])],(2,N))
    
    opt = keras.optimizers.Adam(0.0001)
    model.compile(optimizer=opt, loss= {'encoder' : papr}, loss_weights=[lamba, 1])
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    #print(model.summary())
    
    history = model.fit(dataset_original, dataset_original, epochs=100, callbacks = [callback], validation_data = [dataset_original, dataset_original])
       
    dataset = dataset_original   
    RED = model.predict(dataset)
    
    print('nt_nn:', nt_nn)
    
    IFFT_pilots_NN[nt_nn] = RED
    _PAPR_dB_Pil[nt_nn] = PAPR(RED)
    #IFFT_pilots_NN[nt_nn] = RED
    #_PAPR_dB_Pil[nt_nn] = PAPR(IFFT_pilots_NN[nt_nn])
    
_CCDF_Pil = CCDF(_PAPR_dB_Pil)    

et = time.time()
elapsed_time = (et - st)/60
print('Execution time:', elapsed_time, 'minutes')


#%% PTS:
    
p = [1, -1, -1j, 1j]  # phase factor possible values
B = []

for b1 in range(4):
    for b2 in range(4):
        for b3 in range(4):
            for b4 in range(4):
                B.append([p[b1], p[b2], p[b3], p[b4]])  # all possible combinations
B = np.array(B)

L = 4

ofdm_symbol = symbol_Ori_

NN = Ntx
papr_min = np.zeros((NN,1))
ofdm_symbol_reconstructed = np.zeros((NN, N), dtype=complex)
sig = np.zeros((NN, N), dtype=complex)
PP1 = np.zeros((NN, N), dtype=complex)
PP2 = np.zeros((NN, N), dtype=complex)
PP3 = np.zeros((NN, N), dtype=complex)
PP4 = np.zeros((NN, N), dtype=complex)

a = np.zeros((NN,1), dtype=complex)
b = np.zeros((NN,1), dtype=complex)
c = np.zeros((NN,1), dtype=complex)
d = np.zeros((NN,1), dtype=complex)

ic_complex_pts = np.zeros((NN,1))

for i in range(NN):
   ic_pts = 0
   time_domain_signal = np.abs(IFFT(np.concatenate([ofdm_symbol[i, 0:16], np.zeros((L-1)*N), ofdm_symbol[i, 16:64]])))
   meano = np.mean(np.abs(time_domain_signal)**2)
   peako = np.max(np.abs(time_domain_signal)**2)
   papro = 10 * np.log10(peako/meano)
   
   # Partition OFDM Symbol
   P1 = np.concatenate([ofdm_symbol[i, 0:32], np.zeros(96)])
   P2 = np.concatenate([np.zeros(32), ofdm_symbol[i, 32:64], np.zeros(64)])
   P3 = np.concatenate([np.zeros(64), ofdm_symbol[i, 64:96], np.zeros(32)])
   P4 = np.concatenate([np.zeros(96), ofdm_symbol[i, 96:128]])
   
   Pt1 = (IFFT(P1))
   Pt2 = (IFFT(P2))
   Pt3 = (IFFT(P3))
   Pt4 = (IFFT(P4))
   
   PP1[i, :] = Pt1
   PP2[i, :] = Pt2
   PP3[i, :] = Pt3
   PP4[i, :] = Pt4
   
   papr_min[i] = papro
   
   for k in range(N):
       final_signal = B[k,0]*Pt1 + B[k,1]*Pt2 + B[k,2]*Pt3 + B[k,3]*Pt4 # Combination of the signal.
       meank = np.mean(np.abs(final_signal)**2)
       peak = np.max(np.abs(final_signal)**2)
       papr = 10 * np.log10(peak/meank)
       
       if papr < papr_min[i]:
                            
           a[i] = B[k, 0]
           b[i] = B[k, 1]
           c[i] = B[k, 2]
           d[i] = B[k, 3]

           papr_min[i] = papr
           sig[i, :] = final_signal
           break     
    
CCDF_PTS_re = CCDF(papr_min)

#%% Plot PAPR:
    
fig, ax = plt.subplots(figsize=(10, 8))    
plt.semilogy(np.arange(min(_PAPR_dB), max(_PAPR_dB), 0.0001), _CCDF, label=f'PAPR Original', linewidth=2.5)
plt.semilogy(np.arange(min(_PAPR_dB_Pil), max(_PAPR_dB_Pil), 0.0001), _CCDF_Pil, label=f'NN Unsupervised', linewidth=2.5)
plt.semilogy(np.arange(min(PAPR_final_no), max(PAPR_final_no), 0.0001), CCDF_red_no, label=f'MCSA', linewidth=2.5)
#ax.semilogy(np.arange(min(papr_min), max(papr_min), 0.0001), CCDF_PTS_re, '-', color='C5', label='PTS ')
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

#%% Plot:
'''
fig, ax = plt.subplots(figsize=(10, 8)) 
plt.plot(history.history['mean_squared_error'],label='Train')    
plt.plot(history.history['val_mean_squared_error'], label='Validation')
plt.title('Mean Squared Error per Epochs')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('MSE')

fig, ax = plt.subplots(figsize=(10, 8)) 
plt.plot(history.history['mean_squared_error'],label='Training Loss')    
plt.plot(history.history['val_mean_squared_error'], label='Validation Loss')
plt.xlabel('Epochs', fontsize=17, fontweight='bold')
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
#ax.set_ylim([1e-3, 1])
plt.ylabel('Mean Square Error (MSE)', fontsize=17, fontweight='bold')
plt.show()
'''

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

#%% 
'''
R = 100e6
tempo_ms = np.linspace(0,1/R*((N-Np)+cp_length-1),(N-Np)+cp_length)

tempo_us = tempo_ms * 1e6

fig, ax = plt.subplots(figsize=(15, 5))
plt.stem(tempo_us, np.real(OFDM_time_[0,:]), linefmt='b-', markerfmt='bo', basefmt=' ', label='Real')
plt.stem(tempo_us, np.imag(OFDM_time_[0,:]), linefmt='r-', markerfmt='ro', basefmt=' ', label='Imaginário')
ax.set_xlabel('Tempo (μs)', fontsize=17, fontweight='bold')
ax.set_ylabel('Amplitude', fontsize=17, fontweight='bold')
ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.7, color='gray')
ax.yaxis.grid(True, which='minor', linestyle='--', alpha=0.5, color='gray')
ax.grid(axis='both', linestyle='--', alpha=0.7, color='gray')
plt.axvspan(0, (1/R*(cp_length-1))/1e-6, color='gray', alpha=0.2, linestyle='dotted')
plt.axvspan(((1/R)*(N-Np))/1e-6, (1/R*((N-Np)+cp_length-1))/1e-6, color='gray', alpha=0.2, linestyle='dotted')
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
'''
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
        pilotCarriers = create_pilot(batch_size, Np)
        # Channel     
        self.OFDM_RX_FD = OFDM__
        y = self.awgn_channel([self.OFDM_RX_FD, no]) # no = potência do ruído
        #rem_CP = Remove_CP_(y)
        y_= FFTOri(y)      
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
        #rem_CP_Pil = Remove_CP(y_Pil)
        y_Pil_fft= FFT(y_Pil)         
        y_without_pilots = np.delete(y_Pil_fft, pilotCarriers,axis=1)
        llr_pil = self.demapper([y_without_pilots,no])     
        return bits, llr_pil

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
        return bits, llr_Without

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