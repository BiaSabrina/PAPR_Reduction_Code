import numpy as np
import sionna as sn
import matplotlib.pyplot as plt
from scipy import special
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow import keras
import time
from keras.optimizers import Adam
from keras.layers import Dense, Flatten
from tensorflow.keras import Sequential


NUM_BITS_PER_SYMBOL = int(input("Number of Bits per Symbol: "))
Np = int(input("Pilots Number: "))
N = int(input("Carriers Number: "))

M = 2**NUM_BITS_PER_SYMBOL 

Ntx=4500
BLOCK_LENGTH = (N-Np)*NUM_BITS_PER_SYMBOL

Eo = 1 

E = (2/3)*(M-1)*Eo 

constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL) 
constellation.show()

cont = 0

#%% Allocation of pilots and carriers:
    
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

#%% Generation of bits and modulation:
    
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

#%%  PAPR and CCDF:
    
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
    eixo_x_red = np.arange(mi, ma, 0.00001) 
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
OFDM_time_ = IFFT(symbol_Ori_)
#OFDM_time_with_CP = IFFT_CP_(OFDM_time_)
_PAPR_dB = PAPR(OFDM_time_)
_CCDF = CCDF(_PAPR_dB)    

#%% DFT

Symbol_DFT = mod

P1_DFT = Symbol_DFT[:, 0:16]
P2_DFT = Symbol_DFT[:, 16:32]
P3_DFT = Symbol_DFT[:, 32:48]
P4_DFT = Symbol_DFT[:, 48:62]

Pt1 = FFTOri(P1_DFT)
Pt2 = FFTOri(P2_DFT)
Pt3 = FFTOri(P3_DFT)
Pt4 = FFTOri(P4_DFT)

F_DFT = np.concatenate([Pt1, Pt2, Pt3, Pt4],axis=1)
OFDM_DFT = IFFTOri(F_DFT)

PAPR_dB_DFT = PAPR(OFDM_DFT)
    
CCDF_final_DFT = CCDF(PAPR_dB_DFT)    

#%% MCSA:

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
    ind_complex_no[nt]=ic_no # Complexity in terms of IFFT
    print('nt:', nt)
CCDF_red_no = CCDF(PAPR_final_no)

#%% Neural Network:

IFFT_pilots_NN = np.zeros((Ntx,N),dtype=complex)
symbol_training_NN = np.zeros((Ntx,N),dtype=complex)
_PAPR_dB_Pil = np.zeros((Ntx,1))
initializer = keras.initializers.glorot_normal(seed=25)
#callback = tf.keras.callbacks.EarlyStopping(patience=1000,min_delta=1e-15, verbose=1,restore_best_weights=True)


dataset_original = np.c_[np.real(symbol_Ori_), np.imag(symbol_Ori_)]
dataset_otimizado = np.c_[np.real(symbol_final_no), np.imag(symbol_final_no)]

model = Sequential()
model.add(Flatten(input_shape=(dataset_original.shape[1],)))
model.add(Dense(500, activation='relu', kernel_initializer=initializer))
model.add(Dense(N*2, kernel_initializer=initializer))

optimizer = Adam(learning_rate=0.0001)

model.compile(loss='MSE', optimizer=optimizer, metrics=['accuracy'])


# Training the Model:
st = time.time()
history = model.fit(dataset_original, dataset_otimizado, epochs= 3000, validation_data = [dataset_original, dataset_otimizado], verbose=2, shuffle=True)
et = time.time()
elapsed_time = (et - st)/60
print('Execution time:', elapsed_time, 'minutes')

dataset = dataset_original   
RED = model.predict(dataset)

RED = RED[:,0:N] + 1j*RED[:,N:N*2]
IFFT_pilots_NN = IFFT(RED)
_PAPR_dB_Pil = PAPR(IFFT_pilots_NN)    
_CCDF_Pil = CCDF(_PAPR_dB_Pil)    


#%% PTS:
    
# All permutations of phase factor B
p = [1, -1, -1j, 1j]  # phase factor possible values
B = []

for b1 in range(4):
    for b2 in range(4):
        for b3 in range(4):
            for b4 in range(4):
                B.append([p[b1], p[b2], p[b3], p[b4]])  # all possible combinations
B = np.array(B)

L = 4

ofdm_symbol = symbol__

NN = Ntx
papr_min = np.zeros((NN,1))
ofdm_symbol_reconstructed = np.zeros((NN, N-Np), dtype=complex)
sig = np.zeros((NN, N-Np), dtype=complex)
PP1 = np.zeros((NN, N-Np), dtype=complex)
PP2 = np.zeros((NN, N-Np), dtype=complex)
PP3 = np.zeros((NN, N-Np), dtype=complex)
PP4 = np.zeros((NN, N-Np), dtype=complex)

a = np.zeros((NN,1), dtype=complex)
b = np.zeros((NN,1), dtype=complex)
c = np.zeros((NN,1), dtype=complex)
d = np.zeros((NN,1), dtype=complex)


ic_complex_pts = np.zeros((NN,1))

for i in range(NN):
   ic_pts = 0
   time_domain_signal = np.abs(IFFT(np.concatenate([ofdm_symbol[i, 0:16], np.zeros((L-1)*(N-Np)), ofdm_symbol[i, 16:62]])))
   meano = np.mean(np.abs(time_domain_signal)**2)
   peako = np.max(np.abs(time_domain_signal)**2)
   papro = 10 * np.log10(peako/meano)
   
   # Partition OFDM Symbol
   P1 = np.concatenate([ofdm_symbol[i, 0:16], np.zeros(46)])
   P2 = np.concatenate([np.zeros(16), ofdm_symbol[i, 16:32], np.zeros(30)])
   P3 = np.concatenate([np.zeros(32), ofdm_symbol[i, 32:48], np.zeros(14)])
   P4 = np.concatenate([np.zeros(48), ofdm_symbol[i, 48:62]])
   
   Pt1 = (IFFTOri(P1))
   Pt2 = (IFFTOri(P2))
   Pt3 = (IFFTOri(P3))
   Pt4 = (IFFTOri(P4))
   
   PP1[i, :] = Pt1
   PP2[i, :] = Pt2
   PP3[i, :] = Pt3
   PP4[i, :] = Pt4
   
   papr_min[i] = papro
   
   for k in range(N*L):
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

#%% PAPR, Complexity, Accuracy and Loss:

fig, ax = plt.subplots(figsize=(10, 8))    
ax.semilogy(np.arange(min(_PAPR_dB), max(_PAPR_dB), 0.00001), _CCDF, '-', color='C2', label=f'PAPR Original', linewidth=2.5)
ax.semilogy(np.arange(min(_PAPR_dB_Pil), max(_PAPR_dB_Pil), 0.00001), _CCDF_Pil, '^', color='C1', label=f'Neural Network', linewidth=2.5)
ax.semilogy(np.arange(min(PAPR_final_no), max(PAPR_final_no), 0.00001), CCDF_red_no, '>', color='C3', label=f'MCSA', linewidth=2.5)
ax.semilogy(np.arange(min(papr_min), max(papr_min), 0.00001), CCDF_PTS_re, 'o', color='C0', label='PTS ', linewidth=2.5)
ax.semilogy(np.arange(min(PAPR_dB_DFT), max(PAPR_dB_DFT), 0.00001), CCDF_final_DFT, '<', color='C4', label='DFT Spread', linewidth=2.5)
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

'''
num_carriers = np.arange(16, 66, 2)
papr_target = np.linspace(5, 8, 25) 

X, Y = np.meshgrid(num_portadoras, papr_alvo)

Z = np.array(ind_complex_no) # We have to save the "ind_complex_no" for all "papr_target" and all "num_carriers" to put here.


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Superfície com colormap e transparência ajustados
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', linewidth=0.3, alpha=0.85)

ax.view_init(elev=35, azim=135)
ax.set_xlabel('Number of Carriers', fontsize=12, fontweight='bold', labelpad=15)
ax.set_ylabel(r'$\mathbf{\mathcal{P}_{max}}$', fontsize=16, fontweight='bold', labelpad=15)
ax.set_zlabel('Average Iterations', fontsize=12, fontweight='bold', labelpad=10)
ax.tick_params(axis='both', which='major', labelsize=10, pad=8)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
ax.set_facecolor('white')
plt.savefig('3D_Average.pdf', bbox_inches='tight', dpi=300)
plt.show()
'''

fig, ax = plt.subplots(figsize=(10, 8)) 
plt.semilogy(history.history['loss'], '-', color='blue', label='Training Loss')    
plt.semilogy(history.history['val_loss'],'--', color='red', label='Validation Loss')
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
plt.savefig('Loss.pdf', bbox_inches='tight', dpi=300)
plt.show()

fig, ax = plt.subplots(figsize=(10, 8)) 
plt.semilogy(history.history['accuracy'], '-', color='blue', label='Training Accuracy')    
plt.semilogy(history.history['val_accuracy'], '--', color='red', label='Validation Accuracy')
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
plt.ylabel('Accuracy', fontsize=17, fontweight='bold')
plt.savefig('Accuracy.pdf', bbox_inches='tight', dpi=300)
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
        y_Pil = self.awgn_channel([self.OFDM_RX_FD_Pil, no*(N/(N-Np))]) # no = potência do ruído
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
        y_Without = self.awgn_channel([self.OFDM_RX_FDWithout, no*(N/(N-Np))]) # no = potência do ruído
        y_Without= FFT(y_Without)
        OFDM_demod_Without = np.delete(y_Without, pilotCarriers,axis=1)
        #print(OFDM_demod_Without.size)
        llr_Without = self.demapper([OFDM_demod_Without,no])
        return bits, llr_Without
    
class UncodedSystemAWGN_PTS(Model): 
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
        pilotCarriers = create_pilot(batch_size, Np)  
        h = np.array([1])
        self.H = np.fft.fft(h,self.N)  
        
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        
        self.OFDM_RX_FD_ = sig
                
        # Channel     
        _y_ = self.awgn_channel([self.OFDM_RX_FD_, no]) # no = potência do ruído        
    
        Pt1_original = (_y_ - b * PP2 - c * PP3 - d * PP4) / a
        Pt2_original = (_y_ - a * PP1 - c * PP3 - d * PP4) / b
        Pt3_original = (_y_ - a * PP1 - b * PP2 - d * PP4) / c
        Pt4_original = (_y_ - a * PP1 - b * PP2 - c * PP3) / d
        
        F1 = FFTOri(Pt1_original)
        F2 = FFTOri(Pt2_original)
        F3 = FFTOri(Pt3_original)
        F4 = FFTOri(Pt4_original)   
        
        # Reconstruct the OFDM Symbol
        ofdm_symbol_reconstructed = np.concatenate((F1[:, 0:16], F2[:, 16:32], F3[:, 32:48], F4[:, 48:62]), axis=1)  
        
        print('PRINT PTS OUTPUT:', ofdm_symbol_reconstructed.dtype)  # Deve imprimir complex128
        print('PRINT NO PTS OUTPUT:', no.dtype)
        
        #ofdm_symbol_reconstructed = np.delete(ofdm_symbol_reconstructed, pilotCarriers,axis=1)
        llr_PTS = self.demapper([ofdm_symbol_reconstructed, no])
        #print(llr_PTS)
        self.Out_PTS = ofdm_symbol_reconstructed
        
        return bits, llr_PTS

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
        
        bits1 = Bits(1, self.block_length)
        mod1 = Modulation(bits)
        P1_DFT = mod1[0:16]
        P2_DFT = mod1[16:32]
        P3_DFT = mod1[32:48]
        P4_DFT = mod1[48:62]

        Pt1 = FFT(P1_DFT)
        Pt2 = FFT(P2_DFT)
        Pt3 = FFT(P3_DFT)
        Pt4 = FFT(P4_DFT)

        F_DFT = np.concatenate((Pt1, Pt2, Pt3, Pt4))
        OFDM_DFT = IFFT(F_DFT)
        
        self.OFDM_RX_FD_DFT = OFDM_DFT
        # Channel     
        _y_DFT = self.awgn_channel([self.OFDM_RX_FD_DFT, no]) # no = potência do ruído        
        
        FFT_DFT = FFTOri(_y_DFT)

        P1_DFT_rec = FFT_DFT[0:16]
        P2_DFT_rec = FFT_DFT[16:32]
        P3_DFT_rec = FFT_DFT[32:48]
        P4_DFT_rec = FFT_DFT[48:62]
        
        Pt1_rec = IFFT(P1_DFT_rec)
        Pt2_rec = IFFT(P2_DFT_rec)
        Pt3_rec = IFFT(P3_DFT_rec)
        Pt4_rec = IFFT(P4_DFT_rec)
        
        # Reconstruct the OFDM Symbol
        ofdm_symbol_DFT_reconstructed = np.concatenate((Pt1_rec, Pt2_rec, Pt3_rec, Pt4_rec))
        
        
        ofdm_symbol_DFT_reconstructed = tf.cast(ofdm_symbol_DFT_reconstructed, tf.complex64)
        no = tf.cast(no, tf.float32)
        
        print("Received OFDM:", _y_DFT)
        print("Reconstructed Symbol:", ofdm_symbol_DFT_reconstructed)

        llr_DFT = self.demapper([ofdm_symbol_DFT_reconstructed, no])
    
        return bits1, llr_DFT

  

model_uncoded_awgn = UncodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, 
                                       Subcarriers=N-Np)
model_uncoded_awgn_pilots = UncodedSystemAWGN_pilots(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, 
                                       Subcarriers=N)
model_uncoded_awgn_Without = UncodedSystemAWGN_Without_Memory(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, 
                                                              Subcarriers=N)
model_uncoded_awgn_PTS = UncodedSystemAWGN_PTS(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, 
                                               Subcarriers=N-Np)
model_uncoded_awgn_DFT = UncodedSystemAWGN_DFT(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, 
                                               Subcarriers=N-Np)

SNR = np.arange(0, 15)

EBN0_DB_MIN = min(SNR) # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = max(SNR) # Maximum value of Eb/N0 [dB] for simulations

# DFT Simulation:
    
ber_plots_DFT = sn.utils.PlotBER()
ber_plots_DFT.simulate(model_uncoded_awgn_DFT,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size = 1,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=1000, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM_DFT = np.array(ber_plots_DFT.ber).ravel()

# Original Simulation:

ber_plots = sn.utils.PlotBER()
ber_plots.simulate(model_uncoded_awgn,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size = Ntx,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=1000, # run 100 Monte-Carlo simulations (each with batch_size samples)
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
                  max_mc_iter=1000, # run 100 Monte-Carlo simulations (each with batch_size samples)
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
                  max_mc_iter=1000, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM_Without = np.array(ber_plots_Without.ber).ravel()

# PTS Simulation:
    
ber_plots_PTS = sn.utils.PlotBER()
ber_plots_PTS.simulate(model_uncoded_awgn_PTS,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size = Ntx,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=1000, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM_PTS = np.array(ber_plots_PTS.ber).ravel()
    
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
    BER_THEO_des[i] = (mu/(2*np.log2(M)))*special.erfc(np.sqrt(((N-Np)/N)*Es*NUM_BITS_PER_SYMBOL*10**(idx/10))/np.sqrt(2))
    BER_THEO[i] = (mu/(2*np.log2(M)))*special.erfc(np.sqrt(Es*NUM_BITS_PER_SYMBOL*10**(idx/10))/np.sqrt(2))
    i = i+1

fig, ax = plt.subplots(figsize=(10, 8)) 
ax.plot(ebno_dbs, BER_THEO, '-', label=f'Original Theoretical')
ax.plot(ebno_dbs, BER_THEO_des, '-', color='C7', label='MCSA Theoretical', linewidth=2) 
ax.plot(ebno_dbs, BER_SIM, '^', label=f'Simulation Original') 
ax.plot(ebno_dbs, BER_SIM_Pil, 'o', label=f'Neural Network') 
ax.plot(ebno_dbs, BER_SIM_Without, '>', markersize=5, color='C0', label='MCSA')
ax.plot(ebno_dbs, BER_SIM_PTS, '^', markersize=5, color='C3', label='PTS')  
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