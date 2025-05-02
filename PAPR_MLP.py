import tensorflow as tf
import numpy as np
import sionna as sn
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, BatchNormalization, Activation, Input , Lambda
import time
import tensorflow_probability as tfp
from scipy import special
from tensorflow.keras import Model

learning_Rate = 0.0001
initializer = tf.keras.initializers.glorot_normal(seed=25)
NUM_BITS_PER_SYMBOL = 4 # 16-QAM
N = 32  # Número de subportadoras OFDM
batch_size = 100  # gera 1000 vezes N.
Np = 2
BLOCK_LENGTH = (N-Np)*NUM_BITS_PER_SYMBOL
SNR = np.arange(0, 15)
EBN0_DB_MIN = min(SNR) # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = max(SNR) # Maximum value of Eb/N0 [dB] for simulations
ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)

    
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
        matrix[i] = np.array([1,-1])
    return matrix

def Bits(batch_size, BLOCK_LENGTH):
    binary_source = sn.utils.BinarySource()
    return binary_source([batch_size, BLOCK_LENGTH])

def Modulation(bits):
    constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
    mapper = sn.mapping.Mapper(constellation = constellation)
    return mapper(bits)

def symbol(x):
    allCarriers = create_matrix(batch_size, N)
    pilotCarriers = create_pilot(batch_size, Np)
    dataCarriers = np.delete(allCarriers, pilotCarriers,axis=1)
    
    # Allocate bits to dataCarriers and pilotCarriers:
        
    symbol = np.zeros((batch_size, N))  # the overall N subcarriers
    symbol[:, pilotCarriers] = pilot_value(batch_size, Np)  # assign values to pilots
    symbol[np.arange(batch_size)[:, None], dataCarriers] = x  # assign values to datacarriers
    

    return symbol

def IFFT_training(tensor):
    return tf.signal.ifft(tf.complex(tensor[:, 0:N*2], tensor[:, 0:N*2]))

def IFFT(OFDM_data):
    return np.sqrt(N)*np.fft.ifft(OFDM_data)

def Channel_and_Noise(tensor):
    signal_avg = tf.reduce_mean(tf.square(tf.abs(tensor)))
    noise_avg = signal_avg / 10**(SNR[n]/10)
    rayleigh = (1.0/2.0)*tfp.random.rayleigh((batch_size, N*2))
    noise_real = (tf.sqrt(noise_avg / 2.0))*tf.random.normal((batch_size , N*2), dtype=tf.float32)
    noise_imag = (tf.sqrt(noise_avg / 2.0))*tf.random.normal((batch_size , N*2), dtype=tf.float32)

    #Corruption = Channel Effect + AWGN
    #corruption =  tf.complex(tf.divide(noise_real, rayleigh) , tf.divide(noise_imag, rayleigh))
    corruption =  tf.complex(noise_real , noise_imag)
    #Equalizing done on Signal
    x = tensor + corruption
    #Recieving 
    x_fft = tf.signal.fft(x)

    return tf.keras.layers.Concatenate(axis = -1) ([tf.math.real(x_fft), tf.math.imag(x_fft)])


def PAPR(OFDM_time):
    idx = np.arange(0, 100)
    PAPR = np.zeros(len(idx))

    for i in idx:
        var = np.var(OFDM_time[i])
        peakValue = np.max(abs(OFDM_time[i])**2)
        PAPR[i] = peakValue/var 
        PAPR_dB = 10*np.log10(PAPR)
    return PAPR_dB

def CCDF(PAPR_dB):
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


def papr(y_true,y_pred):
    x = tf.square(tf.abs(y_pred))
    return 10*tf.experimental.numpy.log10(tf.reduce_mean(tf.reduce_max(x, axis = 1) / tf.reduce_mean(x, axis = 1)))

def calculate_ber_red(y_true, y_pred):
    total_errors = 0      
    for i in range(0, batch_size):
        for j in range(0, N*2):
            if y_pred[i,j] > 0.5:
                y_pred[i,j] = 1
            else:
                y_pred[i,j] = 0
                
            if y_true[i,j] != y_pred[i,j]:
                total_errors += 1                                      
    return total_errors


bits = Bits(batch_size, BLOCK_LENGTH)
modulated_bits = Modulation(bits)
ofdm_time_signal = IFFT(symbol(modulated_bits))
papr_dB = PAPR(ofdm_time_signal)
ccdf = CCDF(papr_dB)


#%% Training:
ber = np.zeros(len(SNR))

for n in range(0,len(SNR)):
    
    inputs = Input(shape = (N*2,))
    
    #ENCODER
    x1 = Dense(100, kernel_initializer=initializer, bias_initializer='random_normal') (inputs)
    x2 = BatchNormalization() (x1)
    x3 = Activation('relu') (x2)
    
    x4 = Dense(100, kernel_initializer=initializer, bias_initializer='random_normal') (x3)
    x5 = BatchNormalization() (x4)
    x6 = Activation('relu')(x5)
    
    x7 = Dense(100, kernel_initializer=initializer, bias_initializer='random_normal') (x6)
    x8 = BatchNormalization() (x7)
    x9 = Activation('relu')(x8)
    
    x10 = Dense(100, kernel_initializer=initializer, bias_initializer='random_normal') (x9)
    x11 = BatchNormalization() (x10)
    x12 = Activation('relu')(x11)
    
    x13 = Dense(N*2, kernel_initializer=initializer, bias_initializer='random_normal', activation= 'tanh') (x12)
    
    encoder = Lambda(IFFT_training, name='encoder') (x13) 
    
    
    #RECEIVER
    decoder = Lambda(Channel_and_Noise) (encoder)
    
    #DECODER
    y1 = Dense(100, kernel_initializer=initializer, bias_initializer='random_normal') (decoder)
    y2 = BatchNormalization() (y1)
    y3 = Activation('relu') (y2)
    
    y4 = Dense(100, kernel_initializer=initializer, bias_initializer='random_normal') (y3)
    y5 = BatchNormalization() (y4)
    y6 = Activation('relu')(y5)
    
    y7 = Dense(100, kernel_initializer=initializer, bias_initializer='random_normal') (y6)
    y8 = BatchNormalization() (y7)
    y9 = Activation('relu')(y8)
    
    y10 = Dense(100, kernel_initializer=initializer, bias_initializer='random_normal') (y9)
    y11 = BatchNormalization() (y10)
    y12 = Activation('relu')(y11)
    
    outputs = Dense(N*2, kernel_initializer=initializer, bias_initializer='random_normal', activation= 'sigmoid', name= 'decoder') (y12)
    
    model = keras.Model(inputs=inputs, outputs=[encoder, outputs])
    
    dataset = np.random.randint(2, size=(50000, N*2))
    valset = np.random.randint(2, size=(5000, N*2))
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    
    opt = keras.optimizers.Adam(learning_Rate)
    model.compile(optimizer= opt, loss= {'encoder' : papr, 'decoder' : 'binary_crossentropy'}, metrics={'decoder': tf.keras.metrics.BinaryAccuracy()})
    print(model.summary())
    history = model.fit(dataset, [dataset,dataset], epochs=1, batch_size=100, shuffle=True, callbacks=[callback],validation_data = [valset, valset])
    
    dataset = np.random.randint(2, size=(batch_size, N*2))
    st = time.time()
    RED, bla = model.predict(dataset, batch_size = batch_size)
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    
    # For PAPR:
        
    PAPR_RED = PAPR(RED) 
    ccdf_RED = CCDF(PAPR_RED)
    
    # For BER:
    print('dataset:', dataset.size)
    print('bla:', bla.size) 
    total_errors = calculate_ber_red(dataset, bla) 
    ber[n] = total_errors / (batch_size*N*2)  
    print('n:', n)
    
 
# Plot PAPR:

fig1 = plt.figure(figsize=(10, 8)) 
plt.plot(np.arange(min(papr_dB), max(papr_dB), 0.1), ccdf, 'x-', c=f'C0')
plt.plot(np.arange(min(PAPR_RED), max(PAPR_RED), 0.1), ccdf_RED, 'o-', c=f'C1')
plt.legend(['Original Signal', 'Reduced Signal'], loc='upper right')
plt.title('PAPR_dB x CCDF')
plt.xlabel('PAPR_dB')
plt.ylabel('CCDF')
plt.yscale('log')  
plt.ylim([1e-2, 1])
plt.grid()
plt.show()    

M = 2**(NUM_BITS_PER_SYMBOL)
L = np.sqrt(M)
mu = 4 * (L - 1) / L  # Número médio de vizinhos
E = 3 / (L ** 2 - 1)
V = 1/np.sqrt(2)
Es = 10


BER_THEO = np.zeros((len(ebno_dbs)))
i = 0
for idx in ebno_dbs:
    BER_THEO[i] = (mu/(2*np.log2(M)))*special.erfc(np.sqrt(Es*NUM_BITS_PER_SYMBOL*10**(idx/10))/np.sqrt(2))
    i = i+1


fig = plt.figure(figsize=(10, 8))
#plt.rcParams.update({'font.size': 20})
plt.plot()
plt.show()
plt.plot(ebno_dbs, BER_THEO, label='Theoretical')
plt.scatter(ebno_dbs, ber, facecolor='None', edgecolor='b', label='Autoencoder')
plt.yscale('log')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.grid(True, which="both", ls="-")
plt.title('Sistema OFDM no canal Rayleigh')
plt.legend(fontsize=12)
plt.xlim([EBN0_DB_MIN, EBN0_DB_MAX])
plt.ylim([1e-3, 1])

#%% Generate SNR x BER Original:
    
M = 2**(NUM_BITS_PER_SYMBOL)
L = np.sqrt(M)
mu = 4 * (L - 1) / L  # Número médio de vizinhos
E = 3 / (L ** 2 - 1)
V = 1/np.sqrt(2)
Es = 10

constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)

class UncodedSystemAWGN(Model): # Inherits from Keras Model
    def __init__(self, num_bits_per_symbol, block_length,Subcarriers):

        super().__init__() # Must call the Keras model initializer

        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = BLOCK_LENGTH
        self.N = Subcarriers
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
        

    # @tf.function # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):

        # no channel coding used; we set coderate=1.0
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)

        bits = self.binary_source([batch_size, BLOCK_LENGTH])
        print('bits:',bits)
        x = self.mapper(bits)
        #print('x:',x)

        # Creating the dataCarriers and pilotCarriers:
            
        allCarriers = create_matrix(batch_size, N)
        pilotCarriers = create_pilot(batch_size, Np)
        dataCarriers = np.delete(allCarriers, pilotCarriers,axis=1)

        # Allocate bits to dataCarriers and pilotCarriers:
            
        symbol = np.zeros((batch_size, N))  # the overall N subcarriers
        symbol[:, pilotCarriers] = pilot_value(batch_size, Np)  # assign values to pilots
        symbol[np.arange(batch_size)[:, None], dataCarriers] = x  # assign values to datacarriers

        # Simbolo OFDM:

        OFDM_time = np.sqrt(N)*np.fft.ifft(symbol)
        #print('OFDM_time:', OFDM_time.shape)
        # Channel
        h = np.array([1])
        #hr = h*(np.random(np.size(h))+np.imag(np.random(np.size(h))))*np.sqrt(2)
        
        self.H = np.fft.fft(h,self.N)        
        
        OFDM_RX_FD = OFDM_time
        #print('OFDM_RX_FD:', OFDM_RX_FD.shape)
        y = self.awgn_channel([OFDM_RX_FD, no])
        

        #print('y:',y.shape)
        #OFDM_demod = (np.array(y)/self.H).astype('complex64')

        OFDM_demod = y[dataCarriers] # Removendo as portadoras pilotos
        print('y:',y)
        #print('no:',no)
        #OFDM_demod = (np.sqrt(1/self.K)*np.fft.fft(y)).astype('complex64')
        print('OFDM_demod:',OFDM_demod)

        #self.demapper = self.demapper([batch_size, self.block_length])
        llr = self.demapper([OFDM_demod,no])
        #llr = tf.reshape(llr, [tf.shape(llr)[0], -1])


        print('llr:', llr.shape)

        return bits, llr
    
#%%
    
# Instanciando o modelo
model_uncoded_awgn = UncodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, Subcarriers=N)

#%%

# Sionna provides a utility to easily compute and plot the bit error rate (BER).

EBN0_DB_MIN = min(SNR) # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = max(SNR) # Maximum value of Eb/N0 [dB] for simulations
BATCH_SIZE = 1000 # How many examples are processed by Sionna in parallel

ber_plots = sn.utils.PlotBER()
ber_plots.simulate(model_uncoded_awgn,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM = np.array(ber_plots.ber).ravel()

ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)
BER_THEO = np.zeros((len(ebno_dbs)))

#%% Estimando a BER para cada Eb/No sem utilizar o Sionna

i = 0
for idx in ebno_dbs:
    BER_THEO[i] = (mu/(2*N*NUM_BITS_PER_SYMBOL))*np.sum(special.erfc(np.sqrt(np.abs(model_uncoded_awgn.H) ** 2*E*NUM_BITS_PER_SYMBOL*10**(idx/10)) / np.sqrt(2)))
    
    i = i+1


fig = plt.figure(figsize=(10, 8))
#plt.rcParams.update({'font.size': 20})
plt.plot()
plt.show()
plt.plot(ebno_dbs, BER_THEO, label='Theoretical')
plt.scatter(SNR, ber, facecolor='None', edgecolor='b', label='Autoencoder')
plt.yscale('log')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.grid(True, which="both", ls="-")
plt.title('Sistema OFDM no canal Rayleigh')
plt.legend(fontsize=12)
plt.xlim([EBN0_DB_MIN, EBN0_DB_MAX])
plt.ylim([1e-3, 1])
