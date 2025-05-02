import tensorflow as tf
import numpy as np
from scipy import special
import sionna as sn
import scipy as sp
import matplotlib.pyplot as plt
import gym
from tensorflow.keras import Model
import pickle

NUM_BITS_PER_SYMBOL = 4 # QAM
K = 4
N = 256 # número de subportadoras OFDM
rl_perturbation_var = 0.01
model_weights_path_rl_training = "awgn_autoencoder_weights_rl_training"
num_training_iterations_rl_alt = 1000
BLOCK_LENGTH = N*NUM_BITS_PER_SYMBOL
perturbation_variance=tf.constant(0.0, tf.float32)
batch_size = 1000
cp_length = 16  # comprimento do prefixo cíclico

#%% Modulation OFDM with Sionna

constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
binary_source = sn.utils.BinarySource()
mapper = sn.mapping.Mapper(constellation = constellation)

bits = binary_source([batch_size, BLOCK_LENGTH])

#%% Mapper
    
x = mapper(bits)

OFDM_data = np.fft.ifft(x, N*K)

#%% Geração prefixo cíclico para cada símbolo OFDM

OFDM_data_with_cp = np.zeros((batch_size, N*K+cp_length), dtype=np.complex64)

for ii in range(batch_size):
    data = OFDM_data[ii]
    CP = np.zeros(cp_length, dtype=np.complex64)
    CP[:] = data[-cp_length:]
    data_with_cp = np.concatenate((CP, data))
    OFDM_data_with_cp[ii] = data_with_cp


#%% Find PAPR

idx = np.arange(0, 1000)
PAPR = np.zeros(len(idx))

for i in idx:
    var = np.var(OFDM_data_with_cp[i])
    peakValue = np.max(abs(OFDM_data_with_cp[i])**2)
    PAPR[i] = peakValue/var
    
PAPR_dB = 10*np.log10(PAPR)


#%% Find CCDF

N = len(PAPR_dB)

ma = max(PAPR_dB)
mi = min(PAPR_dB)

eixo_x = np.arange(mi, ma, 0.1)

y = []

for j in eixo_x:
    A = len(np.where(PAPR_dB > j)[0])/N
    y.append(A) #Adicionar A na lista y.
    
CCDF = y

#%% RL Training: 
    
#%% Add perturbation:
    
epsilon_r = tf.random.normal(tf.shape(x))*tf.sqrt(0.5*perturbation_variance)
epsilon_i = tf.random.normal(tf.shape(x))*tf.sqrt(0.5*perturbation_variance)
epsilon = tf.complex(epsilon_r, epsilon_i) # [batch size, num_symbols_per_codeword]
x_p = x + epsilon # [batch size, num_symbols_per_codeword]
OFDM_data_RL = np.fft.ifft(x_p, N*K)

#%% Geração prefixo cíclico para cada símbolo OFDM

OFDM_data_with_cp_RL = np.zeros((batch_size, N*K+cp_length), dtype=np.complex64)

for ii in range(batch_size):
    data_RL = OFDM_data_RL[ii]
    CP = np.zeros(cp_length, dtype=np.complex64)
    CP[:] = data[-cp_length:]
    data_with_cp_RL = np.concatenate((CP, data_RL))
    OFDM_data_with_cp_RL[ii] = data_with_cp_RL

#%% Loss calculation:

class E2ESystemRLTraining(Model):

    def __init__(self, training):
     
        super().__init__() # Must call the Keras model initializer
        self.training = training
        #################
        # Loss function
        #################
        if self.training:
            self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=(True))
  
    # @tf.function # Enable graph execution to speed things up
    def __call__(self, batch_size, perturbation_variance=tf.constant(0.0, tf.float32)):
        
        # Find PAPR

        idx = np.arange(0, 1000)
        PAPR_RL = np.zeros(len(idx))

        for i in idx:
            var_RL = np.var(OFDM_data_with_cp_RL[i])
            peakValue_RL = np.max(abs(OFDM_data_with_cp_RL[i])**2)
            PAPR_RL[i] = peakValue_RL/var_RL
            
        PAPR_dB_RL = 10*np.log10(PAPR_RL)

        # Find CCDF

        N = len(PAPR_dB_RL)

        ma = max(PAPR_dB_RL)
        mi = min(PAPR_dB_RL)

        eixo_x_RL = np.arange(mi, ma, 0.1)

        y_RL = []

        for j in eixo_x_RL:
            A_RL = len(np.where(PAPR_dB_RL > j)[0])/N
            y_RL.append(A_RL) #Adicionar A na lista y.
            
        CCDF = y_RL
        
        # If training, outer decoding is not performed
        if self.training:
            # Average BCE for each baseband symbol and each batch example
            #c = tf.reshape(c, [-1, , num_bits_per_symbol])
            #llr = tf.reshape(llr, [-1, batch_size, self.block_length])
            bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(bits, CCDF)) # Avergare over the bits mapped to a same baseband symbol
            #print('bce',bce)
            # From the TX side, the BCE is seen as a feedback from the RX through which backpropagation is not possible
            p = x_p-x # [batch size, num_symbols_per_codeword] Gradient is backpropagated through `x`
            tx_loss = tf.square(tf.math.real(p)) + tf.square(tf.math.imag(p)) # [batch size, num_symbols_per_codeword]
            tx_loss = -bce*tx_loss/rl_perturbation_var # [batch size, num_symbols_per_codeword]
            tx_loss = tf.reduce_mean(tx_loss)
            return tx_loss
        else: 
            
            return bits,OFDM_data_with_cp_RL
  
def rl_based_training(model):
    # Optimizers used to apply gradients
        optimizer = tf.keras.optimizers.Adam()

        for i in range(num_training_iterations_rl_alt):
            # Sampling a batch of SNRs
            ebno_db = tf.random.uniform(shape=[batch_size], minval=mi, maxval=ma)
            # Forward pass
            with tf.GradientTape() as tape:
                tx_loss = model(batch_size,tf.constant(rl_perturbation_var, tf.float32))
            # Computing and applying gradients
            weights = model.trainable_weights
            grads = tape.gradient(tx_loss, weights)
            optimizer.apply_gradients(zip(grads, weights))
            # Printing periodically the progress
            if i % 100 == 0:
                print('Iteration {}/{}  BCE: {:.4f}'.format(i, num_training_iterations_rl_alt, tx_loss.numpy()), end='\r') 
                
# In the next cell, an instance of the model defined previously is instantiated and trained:
def save_weights(model, model_weights_path):
    weights = model.get_weights()
    with open(model_weights_path, 'wb') as f:
        pickle.dump(weights, f)
# Fix the seed for reproducible trainings
tf.random.set_seed(1)
# Instantiate and train the end-to-end system
model = E2ESystemRLTraining(True)
rl_based_training(model)
# Save weights
save_weights(model, model_weights_path_rl_training)

# Utility function to load and set weights of a model
def load_weights(model, model_weights_path):
    model(1, tf.constant(10.0, tf.float32))
    with open(model_weights_path, 'rb') as f:
        weights = pickle.load(f) #Read the pickled representation of an object from the open file object file and return the reconstituted object hierarchy specified therein.
    model.set_weights(weights)

model_stop = E2ESystemRLTraining(False)

#%%  Plot PAPR x CCDF

fig = plt.figure(figsize=(8, 7))
plt.semilogy(eixo_x, CCDF,"-^r",linewidth=2.5,label="Baseline")
plt.semilogy(eixo_x, model_stop.CCDF_RL,"-^r",linewidth=2.5,label="RL Training")
plt.legend(loc="lower left")
plt.xlabel('PAPR (dB)')
plt.ylabel('CCDF')
plt.title('CCDF x PAPR')
plt.grid()
plt.ylim([1e-2, 1])

