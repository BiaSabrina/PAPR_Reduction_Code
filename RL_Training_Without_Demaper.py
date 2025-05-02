import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0 # Index of the GPU to use
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense
import sionna
from sionna.channel import AWGN
from sionna.utils import BinarySource, ebnodb2no, log10, expand_to_rank, insert_dims
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper, Constellation
from sionna.utils import sim_ber
import time
from scipy import special

tempo_inicial = time.time()

#%% Simulation Parameters:

###############################################
# SNR range for evaluation and training [dB]
###############################################
ebno_db_min = 0
ebno_db_max = 20.0

###############################################
# Modulation and coding configuration
###############################################
num_bits_per_symbol = 6 # Baseline is 64-QAM
modulation_order = 2**num_bits_per_symbol
coderate = 0.5 # Coderate for the outer code
n = 1500 # Codeword length [bit]. Must be a multiple of num_bits_per_symbol
num_symbols_per_codeword = n//num_bits_per_symbol # Number of modulated baseband symbols per codeword
k = int(n*coderate) # Number of information bits per codeword

###############################################
# Training configuration
###############################################
num_training_iterations_rl_alt = 1000
training_batch_size = tf.constant(128, tf.int32) # Training batch size
rl_perturbation_var = 0.01 # Variance of the perturbation used for RL-based training of the transmitter
model_weights_path_rl_training = "awgn_autoencoder_weights_rl_training" # Filename to save the autoencoder weights once RL-based training is done

###############################################
# Evaluation configuration
###############################################
results_filename = "awgn_autoencoder_results" # Location to save the results
# Number of training iterations with RL-based training for the alternating training phase and fine-tuning of the receiver phase
training_batch_size = tf.constant(128, tf.int32) # Training batch size
rl_perturbation_var = 0.01 # Variance of the perturbation used for RL-based training of the transmitter
model_weights_path_conventional_training = "awgn_autoencoder_weights_conventional_training" # Filename to save the autoencoder weights once conventional training is done
model_weights_path_rl_training = "awgn_autoencoder_weights_rl_training" # Filename to save the autoencoder weights once RL-based training is done

###############################################
# Evaluation configuration
###############################################
results_filename = "awgn_autoencoder_results" # Location to save the results



#%% Trainable End-to-end System: RL-based Training:
    
class E2ESystemRLTraining(Model):

    def __init__(self, training):
        super().__init__()

        self._training = training

        ################
        ## Transmitter
        ################
        self._binary_source = BinarySource()
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        # Trainable constellation
        constellation = Constellation("qam", num_bits_per_symbol, trainable=True)
        self.constellation = constellation
        self._mapper = Mapper(constellation=constellation)
        if not self._training:
            self._encoder = LDPC5GEncoder(k, n)
            
        if not self._training:
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

        ################
        ## Channel
        ################
        self._channel = AWGN()

        ################
        ## Receiver
        ################
        # We use the previously defined neural network for demapping
        self._demapper = Demapper("app", constellation=constellation)

    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db, perturbation_variance=tf.constant(0.0, tf.float32)):

        no = ebnodb2no(ebno_db,num_bits_per_symbol=num_bits_per_symbol,
                                coderate=0.5)
        no = expand_to_rank(no, 2)

        ################
        ## Transmitter
        ################
        # Outer coding is only performed if not training
        
        c = self._binary_source([batch_size, n])
        
            #c = self._encoder(b)
        # Modulation
        x = self._mapper(c) # x [batch size, num_symbols_per_codeword]

        # Adding perturbation
        # If ``perturbation_variance`` is 0, then the added perturbation is null
        epsilon_r = tf.random.normal(tf.shape(x))*tf.sqrt(0.5*perturbation_variance)
        epsilon_i = tf.random.normal(tf.shape(x))*tf.sqrt(0.5*perturbation_variance)
        epsilon = tf.complex(epsilon_r, epsilon_i) # [batch size, num_symbols_per_codeword]
        x_p = x + epsilon # [batch size, num_symbols_per_codeword]
        ################
        ## Channel
        ################
        y = self._channel([x_p, no]) # [batch size, num_symbols_per_codeword]
        y = tf.stop_gradient(y) # Stop gradient here

        ################
        ## Receiver
        ################
        llr = self._demapper([y, no])

        # If training, outer decoding is not performed
        if self._training:
            # Average BCE for each baseband symbol and each batch example
            c = tf.reshape(c, [-1, num_symbols_per_codeword, num_bits_per_symbol])
            print('c:',c)
        
            llr = tf.reshape(llr, [-1, num_symbols_per_codeword, num_bits_per_symbol])
            print('llr:',llr)
            bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(c, llr), axis=2) # Avergare over the bits mapped to a same baseband symbol
            
            # From the TX side, the BCE is seen as a feedback from the RX through which backpropagation is not possible
            bce = tf.stop_gradient(bce) # [batch size, num_symbols_per_codeword]
            x_p = tf.stop_gradient(x_p)
            p = x_p-x # [batch size, num_symbols_per_codeword] Gradient is backpropagated through `x`
            tx_loss = tf.square(tf.math.real(p)) + tf.square(tf.math.imag(p)) # [batch size, num_symbols_per_codeword]
            tx_loss = -bce*tx_loss/rl_perturbation_var # [batch size, num_symbols_per_codeword]
            tx_loss = tf.reduce_mean(tx_loss)
            return tx_loss
        else:
            #llr = tf.reshape(llr, [-1, n]) # Reshape as expected by the outer decoder
            #b_hat = self._decoder(llr)
            return c,llr
  
def rl_based_training(model):
    # Optimizers used to apply gradients
        optimizer = tf.keras.optimizers.Adam()

        for i in range(num_training_iterations_rl_alt):
            # Sampling a batch of SNRs
            ebno_db = tf.random.uniform(shape=[training_batch_size], minval=ebno_db_min, maxval=ebno_db_max)
            # Forward pass
            with tf.GradientTape() as tape:
                tx_loss = model(training_batch_size, 15.0,
                                   tf.constant(rl_perturbation_var, tf.float32))
            # Computing and applying gradients
            weights = model.trainable_weights
            grads = tape.gradient(tx_loss, weights)
            optimizer.apply_gradients(zip(grads, weights))
            # Printing periodically the progress
            if i % 100 == 0:
                print('Iteration {}/{}  Loss: {:.4f}'.format(i, num_training_iterations_rl_alt, tx_loss.numpy()), end='\r') 
                
# In the next cell, an instance of the model defined previously is instantiated and trained:
def save_weights(model, model_weights_path):
    weights = model.get_weights()
    with open(model_weights_path, 'wb') as f:
        pickle.dump(weights, f)
# Fix the seed for reproducible trainings
tf.random.set_seed(1)
# Instantiate and train the end-to-end system
model = E2ESystemRLTraining(training=True)
rl_based_training(model)
# Save weights
save_weights(model, model_weights_path_rl_training)

# Range of SNRs over which the systems are evaluated
ebno_dbs = np.arange(ebno_db_min, # Min SNR for evaluation
                     ebno_db_max, # Max SNR for evaluation
                     2.0) # Step

# Utility function to load and set weights of a model
def load_weights(model, model_weights_path):
    model(1, tf.constant(10.0, tf.float32))
    with open(model_weights_path, 'rb') as f:
        weights = pickle.load(f) #Read the pickled representation of an object from the open file object file and return the reconstituted object hierarchy specified therein.
    model.set_weights(weights)

#The next cell evaluate the baseline and the two autoencoder-based communication systems, trained with different method. 
#The results are stored in the dictionary BER:

# Dictionnary storing the results
BER = {}

model_rl = E2ESystemRLTraining(training=False)
load_weights(model_rl, model_weights_path_rl_training)
_,ber = sim_ber(model_rl, ebno_dbs, batch_size=128, num_target_block_errors=1000, max_mc_iter=100)
BER['autoencoder-rl'] = ber.numpy()

with open(results_filename, 'wb') as f:
    pickle.dump((ebno_dbs, BER), f)

#%% Plot:

BER_THEO = np.zeros((len(ebno_dbs)))
M = 2**(num_bits_per_symbol)
L = np.sqrt(M)
mu = 4 * (L - 1) / L  # Número médio de vizinhos
E = 3 / (L ** 2 - 1)

i = 0
for idx in ebno_dbs:
    
    BER_THEO[i] = (mu/(2*num_bits_per_symbol))*np.sum(special.erfc(np.sqrt(E*num_bits_per_symbol*10**(idx/10)) / np.sqrt(2)))
   
    i = i+1


plt.figure(figsize=(10,8))
# Autoencoder - conventional training
plt.semilogy(ebno_dbs, BER['autoencoder-rl'], 'o-.', c=f'C2', label=f'RL Training')
plt.plot(ebno_dbs, BER_THEO, '-', label='Theoretical')
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BER")
plt.grid(which="both")
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()


#%% Time:            
    
tempo_final = time.time()
tempo_total = tempo_final - tempo_inicial
print("Tempo total da simulação: ", tempo_total, "segundos.")   
    
    
    
    
    
    
    







    
    


