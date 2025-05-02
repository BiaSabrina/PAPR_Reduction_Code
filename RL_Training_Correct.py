import os
import sionna as sn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer
from scipy import special
from tensorflow.keras import Sequential
from tensorflow.keras.layers import  Flatten



#%% configurando um sistema de comunicação simples que transmite bits modulados como símbolos QAM através de um canal AWGN:
    
# Binary source to generate uniform i.i.d. bits
binary_source = sn.utils.BinarySource()


NUM_BITS_PER_SYMBOL = 4
constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL, trainable=True) # The constellation is set to be trainable

# Mapper and demapper
mapper = sn.mapping.Mapper(constellation=constellation)
demapper = sn.mapping.Demapper("app", constellation=constellation)
rl_perturbation_var = 0.01 
# AWGN channel
awgn_channel = sn.channel.AWGN()

BATCH_SIZE = 128 # How many examples are processed by Sionna in parallel
EBN0_DB = 17.0 # Eb/N0 in dB

no = sn.utils.ebnodb2no(ebno_db=EBN0_DB,
                        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                        coderate=1.0) # Coderate set to 1 as we do uncoded transmission here

bits = binary_source([BATCH_SIZE,
                        1200]) # Blocklength
x = mapper(bits)
y = awgn_channel([x, no])
llr = demapper([y,no])

#%% entradas e saídas do canal:
    
plt.figure(figsize=(8,8))
plt.axes().set_aspect(1.0)
plt.grid(True)
plt.scatter(tf.math.real(y), tf.math.imag(y), label='Output')
plt.scatter(tf.math.real(x), tf.math.imag(x), label='Input')
plt.legend(fontsize=20);

#%% A função de perda é a entropia cruzada binária (BCE) aplicada a cada bit e a cada símbolo recebido:
    
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

print(f"BCE: {bce(bits, llr)}")

#%% Utilizando SGD e executar uma passagem para frente através do sistema e calcular a função de perda:
    
with tf.GradientTape() as tape:
    bits = binary_source([BATCH_SIZE,
                            1200]) # Blocklength
    x = mapper(bits)
    y = awgn_channel([x, no])
    llr = demapper([y,no])
    loss = bce(bits, llr)
    
gradient = tape.gradient(loss, tape.watched_variables()) # Cálculo do gradiente.

# gradient é uma lista de tensores, cada tensor correspondendo a uma variável treinável do modelo.

#Para este modelo, temos apenas um único tensor treinável: a constelação da forma [2, 2^NUM_BITS_PER_SYMBOL], 
#a primeira dimensão correspondente aos componentes reais e imaginários dos pontos da constelação.

for g in gradient:
    print(g.shape) # (2, 64)
    
optimizer = tf.keras.optimizers.Adam(1e-2) # Aplicar o gradiente aos pesos.

#Usando o otimizador, os gradientes podem ser aplicados aos pesos treináveis para atualizá-los.
optimizer.apply_gradients(zip(gradient, tape.watched_variables()));

#%%  Constelação antes e depois da aplicação do gradiente:
fig = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL).show()
fig.axes[0].scatter(tf.math.real(constellation.points), tf.math.imag(constellation.points), label='After SGD')
fig.axes[0].legend();

#%% Implementa um demapper simples baseado em rede neural que consiste em três camadas densas:

    
#Neural_Demapper = NeuralDemapper()
#Neural_Demapper.summary()
   
class End2EndSystem(Model): # Inherits from Keras Model

    def __init__(self, training):

        super().__init__() # Must call the Keras model initializer

        self.constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL, trainable=True) # Constellation is trainable
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self._demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True) # Loss function
        self.training = training

    @tf.function(jit_compile=True) # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db,perturbation_variance=tf.constant(0.0, tf.float32)):

        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                coderate=1.0)
        
        bits = self.binary_source([batch_size, 1200]) # Blocklength set to 1200 bits
        x = self.mapper(bits)
        print('x:',x)
        epsilon_r = tf.random.normal(tf.shape(x))*tf.sqrt(0.5*perturbation_variance)
        epsilon_i = tf.random.normal(tf.shape(x))*tf.sqrt(0.5*perturbation_variance)
        epsilon = tf.complex(epsilon_r, epsilon_i) # [batch size, num_symbols_per_codeword]
        x_p = x + epsilon # [batch size, num_symbols_per_codeword]

        ################
        ## Channel
        ################
        y = self.awgn_channel([x_p, no])
        print('y:',y)
        llr = self._demapper([y,no])  
        print('llr:',llr)
        if self._training:
            # Average BCE for each baseband symbol and each batch example
            bits = tf.reshape(bits, [-1, batch_size, NUM_BITS_PER_SYMBOL])
            bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(bits, llr), axis=2) # Avergare over the bits mapped to a same baseband symbol
            # The RX loss is the usual average BCE
            rx_loss = tf.reduce_mean(bce)
            # From the TX side, the BCE is seen as a feedback from the RX through which backpropagation is not possible
            bce = tf.stop_gradient(bce) # [batch size, num_symbols_per_codeword]
            x_p = tf.stop_gradient(x_p)
            p = x_p-x # [batch size, num_symbols_per_codeword] Gradient is backpropagated through `x`
            tx_loss = tf.square(tf.math.real(p)) + tf.square(tf.math.imag(p)) # [batch size, num_symbols_per_codeword]
            tx_loss = -bce*tx_loss/rl_perturbation_var # [batch size, num_symbols_per_codeword]
            tx_loss = tf.reduce_mean(tx_loss)
            return tx_loss, rx_loss
        else:
            return bits, llr
        
EBN0_DB_MIN = 0.0
EBN0_DB_MAX = 15.0


###############################
# Baseline
###############################

class Baseline(Model): # Inherits from Keras Model

    def __init__(self):

        super().__init__() # Must call the Keras model initializer

        self.constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()

    @tf.function # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):

        # no channel coding used; we set coderate=1.0
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                coderate=1.0)
        bits = self.binary_source([batch_size, 1200]) # Blocklength set to 1200 bits
        x = self.mapper(bits)
        y = self.awgn_channel([x, no])
        llr = self.demapper([y,no])
        return bits, llr

baseline = Baseline()
model = End2EndSystem(False)
ber_plots_sim = sn.utils.PlotBER()
ber_plots_sim.simulate(baseline,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Baseline",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM = np.array(ber_plots_sim.ber).ravel()
#%% Iniciando treinamento:

# Number of iterations used for training
NUM_TRAINING_ITERATIONS = 500

# Set a seed for reproducibility
tf.random.set_seed(1)

# Instantiating the end-to-end model for training
model_train = End2EndSystem(training=True)

# Adam optimizer (SGD variant)
optimizer = tf.keras.optimizers.Adam()

# Training loop
for i in range(NUM_TRAINING_ITERATIONS):
    # Forward pass
    with tf.GradientTape() as tape:
        loss = model_train(BATCH_SIZE, 15.0,
                           tf.constant(rl_perturbation_var, tf.float32)) # The model is assumed to return the BMD rate
    # Computing and applying gradients
    grads = tape.gradient(loss, model_train.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
    # Print progress
    if i % 100 == 0:
        print(f"{ i}/{NUM_TRAINING_ITERATIONS}  Loss: {loss:.2E}", end="\r")

# Save the weights in a file.
weights = model_train.get_weights()
with open('weights-neural-demapper', 'wb') as f:
    pickle.dump(weights, f)

#%% Primeiro, instanciamos o modelo para avaliação e carregamos os pesos salvos:
# Instantiating the end-to-end model for evaluation
model = End2EndSystem(training=False)
# Run one inference to build the layers and loading the weights
model(tf.constant(1, tf.int32), tf.constant(10.0, tf.float32))
with open('weights-neural-demapper', 'rb') as f:
    weights = pickle.load(f)
    model.set_weights(weights)
  

eb_no_range = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)
BER_THEO = np.zeros((len(eb_no_range)))
M = 2**(NUM_BITS_PER_SYMBOL)
L = np.sqrt(M)
mu = 4 * (L - 1) / L  # Número médio de vizinhos
E = 3 / (L ** 2 - 1)

i = 0
for idx in eb_no_range:
    
    BER_THEO[i] = (mu/(2*NUM_BITS_PER_SYMBOL))*np.sum(special.erfc(np.sqrt(E*NUM_BITS_PER_SYMBOL*10**(idx/10)) / np.sqrt(2)))
   
    i = i+1

# Computing and plotting BER

ber_plots_Trained = sn.utils.PlotBER()
ber_plots_Trained.simulate(model,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100,
                  legend="Trained model",
                  soft_estimates=True,
                  max_mc_iter=100,
                  show_fig=False);

BER_TRAINED = np.array(ber_plots_Trained.ber).ravel()

fig = plt.figure(figsize=(10, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(eb_no_range, BER_THEO, '-', label='Theoretical')
plt.plot(eb_no_range, BER_SIM, '-o', label='Simulation')
plt.plot(eb_no_range, BER_TRAINED,'x-.', label='Trained')
plt.yscale('log')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.ylim([1e-6, 1])
plt.legend(fontsize=18)