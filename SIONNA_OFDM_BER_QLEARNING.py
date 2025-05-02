
# Import TensorFlow and NumPy
import tensorflow as tf
import numpy as np
from scipy import special

import sionna as sn

import matplotlib.pyplot as plt

# for performance measurements
import time

# For the implementation of the Keras models
from tensorflow.keras import Model



#%%

NUM_BITS_PER_SYMBOL = 4 # QAM
K = 1024
BLOCK_LENGTH = K*NUM_BITS_PER_SYMBOL
M = 2**(NUM_BITS_PER_SYMBOL)
L = np.sqrt(M)
mu = 4 * (L - 1) / L  # Número médio de vizinhos
E = 3 / (L ** 2 - 1)
V = 1/np.sqrt(2)
Es = 10

constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)

constellation.show();


#%%

class UncodedSystemAWGN(Model): # Inherits from Keras Model
    def __init__(self, num_bits_per_symbol, block_length,Subcarriers):
        """
        A keras model of an uncoded transmission over the AWGN channel.

        Parameters
        ----------
        num_bits_per_symbol: int
            The number of bits per constellation symbol, e.g., 4 for QAM16.

        block_length: int
            The number of bits per transmitted message block (will be the codeword length later).

        Input
        -----
        batch_size: int
            The batch_size of the Monte-Carlo simulation.

        ebno_db: float
            The `Eb/No` value (=rate-adjusted SNR) in dB.

        Output
        ------
        (bits, llr):
            Tuple:

        bits: tf.float32
            A tensor of shape `[batch_size, block_length] of 0s and 1s
            containing the transmitted information bits.

        llr: tf.float32
            A tensor of shape `[batch_size, block_length] containing the
            received log-likelihood-ratio (LLR) values.
        """

        super().__init__() # Must call the Keras model initializer

        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = block_length
        self.K = Subcarriers
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

        bits = self.binary_source([batch_size, self.block_length]) # Blocklength set to 1024 bits
        x = self.mapper(bits)
        #OFDM_data = np.sqrt(self.K)*np.fft.ifft(x)
        
        #meanSquareValue = outputiFFT*outputiFFT'/length(outputiFFT);
        #peakValue = max(outputiFFT.*conj(outputiFFT));
        
        
        # Channel
        h = np.array([1,0.7,0.5])
        #hr = h*(np.random(np.size(h))+np.imag(np.random(np.size(h))))*np.sqrt(2)
        
        self.H = np.fft.fft(h,self.K)        
        
        OFDM_RX_FD = self.H*x
        
        y = self.awgn_channel([OFDM_RX_FD, no])
        
        #%% Q-Learning:
            
        for ii in len(y):
            Acquire HLSqp for all transmitters p and receivers q
            Acquire and adjust Ce using (9) and (26)
            for every (q, p) pair do
                while |Cb(k)
                    qp | > Ce for any k do
                    Select random subcarrier k from {0, . . . , K − M}
                    Initialize state S(k)
                        while A(k) != φ do
                            Select action a from A(k) using E-greedy
                            Observe S
                            0
                            (k) using (18) and (19)
                            Compute reward r(S(i), a) using (22)
                            Update quality Q(S(k), a) using (24)
                            Update state S(k) ← S
                        end while
                end while
            end for
            Compute ∆F using (25)
            Update F ← F + ∆F
        end for
        
        #%% Return to demod:
        
        OFDM_demod = (np.array(y)/self.H).astype('complex64')

        #OFDM_demod = (np.sqrt(1/self.K)*np.fft.fft(y)).astype('complex64')
        
        llr = self.demapper([OFDM_demod,no])
        return bits, llr
    
#%%
    
# Instanciando o modelo
model_uncoded_awgn = UncodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, Subcarriers=K)

#%%

# Sionna provides a utility to easily compute and plot the bit error rate (BER).

EBN0_DB_MIN = -3.0 # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = 15.0 # Maximum value of Eb/N0 [dB] for simulations
BATCH_SIZE = 1000 # How many examples are processed by Sionna in parallel

ber_plots = sn.utils.PlotBER("AWGN")
ber_plots.simulate(model_uncoded_awgn,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=1000, # simulate until 100 block errors occured
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
    BER_THEO[i] = (mu/(2*K*NUM_BITS_PER_SYMBOL))*np.sum(special.erfc(np.sqrt(np.abs(model_uncoded_awgn.H) ** 2*E*NUM_BITS_PER_SYMBOL*10**(idx/10)) / np.sqrt(2)))
    #sigma = np.sqrt(Es/(2*NUM_BITS_PER_SYMBOL*10**(idx/10)))
    #gamma = E*np.abs(model_uncoded_awgn.H)**2*Es/sigma**2
    
    #BER_THEO[i] = mu /(2*K*NUM_BITS_PER_SYMBOL)*(1-np.sqrt(gamma/(2+gamma)))
    
    #paprSymbol(i) = peakValue/meanSquareValue; 
    
    i = i+1

fig = plt.figure(figsize=(16, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(ebno_dbs, BER_THEO, label='Theoretical')
plt.scatter(ebno_dbs, BER_SIM, facecolor='None', edgecolor='r', label='Simulation')
plt.yscale('log')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.grid(True, which="both", ls="-")
plt.legend(fontsize=18)
plt.xlim([EBN0_DB_MIN, EBN0_DB_MAX])
plt.ylim([1e-5, 1])
