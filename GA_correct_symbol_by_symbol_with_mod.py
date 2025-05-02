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

NUM_BITS_PER_SYMBOL = 6 
M = 2**NUM_BITS_PER_SYMBOL
N = 64  # Número de subportadoras OFDM
batch = 1  # gera 1000 vezes N.
Ntx=1000 #número de símbolos OFDM que serão gerados
Np = 2
BLOCK_LENGTH = (N-Np)*NUM_BITS_PER_SYMBOL

Eo = 1

E = (2/3)*(M-1)*Eo 

constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
constellation.show()

constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL, trainable=True)
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

def Modulation_(bits):
    constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL, trainable=True)
    mapper = sn.mapping.Mapper(constellation = constellation)
    return mapper(bits)

#%% Alocação de pilotos e sinais modulados no simbolo que será transmitido:
    
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
    
def IFFT(symboll):
    OFDM_time = np.sqrt(N) * np.fft.ifft(symboll).astype('complex64')    
    return OFDM_time

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
    eixo_x_red = np.arange(mi, ma, 0.00001) 
    y_red = []
    for jj in eixo_x_red:
        A_red = len(np.where(PAPR_final > jj)[0])/PAPR_Total_red
        y_red.append(A_red)    
    CCDF_red = y_red
    return CCDF_red


#%%

PAPR_dB_ref = 4
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
        Pilot_1 = (np.sqrt(E))
        Pilot_2 = (-np.sqrt(E))
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
     

#%% Histogram:

magnitudes = [np.abs(complex_num) for tupla in successful_pilots for complex_num in tupla]

#for Pilots:
plt.style.use('seaborn-darkgrid')
fig, ax = plt.subplots(figsize=(10, 8))

# Create a histogram
ax.hist(magnitudes, bins=100, edgecolor='black', color='#86bf91')

# Add labels and title
ax.set_xlabel('Successful Pilots', fontsize=17, fontweight='bold')
ax.set_ylabel('Relative Frequency', fontsize=17, fontweight='bold')

# Add dashed lines for mean and median
ax.axvline(np.mean(magnitudes), color='red', linestyle='--', linewidth=2, label='Mean')
ax.axvline(np.median(magnitudes), color='blue', linestyle='-.', linewidth=2, label='Median')

ax.grid(axis='both', linestyle='--', alpha=0.7, color='black')

# Configurando o fundo do gráfico como branco
ax.set_facecolor('white')

# Move the legend to a suitable location
ax.legend(loc='upper right', fontsize=17, bbox_to_anchor=(1.0, 1.0), frameon=True, facecolor='white', edgecolor='black')

ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

# Set the spines linewidth to create a margin
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Show the plot
plt.show()

# For ind_complex:
    
#filtered_test = [value for value in ind_complex if value != 1]

filtered_test = ind_complex

hist, edges = np.histogram(filtered_test, bins='auto')
x = (edges[:-1] + edges[1:]) / 2  # Midpoints of the bins
pdf_mean = expon.pdf(x, scale = np.mean(filtered_test))
pdf_std = expon.pdf(x, scale = np.std(filtered_test))

fig = plt.figure(figsize=(10, 8))
plt.plot(x, hist/len(filtered_test), label='Histogram', drawstyle='steps-mid', alpha=0.7)
plt.plot(x, pdf_mean / np.sum(pdf_mean), label='Exponential PDF (mean)')
plt.plot(x, pdf_std / np.sum(pdf_std), label='Exponential PDF (std)')
plt.legend()
plt.show()

#%% Complexity: 

filtered_test_new = np.array(filtered_test)

filtered_test_IFFT = filtered_test_new * N * np.log10(N)

hist_IFFT, edges_IFFT = np.histogram(filtered_test_IFFT, bins='auto')
x_IFFT = (edges_IFFT[:-1] + edges_IFFT[1:]) / 2  # Midpoints of the bins
pdf_mean_IFFT = expon.pdf(x_IFFT, scale = np.mean(filtered_test_IFFT))
pdf_std_IFFT = expon.pdf(x_IFFT, scale = np.std(filtered_test_IFFT))

fig, ax = plt.subplots(figsize=(10, 8))

# Plot the histogram
ax.plot(x_IFFT, hist_IFFT/len(filtered_test_IFFT), label='Complexity', drawstyle='steps-mid', alpha=1)

# Plot the Exponential PDF for standard deviation
ax.plot(x_IFFT, pdf_std_IFFT / np.sum(pdf_std_IFFT), label='Standard Deviation', linestyle='-.', color='red')

# Plot the Exponential PDF for mean
ax.plot(x_IFFT, pdf_mean_IFFT / np.sum(pdf_mean_IFFT), label='Mean', linestyle='--', color='green')

# Add labels and title
ax.set_xlabel('IFFT Complexity', fontsize=17, fontweight='bold')
ax.set_ylabel('Relative Frequency', fontsize=17, fontweight='bold')

# Add grid lines
# Add grid lines to the background
ax.grid(axis='both', linestyle='--', alpha=0.7, color='black')

# Configurando o fundo do gráfico como branco
ax.set_facecolor('white')

# Move the legend to a suitable location
ax.legend(loc='upper right', fontsize=17, bbox_to_anchor=(1.0, 1.0), frameon=True, facecolor='white', edgecolor='black')

ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

# Set the spines linewidth to create a margin
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Show the plot
plt.show()

#%% Plot 
Pilot_1_Ori = (np.sqrt(E))
Pilot_2_Ori = (-np.sqrt(E))
symbol_Ori_ = symbol(mapper_final, Pilot_1_Ori, Pilot_2_Ori)
OFDM_time_ = IFFT(symbol_Ori_)
_PAPR_dB = PAPR(OFDM_time_)
_CCDF = CCDF(_PAPR_dB)    
CCDF_red = CCDF(PAPR_final)
     
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the CCDF for the original signal
ax.semilogy(np.arange(min(_PAPR_dB), max(_PAPR_dB), 0.00001), _CCDF, '-', color='C0', label="Original Signal")

# Plot the CCDF for the GA signal with markers
ax.plot(np.arange(min(PAPR_final), max(PAPR_final), 0.00001), CCDF_red, 'o', color='C1', label="GA Signal")

# Add labels and title
ax.set_xlabel('PAPR (dB)', fontsize=17, fontweight='bold')
ax.set_ylabel('CCDF', fontsize=17, fontweight='bold')

ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.7, color='gray')
ax.yaxis.grid(True, which='minor', linestyle='--', alpha=0.5, color='gray')
ax.grid(axis='both', linestyle='--', alpha=0.7, color='gray')

# Configurando o fundo do gráfico como branco
ax.set_facecolor('white')

# Move the legend to a suitable location
ax.legend(loc='upper right', fontsize=17, bbox_to_anchor=(1.0, 1.0), frameon=True, facecolor='white', edgecolor='black')

ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

# Set the spines linewidth to create a margin
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Set y-axis limit
ax.set_ylim([1e-3, 1])

# Show the plot
plt.show()
    
#%% Generate SNR x BER:

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
        
        global OFDM_time_     
        global bits_final
        
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        pilotCarriers = create_pilot(batch_size, Np)
        # Channel
        h = np.array([1])
        self.H = np.fft.fft(h,self.N)        
        self.OFDM_RX_FD = OFDM_time_
        y = self.awgn_channel([self.OFDM_RX_FD, no]) # no = potência do ruído
        y= FFT(y)
        
        OFDM_demod = np.delete(y, pilotCarriers,axis=1)
        print(OFDM_demod.size)
        llr = self.demapper([OFDM_demod,no])
        self.Out_Ori = y
        print(llr)
        return bits_final, llr

class UncodedSystemAWGN_GA(Model): 
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
        
        global OFDM_final     
        global bits_final
        
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        pilotCarriers = create_pilot(batch_size, Np)
        # Channel
        h = np.array([1])
        self.H = np.fft.fft(h,self.N)        
        self.OFDM_RX_FD = OFDM_final
        y_ = self.awgn_channel([self.OFDM_RX_FD, no]) # no = potência do ruído
        y_= FFT(y_)
        
        OFDM_demod_ = np.delete(y_, pilotCarriers,axis=1)
        print(OFDM_demod_.size)
        llr_GA = self.demapper([OFDM_demod_,no])
        self.Out_GA = y_
        print(llr_GA)
        return bits_final, llr_GA
    

#%%
  
# Instanciando o modelo
model_uncoded_awgn = UncodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, Subcarriers=N)
model_uncoded_awgn_GA = UncodedSystemAWGN_GA(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, Subcarriers=N)

#%%

# Sionna provides a utility to easily compute and plot the bit error rate (BER).

batch_size = Ntx
SNR = np.arange(0, 20)

EBN0_DB_MIN = min(SNR) # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = max(SNR) # Maximum value of Eb/N0 [dB] for simulations

# Original Simulation:

ber_plots = sn.utils.PlotBER()
ber_plots.simulate(model_uncoded_awgn,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size = batch_size,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM = np.array(ber_plots.ber).ravel()

# GA Simulation:
    
ber_plots_GA = sn.utils.PlotBER()
ber_plots_GA.simulate(model_uncoded_awgn_GA,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size = batch_size,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM_GA = np.array(ber_plots_GA.ber).ravel()


#%% Theoretical:

M = 2**(NUM_BITS_PER_SYMBOL)
L = np.sqrt(M)
mu = 4 * (L - 1) / L  # Número médio de vizinhos
E_M = 3 / (L ** 2 - 1) # Fator de ajuste da constelação
    
ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)
BER_THEO = np.zeros((len(ebno_dbs)))

i = 0
for idx in ebno_dbs:
    BER_THEO[i] = (mu/(2*N*NUM_BITS_PER_SYMBOL))*np.sum(special.erfc(np.sqrt(np.abs(model_uncoded_awgn.H) **2
                                                            *E_M*NUM_BITS_PER_SYMBOL*10**(idx/10)) / np.sqrt(2)))
    
    i = i+1

fig, ax = plt.subplots(figsize=(10, 8))

# Plot BER for the original signal with circles
ax.plot(ebno_dbs, BER_SIM, 'o-', markersize=5, color='C1', label='Original Signal')

# Plot BER for the GA signal with crosses
ax.plot(ebno_dbs, BER_SIM_GA, 'x-', markersize=5, color='C2', label='GA Signal')

# Plot theoretical BER with a solid line
ax.plot(ebno_dbs, BER_THEO, '-', color='C0', label='Theoretical', linewidth=2)

# Add labels and title
ax.set_ylabel('Bit Error Rate (BER)', fontsize=16, fontweight='bold')
ax.set_xlabel('Eb/N0 (dB)', fontsize=16, fontweight='bold')

# Set tick parameters
ax.tick_params(axis='both', which='major', labelsize=17)

ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.7, color='gray')
ax.yaxis.grid(True, which='minor', linestyle='--', alpha=0.5, color='gray')
ax.grid(axis='both', linestyle='--', alpha=0.7, color='gray')

# Configurando o fundo do gráfico como branco
ax.set_facecolor('white')

# Move the legend to a suitable location
ax.legend(loc='upper right', fontsize=17, bbox_to_anchor=(1.0, 1.0), frameon=True, facecolor='white', edgecolor='black')

ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

# Set the spines linewidth to create a margin
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Set x-axis and y-axis limits
ax.set_xlim([EBN0_DB_MIN, EBN0_DB_MAX])
ax.set_ylim([1e-7, 1])

# Set y-axis scale to logarithmic
ax.set_yscale('log')

# Show the plot
plt.show()


#%% Plot PSD:

symbol_end_ori = symbol_Ori_
symbol_end_GA = symbol_final
symbol_output_Ori = model_uncoded_awgn.Out_Ori
symbol_output_GA = model_uncoded_awgn_GA.Out_GA

def plot_psd(signal_data_1, signal_data_2, signal_data_3, signal_data_4, fs):
    # Plot the PSD
    plt.figure(figsize=(7, 5))
    # Compute PSD using Welch method
    frequencies_1, psd_1 = signal.welch(signal_data_1, fs, nperseg=4*N)
    frequencies_2, psd_2 = signal.welch(signal_data_2, fs, nperseg=4*N)
    frequencies_3, psd_3 = signal.welch(signal_data_3, fs, nperseg=4*N)
    frequencies_4, psd_4 = signal.welch(signal_data_4, fs, nperseg=4*N)
    psd_1 = 10*np.log10(psd_1)
    psd_2 = 10*np.log10(psd_2)
    psd_3 = 10*np.log10(psd_3)
    psd_4 = 10*np.log10(psd_4)
    plt.plot(frequencies_1, psd_1, '-.', label='Original Input Signal')
    plt.plot(frequencies_2, psd_2, '--', label='GA Input Signal')
    plt.plot(frequencies_3, psd_3, '*', label='Original Output Signal')
    plt.plot(frequencies_4, psd_4, 'o', label='GA Output Signal')
    plt.title('Power Spectral Density (PSD)')
    plt.xlabel(r'Frequência Normalizada [$ \times 2\pi$ rad/amostra]')
    plt.ylabel('Power/Frequency (dB/Hz)')
    # Adding grid
    plt.grid(axis='both', linestyle='--', alpha=0.7, color='black')
    
    # Creating legend
    plt.legend(loc='upper right', fontsize=12, bbox_to_anchor=(1.0, 1.0), frameon=True, facecolor='white', edgecolor='black')
 
    # Setting spine colors and linewidth
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)
    ax.set_facecolor('white') 
    plt.show()
    return 

def plot_results(ofdm_signal_1, ofdm_signal_2, ofdm_signal_3, ofdm_signal_4):
    power_1, psd_frequencies_1 = psd(ofdm_signal_1, NFFT=4*N, Fs=1, window=mlab.window_none)
    power_2, psd_frequencies_2 = psd(ofdm_signal_2, NFFT=4*N, Fs=1, window=mlab.window_none)
    power_3, psd_frequencies_3 = psd(ofdm_signal_3, NFFT=4*N, Fs=1, window=mlab.window_none)
    power_4, psd_frequencies_4 = psd(ofdm_signal_4, NFFT=4*N, Fs=1, window=mlab.window_none)
    Pxx_dB_1 = 10*np.log10(power_1)
    Pxx_dB_2 = 10*np.log10(power_2)
    Pxx_dB_3 = 10*np.log10(power_3)
    Pxx_dB_4 = 10*np.log10(power_4)
    
    plt.figure(figsize=(7, 5))
    plt.plot(psd_frequencies_1, Pxx_dB_1, '-.', label='Original Input Signal')
    plt.plot(psd_frequencies_2, Pxx_dB_2, '--', label='GA Input Signal')
    plt.plot(psd_frequencies_3, Pxx_dB_3, '*', label='Original Output Signal')
    plt.plot(psd_frequencies_4, Pxx_dB_4, 'o', label='GA Output Signal')
    plt.xlabel(r'Normalized Frequency [$2\pi$ rad/sample]', fontsize=17)
    plt.ylabel('Power/Frequency (dB/Hz)', fontsize=17)
    plt.grid(axis='both', linestyle='--', alpha=0.7, color='black')
    plt.legend(loc='upper right', fontsize=12, bbox_to_anchor=(1.0, 1.0), frameon=True, facecolor='white', edgecolor='black')
 
    # Setting spine colors and linewidth
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)
    ax.set_facecolor('white')    
    # Displaying the plot
    plt.show()
    return

Fs = 1 # cada simbolo está sendo amostrado somente uma vez.
Ktotal = 4*N

# Original Symbol:

X = symbol_end_ori[1,:]
concatenated = np.concatenate([X[:N//2], np.zeros((Ktotal-N, 1)).ravel(), X[N//2:]])
ifft_conc = IFFT(concatenated) 

# GA Symbol:
    
X_GA = symbol_end_GA[1,:]
concatenated_GA = np.concatenate([X_GA[:N//2], np.zeros((Ktotal-N, 1)).ravel(), X_GA[N//2:]])
ifft_conc_GA = IFFT(concatenated_GA)

# Original Symbol (Output):

X_output = symbol_output_Ori[1,:]
concatenated_output = np.concatenate([X_output[:N//2], np.zeros((Ktotal-N, 1)).ravel(), X_output[N//2:]])
ifft_conc_output = IFFT(concatenated_output) 

# GA Symbol (Output):
    
X_GA_output = symbol_output_GA[1,:]
concatenated_GA_Output = np.concatenate([X_GA_output[:N//2], np.zeros((Ktotal-N, 1)).ravel(), X_GA_output[N//2:]])
ifft_conc_GA_Output = IFFT(concatenated_GA_Output)

plot_psd(ifft_conc, ifft_conc_GA, ifft_conc_output, ifft_conc_GA_Output, 1)
plot_results(ifft_conc, ifft_conc_GA, ifft_conc_output, ifft_conc_GA_Output)     
    

#%%

np.savetxt('/Users/bianc/OneDrive/Documentos/successful_pilots.csv', successful_pilots, delimiter=';', fmt='%.4f')   
np.savetxt('/Users/bianc/OneDrive/Documentos/symbol_Ori.csv', symbol_Ori_, delimiter=';', fmt='%.4f') 
np.savetxt('/Users/bianc/OneDrive/Documentos/symbol_final.csv', symbol_final, delimiter=';', fmt='%.4f')   
np.savetxt('/Users/bianc/OneDrive/Documentos/ind_complex.csv', ind_complex, delimiter=';', fmt='%.4f') 
np.savetxt('/Users/bianc/OneDrive/Documentos/_PAPR_dB.csv', _PAPR_dB, delimiter=';', fmt='%.4f') 
np.savetxt('/Users/bianc/OneDrive/Documentos/_CCDF.csv', _CCDF, delimiter=';', fmt='%.4f') 
np.savetxt('/Users/bianc/OneDrive/Documentos/PAPR_final.csv', PAPR_final, delimiter=';', fmt='%.4f') 
np.savetxt('/Users/bianc/OneDrive/Documentos/CCDF_red.csv', CCDF_red, delimiter=';', fmt='%.4f') 
np.savetxt('/Users/bianc/OneDrive/Documentos/OFDM_final.csv', OFDM_final, fmt='%.4f') 