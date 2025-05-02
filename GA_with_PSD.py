import numpy as np
import sionna as sn
import matplotlib.pyplot as plt
import gym, time, csv
from scipy import special
import tensorflow as tf
from scipy.stats import norm, poisson, cauchy, expon
from scipy import stats
from tensorflow.keras import Model
from matplotlib.mlab import psd

NUM_BITS_PER_SYMBOL = 4 
M = 2**NUM_BITS_PER_SYMBOL
N = 32  # Número de subportadoras OFDM
batch = 1  # gera 1000 vezes N.
Ntx=1000 #número de símbolos OFDM que serão gerados
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
    OFDM_time = np.sqrt(N) * np.fft.fft(symbol_).astype('complex64')    
    return OFDM_time
    
def IFFT(symbol):
    OFDM_time = np.sqrt(N) * np.fft.ifft(symbol).astype('complex64')    
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

def PSD(ofdm_sig):
    power, psd_frequencies = psd(ofdm_sig, NFFT=N, Fs=1)
    return power

#%%

PAPR_dB_ref = 5 
bits_final = np.zeros((Ntx,BLOCK_LENGTH))
mapper_final = np.zeros((Ntx,N-Np),dtype=complex)
OFDM_final = np.zeros((Ntx,N),dtype=complex)
symbol_final = np.zeros((Ntx,N),dtype=complex)
PAPR_final = np.zeros((Ntx,1))
ind_complex= np.zeros((Ntx,1))
symbol_rate = 1e3  # Symbol rate in Hz
successful_pilots = []
desvio_padrao = np.zeros((Ntx,1))

for nt in range(0,Ntx):
    ic=1 
    
    bitss = Bits(batch, BLOCK_LENGTH) 
    PSD_bitss = PSD(bitss)
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
    
#%% Histogram:

magnitudes = [np.abs(complex_num) for tupla in successful_pilots for complex_num in tupla]

#for Pilots:
plt.style.use('seaborn-darkgrid')
fig, ax = plt.subplots(figsize=(10, 8))

# Create a histogram
ax.hist(magnitudes, bins=100, edgecolor='black', color='#86bf91')

# Add labels and title
ax.set_xlabel('Successful Pilots', fontsize=17, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=17, fontweight='bold')

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
ax.plot(x_IFFT, hist_IFFT/len(filtered_test_IFFT), label='Complexity', linestyle='-', color='blue')

# Plot the Exponential PDF for standard deviation
ax.plot(x_IFFT, pdf_std_IFFT / np.sum(pdf_std_IFFT), label='Standard Deviation', linestyle='-.', color='red')

# Plot the Exponential PDF for mean
ax.plot(x_IFFT, pdf_mean_IFFT / np.sum(pdf_mean_IFFT), label='Mean', linestyle='--', color='green')

# Add labels and title
ax.set_xlabel('IFFT Complexity', fontsize=17, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=17, fontweight='bold')

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

#%% Both:
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))

# Plotting the first figure
axes[0].plot(x, hist/len(filtered_test), label='Histogram', drawstyle='steps-mid', alpha=0.7)
axes[0].plot(x, pdf_mean/np.sum(pdf_mean), label='Exponential PDF (mean)')
axes[0].plot(x, pdf_std/np.sum(pdf_std), label='Exponential PDF (std)')
axes[0].legend()
axes[0].set_title('Histogram and Exponential PDFs')
axes[1].plot(x_IFFT, hist_IFFT/len(filtered_test_IFFT), label='Histogram', drawstyle='steps-mid', alpha=0.7)
axes[1].plot(x_IFFT, pdf_mean_IFFT/np.sum(pdf_mean_IFFT), label='Exponential PDF (mean)')
axes[1].plot(x_IFFT, pdf_std_IFFT/np.sum(pdf_std_IFFT), label='Exponential PDF (std)')
axes[1].legend()
axes[1].set_title('Histogram and Exponential PDFs (IFFT)')
plt.show()

#%% Plot 

symbol_Ori_ = symbol(mapper_final, -np.sqrt(E), np.sqrt(E))
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
ax.set_ylim([1e-4, 1])

# Show the plot
plt.show()

np.savetxt('/Users/bianc/OneDrive/Documentos/successful_pilots.csv', successful_pilots, delimiter=';', fmt='%.4f')   
np.savetxt('/Users/bianc/OneDrive/Documentos/symbol_Ori.csv', symbol_Ori_, delimiter=';', fmt='%.4f')   
np.savetxt('/Users/bianc/OneDrive/Documentos/symbol_final.csv', symbol_final, delimiter=';', fmt='%.4f')   
np.savetxt('/Users/bianc/OneDrive/Documentos/ind_complex.csv', ind_complex, delimiter=';', fmt='%.4f') 
np.savetxt('/Users/bianc/OneDrive/Documentos/_PAPR_dB.csv', _PAPR_dB, delimiter=';', fmt='%.4f') 
np.savetxt('/Users/bianc/OneDrive/Documentos/_CCDF.csv', _CCDF, delimiter=';', fmt='%.4f') 
np.savetxt('/Users/bianc/OneDrive/Documentos/PAPR_final.csv', PAPR_final, delimiter=';', fmt='%.4f') 
np.savetxt('/Users/bianc/OneDrive/Documentos/CCDF_red.csv', CCDF_red, delimiter=';', fmt='%.4f') 

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
        y= (np.sqrt(1/N)*np.fft.fft(y)).astype('complex64')
        OFDM_demod = np.delete(y, pilotCarriers,axis=1)
        print(OFDM_demod.size)
        llr = self.demapper([OFDM_demod,no])
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
        y_= (np.sqrt(1/N)*np.fft.fft(y_)).astype('complex64')
        OFDM_demod_ = np.delete(y_, pilotCarriers,axis=1)
        print(OFDM_demod_.size)
        llr_GA = self.demapper([OFDM_demod_,no])
        print(llr_GA)
        return bits_final, llr_GA

#%%
  
# Instanciando o modelo
model_uncoded_awgn = UncodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, Subcarriers=N)
model_uncoded_awgn_GA = UncodedSystemAWGN_GA(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, Subcarriers=N)

#%%

# Sionna provides a utility to easily compute and plot the bit error rate (BER).

batch_size = Ntx
SNR = np.arange(0, 15)

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
E = 3 / (L ** 2 - 1) # Fator de ajuste da constelação
    
ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)
BER_THEO = np.zeros((len(ebno_dbs)))

i = 0
for idx in ebno_dbs:
    BER_THEO[i] = (mu/(2*N*NUM_BITS_PER_SYMBOL))*np.sum(special.erfc(np.sqrt(np.abs(model_uncoded_awgn.H) **2
                                                            *E*NUM_BITS_PER_SYMBOL*10**(idx/10)) / np.sqrt(2)))
    
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
ax.set_ylim([1e-5, 1])

# Set y-axis scale to logarithmic
ax.set_yscale('log')

# Show the plot
plt.show()
