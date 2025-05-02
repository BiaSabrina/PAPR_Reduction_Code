import numpy as np
import sionna as sn
import matplotlib.pyplot as plt
import csv
import time

NUM_BITS_PER_SYMBOL = 4 
N = 32  # Número de subportadoras OFDM
batch_size = 1000  # gera 1000 vezes N.
Np = 2
BLOCK_LENGTH = (N-Np)*NUM_BITS_PER_SYMBOL
call_count = 0  # Variável global para contar as chamadas
call_count1 = 0
start_time = time.time()

#%% Functions:
    
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
        matrix[i] = np.array([-1,1])
    return matrix

def pilot_value_change(batch_size, Np, Pilot_1, Pilot_2):
    matrix = np.zeros((batch_size, Np), dtype=int)
    for i in range(batch_size):
        matrix[i] = np.array([Pilot_1, Pilot_2])
    return matrix


def Bits(batch_size, BLOCK_LENGTH):
    global call_count1
    binary_source = sn.utils.BinarySource()
    call_count1 += 1 
    return binary_source([batch_size, BLOCK_LENGTH])

def Modulation(bits):
    constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
    mapper = sn.mapping.Mapper(constellation = constellation)   
    return mapper(bits)

def symbol(x, Pilot_1, Pilot_2):  
    #print('ENTREI')
    allCarriers = create_matrix(batch_size, N)
    pilotCarriers = create_pilot(batch_size, Np)
    dataCarriers = np.delete(allCarriers, pilotCarriers,axis=1)       
    # Allocate bits to dataCarriers and pilotCarriers:            
    symbol = np.zeros((batch_size, N), dtype=complex)  # the overall N subcarriers
    symbol[:, pilotCarriers] = pilot_value_change(batch_size, Np, Pilot_1, Pilot_2)  # assign values to pilots
    symbol[np.arange(batch_size)[:, None], dataCarriers] = x # assign values to datacarriers
    return symbol

def IFFT(symbol):
    global call_count
    OFDM_time = np.sqrt(N) * np.fft.ifft(symbol).astype('complex64')
    call_count += 1  # Incrementa o contador a cada chamada
    #print(call_count)  # Imprime o número de chamadas
    return OFDM_time

def PAPR(info):
    idy = np.arange(0, batch_size)
    PAPR_red = np.zeros(len(idy))

    for i in idy:
        var_red = np.var(info[i])
        peakValue_red = np.max(abs(info[i])**2)
        PAPR_red[i] = peakValue_red / var_red

    PAPR_dB_red = 10 * np.log10(PAPR_red)
    return PAPR_dB_red


def CCDF(PAPR_dB_red):
    PAPR_Total_red = len(PAPR_dB_red)
    ma_red = max(PAPR_dB_red)
    mi_red = min(PAPR_dB_red)
    eixo_x_red = np.arange(mi_red, ma_red, 0.01)
    y_red = []
    for jj in eixo_x_red:
        A_red = len(np.where(PAPR_dB_red > jj)[0])/PAPR_Total_red
        y_red.append(A_red) #Adicionar A na lista y.       
    CCDF_red = y_red
    return CCDF_red
   


#%% PAPR reduction with GA:

PAPR_dB = PAPR(IFFT(symbol(Modulation(Bits(batch_size, BLOCK_LENGTH)), -1, 1)))
   
PAPR_dB_ref = 5
_PAPR_dB = PAPR_dB
PAPR_dB_red = np.zeros(len(_PAPR_dB))

done = False
Pilot_1_values = []
Pilot_2_values = []

while not done:
    Pilot_1 = round(np.random.randn(), 2)
    Pilot_2 = round(np.random.randn(), 2)
    
    Pilot_1_values.append(Pilot_1)
    Pilot_2_values.append(Pilot_2)
   
    for idw in range(0, batch_size):       
        if PAPR_dB_ref > _PAPR_dB[idw]:         
            PAPR_dB_red[idw] = _PAPR_dB[idw] 
            print('PAPR_dB_red:', PAPR_dB_red)
    _PAPR_dB = PAPR(IFFT(symbol(Modulation(Bits(batch_size, BLOCK_LENGTH)), Pilot_1, Pilot_2)))
    if np.all(PAPR_dB_red != 0):
        done = True
CCDF_red = CCDF(PAPR_dB_red)
CCDF = CCDF(PAPR_dB)     

# Calcule os valores máximo e mínimo para cada variável
max_Pilot_1 = max(Pilot_1_values)
min_Pilot_1 = min(Pilot_1_values)

max_Pilot_2 = max(Pilot_2_values)
min_Pilot_2 = min(Pilot_2_values)

# Salve os valores em um arquivo CSV
with open('valores.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Escreva os cabeçalhos das colunas
    writer.writerow(['Variável', 'Valores', 'Máximo', 'Mínimo'])
    
    # Escreva os valores máximo e mínimo para cada variável
    writer.writerow(['Pilot_1', Pilot_1_values, max_Pilot_1, min_Pilot_1])
    writer.writerow(['Pilot_2', Pilot_2_values, max_Pilot_2, min_Pilot_2])

print("Valores salvos com sucesso no arquivo 'valores.csv'.")

#  Plot PAPR x CCDF

fig = plt.figure(figsize=(10, 8))
plt.semilogy(np.arange(min(PAPR_dB), max(PAPR_dB), 0.01), CCDF,'x-', c=f'C0',label="Original Signal")
plt.plot(np.arange(min(PAPR_dB_red), max(PAPR_dB_red), 0.01), CCDF_red,'o-', c=f'C1',label="GA Signal")
plt.legend(loc="lower left")
plt.xlabel('PAPR (dB)', fontsize=17, fontweight='bold')
plt.ylabel('CCDF', fontsize=17, fontweight='bold')
plt.grid()
plt.ylim([1e-2, 1])


#%% Complexity:
    
detection_times = []
comp = []

Rep = np.arange(0, call_count)

for sample_size in Rep:
    # Calculate the value of comp for the given sample size
    comp_value =  sample_size * (batch_size * N*np.log2(N))
    comp.append(comp_value)

    # Measure the execution time
    elapsed_time = (time.time() - start_time)/60
    detection_times.append(elapsed_time)
    
#detection_times_minutes = [time / 60 for time in detection_times]  # Convert seconds to minutes

fig = plt.figure(figsize=(10, 8))
plt.plot(comp, detection_times)
plt.xlabel('Complexity', fontsize=17, fontweight='bold')
plt.ylabel('Execution Time (minutes)', fontsize=17, fontweight='bold')
plt.title('Complexity vs. Execution Time', fontsize=17, fontweight='bold')
plt.grid(True)
