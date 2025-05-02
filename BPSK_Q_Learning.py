import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import random

random.seed(42)
# ------------------------------- AMBIENTE ----------------------------------------------------------------
class Environment():
    def __init__(self, N_data=4, N_total=6, num_symbols = 16):
      self.N_data = N_data
      self.N_total = N_total
      self.num_symbols = num_symbols
      self.N_pilots = 2
      self.M = 2
      self.Eo = 1
      self.E = (2 / 3) * (self.M - 1) * self.Eo

    def transmitir_sem_pilotos(self, tx_signal_total, bits_total):
          # Transmissor sem pilotos
          bits = np.random.randint(0, 2, (self.num_symbols, self.N_total))  # Gera bits aleatórios
          symbols = 2 * bits - 1                                           # Modula em BPSK

          tx_signal = np.fft.ifft(symbols, n=self.N_total, axis=1) * np.sqrt(self.N_total)  # IFFT para sinal OFDM
          
          return bits, tx_signal

    def transmitir_com_pilotos(self, Pil, tx_signal_total_Pil, bits_total_pil, episode, symbols, state):
          # Transmissor com pilotos
          Pil = np.array(Pil)
          
          if episode == 0:
              bits = np.random.randint(0, 2, (self.N_data,)) # Tentar gerar de forma ordenada. Não precisaria gerar de forma aleatória.
              bits_total_pil.append(bits)
              symbol = 2 * bits - 1
              symbols[state] = symbol
              
          frame = np.zeros((self.N_total,), dtype=complex)
          frame[1:1+self.N_data] = symbols[state]  # Inserir dados depois dos pilotos
          frame[0] = Pil[0][0]   # primeiro elemento 
          frame[-1] = Pil[0][1]  # segundo elemento
          tx_signal = np.fft.ifft(frame, n=self.N_total) * np.sqrt(self.N_total)
          tx_signal_total_Pil[state] = tx_signal

          return bits_total_pil, tx_signal, tx_signal_total_Pil, symbols

    def receptor(self, tx_signal, bits_transmitidos, EbN0_dB, modo):
          # Receptor comum para sinais com ou sem pilotos

          # Cálculo de energia e ruído
          Es = 1 # Para o sinal normalizado
          EbN0 = 10**(EbN0_dB/10)
          N0 = Es / (np.log2(2) * EbN0)
          if modo == 'com_pilotos':
             N0 = N0 * (self.N_total / (self.N_total - self.N_pilots))
          
          tx_signal = np.array(tx_signal)
        
          # Adiciona ruído
          ruido = np.sqrt(N0/2) * (np.random.randn(*tx_signal.shape) + 1j*np.random.randn(*tx_signal.shape))
          rx_signal = tx_signal + ruido

          # FFT no receptor
          rx_symbols = np.fft.fft(rx_signal, n=self.N_total, axis=1) / np.sqrt(self.N_total)

          if modo == 'sem_pilotos':
              rx_bits = (rx_symbols.real > 0).astype(int)  # Detecção BPSK
          else:
              rx_bits = (rx_symbols[:, 1:1+self.N_data].real > 0).astype(int)  # Ignora as pilotos
              bits_transmitidos = np.array(bits_transmitidos)
          # Cálculo do BER (Bit Error Rate)
          ber = np.mean(bits_transmitidos != rx_bits)
          
          return ber, bits_transmitidos, rx_bits


    def calcular_papr(self, tx_signal):
          potencia = np.abs(tx_signal)**2
          potencia_max = np.max(potencia)
          potencia_media = np.mean(potencia)
          papr = 10 * np.log10(potencia_max / potencia_media)
          return papr

    def calcular_ccdf(self, PAPR_final):
         PAPR_final = np.array(PAPR_final).flatten()  
         PAPR_Total_red = PAPR_final.size
         eixo_x_red = np.arange(min(PAPR_final), max(PAPR_final), 0.001)
         CCDF_red = [len(np.where(PAPR_final > jj)[0]) / PAPR_Total_red for jj in eixo_x_red]
         return eixo_x_red, CCDF_red


    def plotar_ccdf(self, eixo_x1, ccdf1, eixo_x2, ccdf2):
         plt.figure(figsize=(8,6))
         plt.semilogy(eixo_x1, ccdf1, label='Sem Pilotos')
         plt.semilogy(eixo_x2, ccdf2, label='Com Pilotos')
         plt.grid(True, which='both')
         plt.xlabel('PAPR [dB]')
         plt.ylabel('CCDF')
         plt.title('Comparação CCDF - Sem e Com Pilotos')
         plt.ylim(1e-4, 1)
         plt.legend()
         plt.show()

    def step(self, act, state, papr_total_Pil, tx_signal_total_Pil, bits_total_pil, episode, symbols, Pilots1, Pilots2):
          
          Pilots = []
          Pilots.append((Pilots1[act], Pilots2[act]))
          bits_total_pil, tx_signal_pil, tx_signal_total_Pil, symbols = self.transmitir_com_pilotos(Pilots, 
                                                tx_signal_total_Pil, bits_total_pil, episode, symbols, state)
          
          papr_val_Pil = env.calcular_papr(tx_signal_pil)
          
          #papr_total_Pil[state].append(papr_val_Pil)
          
          
          if papr_total_Pil[state] is None or papr_val_Pil <= papr_total_Pil[state]:

        #if papr_val_Pil < papr_total_Pil[state]: # Sempre guardar a menor delas (papr_total_Pil). O mesmo payload.
            #print('entrei')
            #print(papr_total_Pil[state])
             papr_total_Pil[state] = papr_val_Pil
             r = 1
          else:
             r = -1
                  
          
          #papr_total_Pil.append(papr_val_Pil)  
          
          next_state = state + 1
          done = False
          if next_state == (self.num_symbols):
              next_state = 0
              done = True

          return bits_total_pil, papr_total_Pil, tx_signal_total_Pil, symbols, Pilots1, Pilots2, next_state, r, done

    def reset(self):
          return 0

env = Environment()

#----------------------AGENTE------------------------------------------------#

# Ordem de alocação do Wi-Fi. 64/Q-Psk/2 pilots

def Agent(env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodios):
    # Inicializar Q(s,a) arbitrariamente
    num_states = env.M**(env.N_total - env.N_pilots)
    num_actions = env.M**(env.N_pilots)
    #num_states = env.num_symbols
    #num_actions = int(env.num_symbols/(env.M**(env.N_pilots)))
    Q = np.zeros((num_states, num_actions))
    symbols = np.zeros((num_states, env.N_total - env.N_pilots))
    tx_signal_total_Pil = np.zeros((num_states,env.N_total), dtype=complex)
    papr_total_Pil = [None for _ in range(num_states)]
    bits_total_pil = []
    total_rewards = []
    Pilots1 = np.random.rand(num_actions) + 1j*np.random.rand(num_actions)
    Pilots2 = np.random.rand(num_actions) + 1j*np.random.rand(num_actions)
    
    for episodio in range(episodios):
        done = False
        s = env.reset()

        while not done:
            # Escolha a ação usando política epsilon-greedy
            if np.random.rand() < epsilon:
                a = np.random.choice(num_actions) # ação aleatória
            else:
                a = np.argmax(Q[s, :]) # melhor ação

            # Executar ação a no ambiente
            bits_pil, papr_total_Pil, tx_signal_total_Pil, symbols, Pilots1, Pilots2, next_s, r, done = env.step(a, s, 
                                        papr_total_Pil, tx_signal_total_Pil, bits_total_pil, episodio, symbols, Pilots1, Pilots2)

            # Atualizar Q(s,a)
            Q[s, a] += alpha * (r + gamma * np.max(Q[next_s, :]) - Q[s, a])
            
            total_rewards.append(r)
            # Atualizar estado
            s = next_s
            print(r)
        # Decaimento do epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_dec
        print(epsilon)
    return bits_pil, papr_total_Pil, tx_signal_total_Pil, Q, total_rewards, symbols, Pilots1, Pilots2

bits_pil, papr_total_Pil, tx_signal_total_Pil, Q, total_rewards, symbols, Pilots1, Pilots2 = Agent(env, 0.9, 0.85, 1, 0.1, 0.9995, 10000)
#print('Q-Table:', Q)
#-----------------------------------------------------------------------------------#

# BER:
tx_signal_total = []
bits_total = []
papr_total = []


bits_total, tx_signal_total = env.transmitir_sem_pilotos(tx_signal_total, bits_total)

if __name__ == "__main__":

    EbN0_dB = np.arange(0, 11)
    ber_sem_pilotos = []
    ber_com_pilotos = []
    
    for ebn0 in EbN0_dB:
        ber1, bits_transmitidos1, rx_bits1 = env.receptor(tx_signal_total, bits_total, ebn0, modo='sem_pilotos')
        ber_sem_pilotos.append(ber1)
        
        ber2, bits_transmitidos2, rx_bits2 = env.receptor(tx_signal_total_Pil, bits_pil, ebn0, modo='com_pilotos')
        ber_com_pilotos.append(ber2)

    # BER teórica para BPSK
    EbN0_lin = 10**(EbN0_dB/10)
    ber_theoretical = 0.5 * erfc(np.sqrt(EbN0_lin))
    ber_theoretical_Pil = 0.5 * erfc(np.sqrt(EbN0_lin*((env.N_total - env.N_pilots)/env.N_total)))

    # Plota resultado
    plt.semilogy(EbN0_dB, ber_theoretical, '-', label='Teórica Sem Pilotos')
    plt.semilogy(EbN0_dB, ber_theoretical_Pil, '--', label='Teórica Com Pilotos')
    plt.semilogy(EbN0_dB, ber_sem_pilotos, 'o', label='Sem Pilotos')
    plt.semilogy(EbN0_dB, ber_com_pilotos, 's', label='Com Pilotos')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.grid(True)
    plt.legend()
    plt.show()
    

#%% PAPR:


# Sem pilotos
for i in range(env.num_symbols):
    papr_w_Pil = env.calcular_papr(tx_signal_total[i])
    papr_total.append(papr_w_Pil)
eixo_x_w_Pil, ccdf_w_Pil = env.calcular_ccdf(papr_total)

# Com pilotos
eixo_x_Pil, ccdf_Pil = env.calcular_ccdf(papr_total_Pil)

env.plotar_ccdf(eixo_x_w_Pil, ccdf_w_Pil, eixo_x_Pil, ccdf_Pil)


#%% Usando a tabela Q:
    
batch_size = symbols.shape[0]
frames = np.zeros((batch_size, env.N_total), dtype=complex)

for ii in range(batch_size):
    act = np.argmax(Q[ii, :])
    Pilots = []
    Pilots.append((Pilots1[act], Pilots2[act]))
    frames[ii, 1:1+env.N_data] = symbols[ii]
    frames[ii, 0] = Pilots[0][0]
    frames[ii, -1] = Pilots[0][1]

tx_signals = np.fft.ifft(frames, n=env.N_total, axis=1) * np.sqrt(env.N_total)

papr_test = []
for i in range(env.num_symbols):
    papr_Pil = env.calcular_papr(tx_signals[i])
    papr_test.append(papr_Pil)
eixo_test, ccdf_test = env.calcular_ccdf(papr_test)

