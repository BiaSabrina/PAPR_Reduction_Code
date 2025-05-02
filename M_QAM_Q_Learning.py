import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import sionna as sn
from scipy import special
import tensorflow as tf

# ------------------------------- AMBIENTE ----------------------------------------------------------------
class Environment():
    def __init__(self, N_data=62, N_total=64, num_symbols = 250):
      self.N_data = N_data
      self.N_total = N_total
      self.M = 4
      self.NUM_BITS_PER_SYMBOL = np.log2(self.M)
      self.num_symbols = num_symbols
      self.N_pilots = 2
      self.Eo = 1
      self.E = (2 / 3) * (self.M - 1) * self.Eo
      self.BLOCK_LENGTH_Ori = (self.N_total) * self.NUM_BITS_PER_SYMBOL
      self.BLOCK_LENGTH = (self.N_total - self.N_pilots) * self.NUM_BITS_PER_SYMBOL
      self.binary_source = sn.utils.BinarySource()
      self.constellation = sn.mapping.Constellation("qam", self.NUM_BITS_PER_SYMBOL, normalize=True)
      self.mapper = sn.mapping.Mapper(constellation=self.constellation)
      self.demapper = sn.mapping.Demapper("app", constellation=self.constellation, hard_out=True)
      self.awgn_channel = sn.channel.AWGN()

    def transmitir_sem_pilotos(self, tx_signal_total, bits_total):
          # Transmissor sem pilotos
          bits = self.binary_source([self.num_symbols, 128])
          symbols = self.mapper(bits)

          tx_signal = np.fft.ifft(symbols, n=self.N_total, axis=1) * np.sqrt(self.N_total)  # IFFT para sinal OFDM
          
          return bits, tx_signal

    def transmitir_com_pilotos(self, Pil, tx_signal_total_Pil, bits_total_pil, episode, symbols, state):
          # Transmissor com pilotos
          Pil = np.array(Pil)
          
          if episode == 0:
              bits = self.binary_source([1, 124])
              bits_total_pil.append(bits)
              symbol = self.mapper(bits)
              symbols[state] = symbol
              
          frame = np.zeros((self.N_total,), dtype=complex)
          frame[1:1+self.N_data] = symbols[state]  # Inserir dados depois dos pilotos
          frame[0] = Pil[0][0]   # primeiro elemento 
          frame[-1] = Pil[0][1]  # segundo elemento
          tx_signal = np.fft.ifft(frame, n=self.N_total) * np.sqrt(self.N_total)
          tx_signal_total_Pil[state] = tx_signal

          return bits_total_pil, tx_signal, tx_signal_total_Pil, symbols

    def receptor(self, tx_signal, bits_transmitidos, EbN0_dB, modo):
          Es = 1 # Para o sinal normalizado
          EbN0 = 10**(EbN0_dB/10)
          N0 = Es / (np.log2(self.M) * EbN0)
          if modo == 'com_pilotos':
             N0 = N0 * (self.N_total / (self.N_total - self.N_pilots))
          N0 = tf.cast(N0, tf.float32)
          
          rx_signal = self.awgn_channel([tx_signal, N0])
          
          rx_symbols = np.fft.fft(rx_signal, n=self.N_total, axis=1) / np.sqrt(self.N_total)
          rx_symbols = rx_symbols.astype(np.complex64)
          
          print('rx_symbols:', rx_symbols)
          if modo == 'sem_pilotos':
              rx_bits_binary = self.demapper([rx_symbols, N0])
          else:
              rx_symbols = rx_symbols[:, 1:1+self.N_data]
              rx_bits_binary = self.demapper([rx_symbols, N0]) # Ignora as pilotos
              bits_transmitidos = tf.squeeze(bits_transmitidos, axis=1)
          
          ber = np.mean(bits_transmitidos != rx_bits_binary)
          
          return ber, bits_transmitidos, rx_bits_binary


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

          if len(papr_total_Pil) == 0:
              r = 0  # Primeira vez, não tem como comparar. Pode definir recompensa 0.
          else:
              if papr_val_Pil < papr_total_Pil[-1]:
                  r = 1
              else:
                  r = -1

          papr_total_Pil.append(papr_val_Pil)  
          
          next_state = state + 1
          done = False
          if next_state == self.num_symbols:
              next_state = 0
              done = True

          return bits_total_pil, papr_total_Pil, tx_signal_total_Pil, symbols, Pilots1, Pilots2, next_state, r, done

    def reset(self):
          return 0

env = Environment()

#----------------------AGENTE------------------------------------------------#

def Agent(env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodios):
    # Inicializar Q(s,a) arbitrariamente
    num_states = env.num_symbols
    num_actions = 50#env.M**(env.N_pilots)
    Q = np.zeros((num_states, num_actions))
    symbols = np.zeros((num_states, env.N_total - env.N_pilots), dtype=complex)
    tx_signal_total_Pil = np.zeros((num_states,env.N_total), dtype=complex)
    papr_total_Pil = []
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

        # Decaimento do epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_dec
        print('epsilon:', epsilon)
    return bits_pil, papr_total_Pil, tx_signal_total_Pil, Q, total_rewards, symbols, Pilots1, Pilots2

bits_pil, papr_total_Pil, tx_signal_total_Pil, Q, total_rewards, symbols, Pilots1, Pilots2 = Agent(env, 0.9, 0.8, 1, 0.1, 0.995, 5000)
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

    # BER teórica:
        
    M = 2**(env.NUM_BITS_PER_SYMBOL)
    L = np.sqrt(M)
    mu = 4 * (L - 1) / L  # Número médio de vizinhos
    Es = 3 / (L ** 2 - 1) # Fator de ajuste da constelação
        
    #ebno_dbs = np.linspace(min(EbN0_dB), max(EbN0_dB), 20)
    BER_THEO = np.zeros((len(EbN0_dB)))
    BER_THEO_des = np.zeros((len(EbN0_dB)))

    i = 0
    for idx in EbN0_dB:
        #BER_THEO[i] = (mu/(2*(N-Np)*NUM_BITS_PER_SYMBOL))*np.sum(special.erfc(np.sqrt(np.abs(model_uncoded_awgn.H)**2*Es*NUM_BITS_PER_SYMBOL*10**(idx/10)) / np.sqrt(2)))
        
        BER_THEO_des[i] = (mu/(2*np.log2(M)))*special.erfc(np.sqrt(((env.N_total-env.N_pilots)/env.N_total)*Es*env.NUM_BITS_PER_SYMBOL*10**(idx/10))/np.sqrt(2))
        #(mu/(2*N*NUM_BITS_PER_SYMBOL))*np.sum(special.erfc(np.sqrt(np.abs(model_uncoded_awgn.H) **2
        #                                                        *((N-Np)/N)*Es*NUM_BITS_PER_SYMBOL*10**(idx/10)) / np.sqrt(2)))
        BER_THEO[i] = (mu/(2*np.log2(M)))*special.erfc(np.sqrt(Es*env.NUM_BITS_PER_SYMBOL*10**(idx/10))/np.sqrt(2))
        i = i+1

    # Plota resultado
    plt.semilogy(EbN0_dB, BER_THEO, '-', label='Teórica Sem Pilotos')
    plt.semilogy(EbN0_dB, BER_THEO_des, '--', label='Teórica Com Pilotos')
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

