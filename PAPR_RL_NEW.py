import numpy as np
import gym
from gym import spaces
import sionna as sn
import matplotlib.pyplot as plt

NUM_BITS_PER_SYMBOL = 4 # 16-QAM
N = 32  # Número de subportadoras OFDM
batch_size = 100  # gera 1000 vezes N.
Np = 2
BLOCK_LENGTH = (N-Np)*NUM_BITS_PER_SYMBOL

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
    
def pilot_value_change(batch_size, Np, Pilot_1, Pilot_2):
    matrix = np.zeros((batch_size, Np), dtype=int)
    for i in range(batch_size):
        matrix[i] = np.array([Pilot_1, Pilot_2])
    return matrix

def Bits(batch_size, BLOCK_LENGTH):
    binary_source = sn.utils.BinarySource()
    return binary_source([batch_size, BLOCK_LENGTH])

def Modulation(bits):
    constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
    mapper = sn.mapping.Mapper(constellation = constellation)
    return mapper(bits)

class PAPRReductionEnv(gym.Env):
    def __init__(self, batch_size, BLOCK_LENGTH, N, Np, NUM_BITS_PER_SYMBOL):
        super(PAPRReductionEnv, self).__init__()
        
        self.batch_size = batch_size
        self.BLOCK_LENGTH = BLOCK_LENGTH
        self.N = N
        self.Np = Np
        self.NUM_BITS_PER_SYMBOL = NUM_BITS_PER_SYMBOL

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.N,), dtype=np.float32)

        # Initialize variables
        self.Pilot_1 = -1
        self.Pilot_2 = 1
        self.Pmax_ref = 0
        print("Initialized environment with N =", self.N, "Np =", self.Np)

    def _reset(self):
        self.Pmax_ref = 0
        state_index_1 = np.random.randint(NUM_ACTIONS)  # Random initialization of state_index_1
        state_index_2 = np.random.randint(NUM_ACTIONS)  # Random initialization of state_index_2
        initial_state = (state_index_1, state_index_2)
        print("Resetting environment with state:", initial_state)
        return initial_state


    def _symbol(self, Pilot_1, Pilot_2):
        allCarriers = create_matrix(batch_size, N)
        pilotCarriers = create_pilot(batch_size, Np)
        dataCarriers = np.delete(allCarriers, pilotCarriers, axis=1)
        # Allocate bits to dataCarriers and pilotCarriers:
        symbol = np.zeros((batch_size, N))  # the overall N subcarriers
        symbol[:, pilotCarriers] = pilot_value_change(batch_size, Np, Pilot_1, Pilot_2)  # assign values to pilots
        bits = Bits(batch_size, BLOCK_LENGTH)
        symbol[np.arange(batch_size)[:, None], dataCarriers] = Modulation(bits)  # assign values to datacarriers
        OFDM_time = np.sqrt(N) * np.fft.ifft(symbol)
        Pmax_red = np.max(abs(OFDM_time) ** 2)
        return symbol, Pmax_red

    def _IFFT(self, symbol):
        OFDM_time = np.sqrt(N)*np.fft.ifft(symbol)
        Pmax_ref = np.max(abs(OFDM_time)**2)
        return Pmax_ref, OFDM_time
    
    def step(self, action):
        Pilot_1, Pilot_2 = action
    
        # Calculate Pmax_red using the new pilot values
        symbol, Pmax_red = self._symbol(Pilot_1, Pilot_2)
    
        if Pmax_red > self.Pmax_ref:
            reward = -1
            Pilot_1, Pilot_2 = np.random.uniform(-1, 1, size=(2,))
            while Pilot_1 == 0 or Pilot_2 == 0:
                Pilot_1, Pilot_2 = np.random.uniform(-1, 1, size=(2,))
        else:
            self.Pmax_ref = Pmax_red
            reward = 1
            self.Pilot_1, self.Pilot_2 = Pilot_1, Pilot_2
            
        # Convert the pilot values to discrete state indices
        state_index_1 = int((Pilot_1 + 1) * NUM_ACTIONS / 2)
        state_index_2 = int((Pilot_2 + 1) * NUM_ACTIONS / 2)

        return (state_index_1, state_index_2), reward, False, {}
    
    def _PAPR(self, OFDM_time):
        idx = np.arange(0, 100)
        PAPR = np.zeros(len(idx))

        for i in idx:
            var = np.var(OFDM_time[i])
            peakValue = np.max(abs(OFDM_time[i])**2)
            PAPR[i] = peakValue/var 
            PAPR_dB = 10*np.log10(PAPR)
        return PAPR_dB
    
    def CCDF(self, PAPR_dB):
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


# Create the RL environment
env = PAPRReductionEnv(batch_size, BLOCK_LENGTH, N, Np, NUM_BITS_PER_SYMBOL)

# Initialize the Q-table with zeros
NUM_ACTIONS = 21  # Number of discrete actions for each Pilot (Pilot_1 and Pilot_2)
NUM_STATES = NUM_ACTIONS * NUM_ACTIONS
q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

# Hyperparameters
alpha = 0.1   # Learning rate
gamma = 0.6   # Discount factor
epsilon = 0.1  # Exploration rate

# Training loop
for i in range(1, 5):  # Run 100000 episodes or any desired number
    state_index_1, state_index_2 = env.reset()  # Random initialization of the environment
    done = False

    while not done:
        # Epsilon-greedy exploration
        if np.random.rand() < epsilon:
            action = np.random.uniform(-1, 1, size=(2,))
        else:
            state_index_discrete = state_index_1 * NUM_ACTIONS + state_index_2
            action_index = np.argmax(q_table[state_index_discrete])

            # Map action index to continuous action value
            action = action_index * 2 / (NUM_ACTIONS - 1) - 1

        next_state_index_1, next_state_index_2, reward, done, _ = env.step(action)

        # Map the next state's discrete indices to a single state index
        next_state_index_discrete = next_state_index_1 * NUM_ACTIONS + next_state_index_2

        # Map the continuous action values to the discrete action indices
        action_index = int((action + 1) * NUM_ACTIONS / 2)

        # Update Q-value using the Q-learning formula
        old_value = q_table[state_index_discrete, action_index]
        next_max = np.max(q_table[next_state_index_discrete])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state_index_discrete, action_index] = new_value

        state_index_1, state_index_2 = next_state_index_1, next_state_index_2



# Calculate PAPR and plot PAPR_dB vs. CCDF
papr_dB_values = []
for episode in range(5):
    state = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        next_state, _, done, _ = env.step(action)

        Pmax_red, OFDM_time = env._IFFT(env._symbol)  # Assuming you have the IFFT method in the environment
        PAPR_dB = env._PAPR(OFDM_time)  # Assuming you have the PAPR method in the environment
        papr_dB_values.extend(PAPR_dB)

    old_value = q_table[state, action]  # Valor da ação escolhida no estado atual



# Plot PAPR_dB vs. CCDF
ccdf = env.CCDF(np.array(papr_dB_values))  # Assuming you have the CCDF method
plt.semilogy(np.sort(papr_dB_values), ccdf)
plt.xlabel('PAPR (dB)')
plt.ylabel('CCDF')
plt.grid()
plt.title('CCDF of PAPR')
plt.show()

env.close()