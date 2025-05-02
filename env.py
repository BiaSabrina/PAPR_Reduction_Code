from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

import tensorflow as tf
import numpy as np
from scipy import special
import sionna as sn
import scipy as sp
import matplotlib.pyplot as plt
import gym

class OfdmPaprEnv(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,), dtype=np.int32, minimum=0, name='observation')
    self._state = 0
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = 0
    self._episode_ended = False
    return ts.restart(np.array([self._state], dtype=np.int32))

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    # Make sure episodes don't go on forever.
    if action == 1:
      self._episode_ended = True
    elif action == 0:
      new_card = np.random.randint(1, 11)
      self._state += new_card
    else:
      raise ValueError('`action` should be 0 or 1.')

    if self._episode_ended or self._state >= 21:
      reward = self._state - 21 if self._state <= 21 else -21
      return ts.termination(np.array([self._state], dtype=np.int32), reward)
    else:
      return ts.transition(
          np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)
  
environment = OfdmPaprEnv()

utils.validate_py_environment(environment, episodes=5)

#%% Start the implemantation:

NUM_BITS_PER_SYMBOL = 4 # QAM
K = 4
N = 256 # número de subportadoras OFDM

BLOCK_LENGTH = N*NUM_BITS_PER_SYMBOL

batch_size = 1000
cp_length = 16  # comprimento do prefixo cíclico

#%% Modulation OFDM with Sionna

constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
constellation.show()
binary_source = sn.utils.BinarySource()
mapper = sn.mapping.Mapper(constellation = constellation)

bits = binary_source([batch_size, BLOCK_LENGTH])

#%% Mapper
    
x = mapper(bits)
print('X:',x)

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

#%% Using the invironment:
    
IFFT_new = []

for j in PAPR_dB:
    ma = max(PAPR_dB[j])

    if PAPR_dB > ma:
       pass
    else:
       IFFT_new = np.append(PAPR_dB[j])
       
#%% Calculate CCDF:

y = []

for j in eixo_x:
    A = len(np.where(PAPR_dB > j)[0])/N
    y.append(A) #Adicionar A na lista y.
    
CCDF = y

#%%  Plot PAPR x CCDF

fig = plt.figure(figsize=(8, 7))
plt.semilogy(eixo_x, CCDF,"-^r",linewidth=2.5,label="Conventional")
plt.legend(loc="lower left")
plt.xlabel('PAPR (dB)')
plt.ylabel('CCDF')
plt.title('CCDF x PAPR')
plt.grid()
plt.ylim([1e-2, 1])
