# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:59:07 2023

@author: Bianca
"""

import gym
import numpy as np
from gym import spaces
import sionna as sn
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_agents.trajectories import time_step as ts

# Inicialization:
    
NUM_BITS_PER_SYMBOL = 4 # 16-QAM
N = 32  # Número de subportadoras OFDM
batch_size = 1000  # gera 1000 vezes N.
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

#%% Reinforcement Learning Method:
    
def pilot_value_change(batch_size, Np, Pilot_1, Pilot_2):
    matrix = np.zeros((batch_size, Np), dtype=int)
    for i in range(batch_size):
        matrix[i] = np.array([Pilot_1, Pilot_2])
    return matrix

class PAPR_Reduction_Env(gym.Env):
    
    def __init__(self):
        
        super().__init__()
        # Define the action space based on the q_table's number of actions
        self.action_space = spaces.Discrete(21)  # Replace NUM_ACTIONS with the actual number of actions
        #self._episode_ended = False  # Initialize the attribute
        
    def _Bits(self, batch_size, BLOCK_LENGTH):
        binary_source = sn.utils.BinarySource()
        return binary_source([batch_size, BLOCK_LENGTH])

    def _Modulation(self, bits):
        constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
        mapper = sn.mapping.Mapper(constellation = constellation)
        return mapper(bits)
    
    def _symbol(self, x, Pilot_1, Pilot_2):
        allCarriers = create_matrix(batch_size, N)
        pilotCarriers = create_pilot(batch_size, Np)
        dataCarriers = np.delete(allCarriers, pilotCarriers,axis=1)       
        # Allocate bits to dataCarriers and pilotCarriers:            
        symbol = np.zeros((batch_size, N))  # the overall N subcarriers
        symbol[:, pilotCarriers] = pilot_value_change(batch_size, Np, Pilot_1, Pilot_2)  # assign values to pilots
        symbol[np.arange(batch_size)[:, None], dataCarriers] = x  # assign values to datacarriers
        return symbol
    
    def _IFFT(self, symbol):
        OFDM_time = np.sqrt(N) * np.fft.ifft(symbol)
        return OFDM_time

    
    def Step_Action(self, PAPR_dB_ref, PAPR_dB_red, state):
        reward = 0    
        PAPR_dB_red_nov = np.zeros(len(PAPR_dB_red))
        
        for idx in range(0, batch_size):
            if PAPR_dB_ref < PAPR_dB_red[idx]:
                reward = -1
                Pilot_1 = np.random.uniform(-1, 1)
                Pilot_2 = np.random.uniform(-1, 1)
                info = env._IFFT(env._symbol(env._Modulation(env._Bits(batch_size, BLOCK_LENGTH)), Pilot_1, Pilot_2))
                PAPR_dB_red = env._PAPR(info)
                print('reward:', reward)
                print('Pmax_red:', PAPR_dB_red[idx])
                print('Pmax_ref:', PAPR_dB_ref)
                print('PAPR_dB_red_nov:', PAPR_dB_red_nov)
                
            else:
                reward = 1              
                PAPR_dB_red_nov[idx] = PAPR_dB_red[idx] 
                print('PAPR_dB_red_nov:', PAPR_dB_red_nov)
                print('reward:', reward)
                print('Pmax_ref:', PAPR_dB_ref)
                
            next_state = state + 1
            print('next_state:', next_state)
            print('idx:', idx)
        return next_state, reward, PAPR_dB_red_nov

    
    def _PAPR(self, info):
        idy = np.arange(0, batch_size)
        PAPR_red = np.zeros(len(idy))
    
        for i in idy:
            var_red = np.var(info[i])
            peakValue_red = np.max(abs(info[i])**2)
            PAPR_red[i] = peakValue_red / var_red
    
        PAPR_dB_red = 10 * np.log10(PAPR_red)
        return PAPR_dB_red


    def _CCDF(self, PAPR_dB_red):
        PAPR_Total_red = len(PAPR_dB_red)
        ma_red = max(PAPR_dB_red)
        mi_red = min(PAPR_dB_red)
        eixo_x_red = np.arange(mi_red, ma_red, 0.1)
        y_red = []
        for jj in eixo_x_red:
            A_red = len(np.where(PAPR_dB_red > jj)[0])/PAPR_Total_red
            y_red.append(A_red) #Adicionar A na lista y.       
        CCDF_red = y_red
        return CCDF_red
    
env = PAPR_Reduction_Env()

#%% DDPG:
    
class DDPGAgent:
    def __init__(self, env, learning_rate=0.8, discount_factor=0.95, exploration_prob=1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.num_states = 1000
        self.num_actions = 2  # Number of actions in the action space
        self.q_table = np.zeros((self.num_states, self.num_actions))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        