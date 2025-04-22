# PAPR_Reduction_Code

The code evaluates the performance of different techniques for reducing PAPR (Peak-to-Average Power Ratio) in OFDM systems, including the PTS (Partial Transmit Sequence), MCSA (Memoryless Continuous Search Algorithm), DFT Spread and a neural network (NN)-based approach. For each method, the PAPR ratio is calculated, which is a fundamental metric for evaluating the transmission efficiency of OFDM signals, in addition to the bit error rate (BER), which indicates the reliability of the system in the presence of noise.

In the case of the neural network, the code also monitors the accuracy and loss function during training, providing insight into the model's performance in the task of learning to generate signals with lower PAPR and good reception quality. These metrics help in comparing the methods in terms of spectral efficiency, computational complexity and robustness against interference.
