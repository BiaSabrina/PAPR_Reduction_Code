# PAPR Reduction Technique for Mobile Communication Systems Using Neural Networks

**Authors:** Bianca S. de C. da Silva, Pedro H. C. de Souza, and Luciano L. Mendes  
**Manuscript ID:** 9381  
**Submitted to:** *IEEE Latin America Transactions*

## 📄 Overview

This repository contains the script **`Generate_PAPR_BER_Complexity_Acc_Loss`**, which provides a complete framework to reproduce the simulation results presented in the manuscript.

The study investigates several techniques for reducing the **Peak-to-Average Power Ratio (PAPR)** in **OFDM systems**, including:

- 🧠 Neural Network (NN)-based approach  
- 📶 Partial Transmit Sequence (**PTS**)  
- 🔍 Memoryless Continuous Search Algorithm (**MCSA**)  
- 🔁 **DFT Spread**

### 📊 Evaluation Metrics

All techniques are evaluated based on:

- **PAPR** (Peak-to-Average Power Ratio)  
- **BER** (Bit Error Rate)
- **Accuracy and Loss** for the Neural Network. 
- **Computational Complexity**

✅ Everything is implemented in **one unified script**:  
`Generate_PAPR_BER_Complexity_Acc_Loss`

---

## 📁 Files

- `Generate_PAPR_BER_Complexity_Acc_Loss.py` – Main script that runs all simulations and generates plots for PAPR, BER, accuracy, loss and complexity analysis.

---

## ▶️ How to Run

1. Open Pyhton.
2. Navigate to the folder containing the script.
3. Run the following command:

```Python
Generate_PAPR_BER_Complexity_Acc_Loss
