import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Input , Lambda
import matplotlib.pyplot as plt
import time
import sionna as sn


NUM_BITS_PER_SYMBOL = 4   

N = 64
Np = 2
learning_Rate = 0.0001
initializer = tf.keras.initializers.glorot_normal(seed=25)
batchsize = 400
lamba = 0.002
do = 0


def Modulation(random_Data):
    constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
    mapper = sn.mapping.Mapper(constellation = constellation)
    return mapper(random_Data)

def IFFT(tensor):
    real = tf.math.real(tensor)
    imag = tf.math.imag(tensor)
    tensor_complex = tf.complex(real, imag)
    inverse_fft = tf.signal.ifft(tensor_complex)
    return tf.math.real(inverse_fft)  # Extract the real part after IFFT

def PAPR_of_Data(data, o):
    PAPR = np.zeros(o,dtype=np.complex64)
    for i in range(o):
        x = np.square(np.abs(data[i,:]))
        PAPR[i] = np.max(x) / np.mean(x)
    PAPR_db = 10*np.log10(PAPR)
    return PAPR_db 

def PAPR_of_Data_IFFT(data, o):
    PAPR = np.zeros(o, dtype=np.complex64)
    for i in range(o):
        x = np.square(np.abs(np.fft.ifft(data[i,:])))
        PAPR[i] = np.max(x) / np.mean(x)
    PAPR_db = 10*np.log10(PAPR)
    return PAPR_db 


inputs = Input(shape = (N*2,))
#ENCODER
x1 = Dense(1024, kernel_initializer=initializer, bias_initializer='random_normal') (inputs)
x2 = BatchNormalization() (x1)
x2 = Dropout(do) (x2)
x3 = Activation('relu') (x2)

x4 = Dense(1024, kernel_initializer=initializer, bias_initializer='random_normal') (x3)
x5 = BatchNormalization() (x4)
x5 = Dropout(do) (x5)
x6 = Activation('relu')(x5)

x7 = Dense(1024, kernel_initializer=initializer, bias_initializer='random_normal') (x6)
x8 = BatchNormalization() (x7)
x8 = Dropout(do) (x8)
x9 = Activation('relu')(x8)

x10 = Dense(1024, kernel_initializer=initializer, bias_initializer='random_normal') (x9)
x11 = BatchNormalization() (x10)
x11 = Dropout(do) (x11)
x12 = Activation('relu')(x11)

x13 = Dense(N*2, kernel_initializer=initializer, bias_initializer='random_normal', activation= 'tanh') (x12)

encoder = Lambda(IFFT, name='encoder') (x13)

model = keras.Model(inputs=inputs, outputs=encoder)


dataset = np.random.randint(2, size=(5000, N*2))
valset = np.random.randint(2, size=(500, N*2))

# Split validation set into inputs and outputs
valset_inputs = valset
valset_outputs = None

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
opt = keras.optimizers.Adam(learning_Rate)
model.compile(optimizer=opt, loss=['mse'], metrics=['accuracy'])
print(model.summary())
history = model.fit(dataset, dataset, epochs=400, batch_size=400, shuffle=True, callbacks=[callback],
                    validation_data=(valset_inputs, valset_outputs))

size = 10000
dataset = np.random.randint(2, size=(size, N*2))
st = time.time()
NET = model.predict(dataset, batch_size = batchsize)  # adapt size
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
Modulated_Data = Modulation(dataset)

PRNET = PAPR_of_Data(NET, size)
OFDM = PAPR_of_Data_IFFT(Modulated_Data, size)


plt.figure(1)
plt.xlabel('PAPR in db')
plt.ylabel('CCDF')
plt.yscale('log')
plt.ylim(0.001, 1.2)
plt.grid(True, which="both")
m, n = np.histogram(PRNET)
plt.plot(n[1:], 1 - np.cumsum(m)/size, 'r', label = 'Reduced Signal')
m1, n1 = np.histogram(OFDM)
plt.plot(n1[1:], 1 - np.cumsum(m1)/size, 'b', label = 'Original Signal')

plt.show()