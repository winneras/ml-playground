import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(units=2, activation='relu',
                       input_shape=[1], use_bias=True),
    keras.layers.Dense(2, activation='relu', use_bias=True),
    keras.layers.Dense(1)
])

model.compile(optimizer='sgd', loss='mean_squared_error')


m = 2
n = 3
b = 0.5
xs = np.linspace(0, 4, 200)

ys = m * xs + n * xs + b

model.fit(xs, ys, epochs=500)


model.summary()

print(model.predict([10.0]))
