import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class Model:
    def __init__(self):
        self.weight = tf.Variable(10.0)
        self.bias = tf.Variable(10.0)

    def __call__(self, x):
        return self.weight * x + self.bias


def calculate_loss(y_actual, y_output):
    return tf.reduce_mean(tf.square(y_actual - y_output))


def train(model, x, y, learning_rate):
    with tf.GradientTape() as gt:
        y_output = model(x)
        loss = calculate_loss(y, y_output)

    new_weight, new_bias = gt.gradient(loss, [model.weight, model.bias])
    model.weight.assign_sub(new_weight * learning_rate)
    model.bias.assign_sub(new_bias * learning_rate)


m = 2
b = 0.5
x = np.linspace(0, 4, 200)

y = m * x + b + np.random.randn(*x.shape) + 0.25

plt.scatter(x, y)

model = Model()
epochs = 500
learning_rate = 0.05
for epoch in range(epochs):
    y_output = model(x)
    loss = calculate_loss(y, y_output)
    print(f"Epoch: {epoch}, loss: {loss.numpy()}")
    train(model, x, y, learning_rate)


print(f"weight: {model.weight.numpy()}, bias: {model.weight.numpy()}")

new_x = np.linspace(0, 4, 50)
new_y = model.weight.numpy() * new_x + model.weight.numpy()
plt.scatter(new_x, new_y)

plt.show()