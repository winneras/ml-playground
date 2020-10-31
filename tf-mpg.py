import os
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import plot_history, norm, show_predictions

print(tf.__version__)


checkpoint_path = "mpg_training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


dataset_path = keras.utils.get_file(
    "auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset = dataset.dropna()
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print(f"last 10 rows of one hot dataset: \n {dataset.tail()}")
#print(f"dataset.isna().sum(): \n {dataset.isna()}")

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

print(f"train_stats: \n {train_stats}")





normed_train_data = norm(train_dataset, train_stats)
normed_test_data = norm(test_dataset, train_stats)

print(f"last 10 rows of normed train data: \n {normed_train_data.tail()}")


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[
                     len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse', 'accuracy'])
    return model


model = build_model()
model.summary()


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1, period=5)
EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[cp_callback, early_stop, PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(f"last 10 rows of training history: \n {hist.tail()}")

# plot_history(history)
print(model.metrics_names)
loss, mae, mse, acc = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

#show_predictions(test_labels, test_predictions, 'True Values [MPG]', 'Predictions [MPG]')

model2 = build_model()
loss, mae, mse, acc = model2.evaluate(normed_test_data, test_labels, verbose=2)
print("Untrained model2, Abs Error: {:5.2f}".format(mae))

model2.load_weights(checkpoint_path)
loss, mae, mse, acc = model2.evaluate(normed_test_data, test_labels, verbose=2)
print("after loading model, Abs Error: {:5.2f}".format(mae))


#example_batch = normed_train_data[:10]
#example_result = model.predict(example_batch)
#print(f"example result: \n {example_result}")
#sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
# plt.show()
