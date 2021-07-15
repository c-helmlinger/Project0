#!/usr/bin/env python
# coding: utf-8
file_location = # Eg: 'C:/Users/...' Don't forget to include the quotation mark
test_size = # float between 0 and 1
neuron = # int
dimension_x = # number of variables for X
learning_rate = # float
epoch_num = # number of epochs
size_batch = # size of the batch 
validation_size = # float between 0 and 1

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load the dataset
dataset = pd.read_csv(file_location) # use your own datasets ()
# split into input (X) and output (y) variables
train_dataset = dataset.sample(frac=1 - test_size,random_state=2021)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("Price")
train_stats = train_stats.transpose()

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

del normed_train_data['Price']
normed_train_data

del normed_test_data['Price']
normed_test_data

train_labels = train_dataset.pop('Price')
test_labels = test_dataset.pop('Price')

train_X = normed_train_data
train_y = train_labels
test_X = normed_test_data
test_y = test_labels


from sklearn.model_selection import train_test_split
# define the keras model
model = Sequential()
model.add(Dense(neuron, input_dim=dimension_x, activation='relu'))
# model.add(Dense(4, activation='relu'))
model.add(Dense(1))
# compile the keras model
model.compile(loss='mse',
                optimizer=keras.optimizers.RMSprop(learning_rate),
                metrics=['mae', 'mse'])
# fit the keras model on the dataset
history = model.fit(train_X, train_y, epochs=epoch_num, batch_size=size_batch, validation_split=(1 - test_size)/validation_size)

# evaluate the keras model
_,_, MSE = model.evaluate(train_X, train_y)
print('RMSE: %.2f' % (np.sqrt(MSE)))

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# summarize history for mse
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('Model Fit')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

test_predictions = model.predict(test_X)

np.size(test_predictions)
np.array(test_labels)[1]

error = []
error_squared = []
for i in range(np.size(test_predictions)):
    error.append(test_predictions[i][0] - np.array(test_labels)[i])
    error_squared.append(np.square(test_predictions[i][0] - np.array(test_labels)[i]))
error_squared

RMSE = np.sqrt(np.sum(error_squared)/np.size(test_predictions))
RMSE

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Price]')
plt.ylabel('Predictions [Price]')
lims = [0, 5e+06]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [Price]")
_ = plt.ylabel("Count")

return RMSE


