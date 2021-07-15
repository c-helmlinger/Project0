#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ftp://ftp.sas.com/pub/neural/dojo/dojo.html
# stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw


# In[135]:


from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[136]:


# load the dataset
dataset = pd.read_csv("C:/Users/Zz240/Desktop/Big Data & ML/data_cleaned.csv")
# split into input (X) and output (y) variables
train_dataset = dataset.sample(frac=0.85,random_state=2021)
test_dataset = dataset.drop(train_dataset.index)


# In[137]:


train_stats = train_dataset.describe()
train_stats.pop("Price")
train_stats = train_stats.transpose()


# In[138]:


def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# In[139]:


del normed_train_data['Price']
normed_train_data


# In[140]:


del normed_test_data['Price']
normed_test_data


# In[141]:


train_labels = train_dataset.pop('Price')
test_labels = test_dataset.pop('Price')


# In[142]:


train_X = normed_train_data
train_y = train_labels
test_X = normed_test_data
test_y = test_labels


# model = Sequential()
# model.add(Dense(16, input_dim=10, activation='relu'))
# model.add(Dense(1))
# model.compile(loss='mse',
# optimizer=keras.optimizers.RMSprop(0.005), metrics=['mae', 'mse'])
# history = model.fit(train_X, train_y, epochs=1000, batch_size=16, validation_split = 0.2)
# _,_, MSE = model.evaluate(train_X, train_y)
# print('RMSE: %.2f' % (np.sqrt(MSE)))

# In[143]:


from sklearn.model_selection import train_test_split
# define the keras model
model = Sequential()
model.add(Dense(8, input_dim=10, activation='relu'))
# model.add(Dense(4, activation='relu'))
model.add(Dense(1))
# compile the keras model
model.compile(loss='mse',
                optimizer=keras.optimizers.RMSprop(0.005),
                metrics=['mae', 'mse'])
# fit the keras model on the dataset
history = model.fit(train_X, train_y, epochs=1000, batch_size=16, validation_split=0.2)

# evaluate the keras model
_,_, MSE = model.evaluate(train_X, train_y)
print('RMSE: %.2f' % (np.sqrt(MSE)))


# In[144]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[145]:


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


# In[146]:


test_predictions = model.predict(test_X)


# In[147]:


np.size(test_predictions)


# In[148]:


np.array(test_labels)[1]


# In[149]:


error = []
error_squared = []
for i in range(np.size(test_predictions)):
    error.append(test_predictions[i][0] - np.array(test_labels)[i])
    error_squared.append(np.square(test_predictions[i][0] - np.array(test_labels)[i]))
error_squared


# In[150]:


RMSE = np.sqrt(np.sum(error_squared)/np.size(test_predictions))
RMSE


# In[153]:


a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Price]')
plt.ylabel('Predictions [Price]')
lims = [0, 5e+06]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[154]:


plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [Price]")
_ = plt.ylabel("Count")


# In[ ]:





# In[ ]:




