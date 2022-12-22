#!/usr/bin/env python
# coding: utf-8

# In[1]:


#James Gibson
#12/03/22
#Script is based on https://www.tensorflow.org/tutorials/keras/regression

#Error warning about tensorflow

#2022-12-03 17:53:16.980823: I tensorflow/core/platform/cpu_feature_guard.cc:193] 
#This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use 
#the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2
#To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


# In[2]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import seaborn as sns
#import rasterio
#import geopandas as gpd
#import rioxarray


# In[3]:


#Loading my data set (except for 2021 data)
data = r'/Users/bustergibson/documents/ml/pm25_aod_complete.csv'
dataset = pd.read_csv(data)
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
dataset = dataset[dataset['YEAR'] < 2021]
dataset = dataset[['SITE_LATITUDE','SITE_LONGITUDE','YEAR','PM25','AOD']]
dataset


# In[4]:


#Create training and test data sets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# In[5]:


#Linear Regression
train_features = train_dataset.copy()
test_features = test_dataset.copy()

#We are predicting PM 2.5
train_labels = train_features.pop('PM25')
test_labels = test_features.pop('PM25')


# In[6]:


#We need to normalize the data
train_dataset.describe().transpose()[['mean', 'std']]


# In[7]:


#Normalization
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))


# In[8]:


#First we are going to obtain a linear regression model simply testing the ability of AOD to predict to PM 2.5
#Ex. 
# linear_model = tf.keras.Sequential([
#     normalizer,
#     layers.Dense(units=1)
# ])

aod = np.array(train_features['AOD'])
aod_normalizer = layers.Normalization(input_shape=[1,], axis=None)
aod_normalizer.adapt(aod)


# In[9]:


aod_model = tf.keras.Sequential([
    aod_normalizer,
    layers.Dense(units=1)
])

aod_model.summary()


# In[10]:


aod_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


# In[11]:


get_ipython().run_cell_magic('time', '', "history = aod_model.fit(\n    train_features['AOD'],\n    train_labels,\n    epochs=100,\n    # Suppress logging.\n    verbose=0,\n    # Calculate validation results on 20% of the training data.\n    validation_split = 0.2)")


# In[12]:


plt.clf()
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [AOD]')
  plt.legend()
  plt.grid(True)

plot_loss(history)
plt.show()


# In[13]:


#Collect the results on the test set for later
test_results = {}

test_results['aod_model'] = aod_model.evaluate(
    test_features['AOD'],
    test_labels, verbose=0)


# In[15]:


plt.clf()
test_predictions = aod_model.predict(test_features['AOD']).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [PM25]')
plt.ylabel('Predictions [PM25]')
lims = [0, 20]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


plt.show()


# In[16]:


mae = tf.keras.losses.MeanAbsoluteError()
mae(test_labels, test_predictions).numpy()


# In[17]:


mse = tf.keras.losses.MeanSquaredError()
mse(test_labels, test_predictions).numpy()


# In[ ]:




