# ----------------------------------------------------------------------
# Libraries ------------------------------------------------------------
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow as tf
from keras.optimizers import Adam
import time
from keras import regularizers
from keras_tuner.tuners import BayesianOptimization

# Confirming GPU is being used
tf.test.is_gpu_available()
tf.config.list_physical_devices('GPU')

# ------------------------------------------------------------------------
# AAPL -------------------------------------------------------------------
# ------------------------------------------------------------------------

# Reading in data
AAPL = pd.read_csv("Data/AAPL.csv",index_col= "Date", parse_dates = True)
AAPL['Close/Last'] = AAPL['Close/Last'].str.replace('$', '')

# Splitting dataset for cross-validation
train_test_split = 0.9
train_size = int(len(AAPL) * train_test_split) # Use 90% of data for training
train = AAPL.iloc[0:train_size]['Close/Last'] # Selecting closing price as target
test = AAPL.iloc[train_size:len(AAPL)] ['Close/Last']
test = pd.to_numeric(test)

# Reshaping data sets from Panda Series to 1D Array
train = train.values.flatten()
train = train.reshape(-1,1)
test = test.values.flatten()
test = test.reshape(-1,1)

# Scaling time-series train and test values values
stage_transformer = RobustScaler()
stage_transformer = stage_transformer.fit(train)
scaled_train = stage_transformer.transform(train)
scaled_test = stage_transformer.transform(test)

# Define inputs  
n_input = 31 # use last months information to predict next day
n_features = 1 # only 1 feature (for now)

# Define train and test
training = TimeseriesGenerator(scaled_train, scaled_train, 
                                length = n_input,
                                batch_size = 5000)

validation = TimeseriesGenerator(scaled_test, scaled_test, 
                                length = n_input,
                                batch_size = 1000)

# Load best model
best_model_AAPL = tf.keras.models.load_model('Models/Bayes_HT_AAPL.keras')    

# Training Best AAPL Model

for j in range(12):
    
   # Fitting model  
   with tf.device('/device:GPU:0'): 
        best_model_AAPL.fit(training, epochs = 100, validation_data = validation)

   duration = len(test)
   test_predictions = []
   first_eval_batch = scaled_train[-n_input:]
   current_batch = first_eval_batch.reshape((1, n_input, n_features))
   for i in range(duration):
      current_pred = best_model_AAPL.predict(current_batch)[0]
      test_predictions.append(current_pred) 
      current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
   true_predictions = stage_transformer.inverse_transform(test_predictions)
   
   fig = plt.figure(figsize=(10,5))
   ax = fig.add_subplot(111)
   plt.plot(test, color = 'b', label = "AAPL")
   plt.plot(true_predictions, color = 'r', label = "LSTM Predictions")
   plt.legend()
   ax.set_ylabel("Response")
   ax.set_xlabel("Day's Past Training Data")
   ax.set_title("LSTM Predictions on Observed AAPL Stock")
   plt.savefig(f"Training/AAPL_model_{j}.png")
   plt.clf()
   
# Check when loss levels out
loss_per_epoch = best_model_AAPL.history.history["loss"]
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
plt.show()

# ---------------------------------------------------------------------
# AMZN ----------------------------------------------------------------
# ---------------------------------------------------------------------

# Reading in data
AMZN = pd.read_csv("Data/AMZN.csv",index_col= "Date", parse_dates = True)
AMZN['Close/Last'] = AMZN['Close/Last'].str.replace('$', '')

# Splitting dataset for cross-validation
train_test_split = 0.9
train_size = int(len(AMZN) * train_test_split) # Use 90% of data for training
train = AMZN.iloc[0:train_size]['Close/Last'] # Selecting closing price as target
test = AMZN.iloc[train_size:len(AMZN)] ['Close/Last']
test = pd.to_numeric(test)

# Reshaping data sets from Panda Series to 1D Array
train = train.values.flatten()
train = train.reshape(-1,1)
test = test.values.flatten()
test = test.reshape(-1,1)

# Scaling time-series train and test values values
stage_transformer = RobustScaler()
stage_transformer = stage_transformer.fit(train)
scaled_train = stage_transformer.transform(train)
scaled_test = stage_transformer.transform(test)

# Define inputs  
n_input = 31 # use last months information to predict next day
n_features = 1 # only 1 feature (for now)

# Define train and test
training = TimeseriesGenerator(scaled_train, scaled_train, 
                                length = n_input,
                                batch_size = 5000)

validation = TimeseriesGenerator(scaled_test, scaled_test, 
                                length = n_input,
                                batch_size = 1000)

# Load best model
best_model_AMZN = tf.keras.models.load_model('Models/Bayes_HT_AMZN.keras')    

# Training Best AMZN Model

for j in range(12):
    
   # Fitting model  
   with tf.device('/device:GPU:0'): 
        best_model_AMZN.fit(training, epochs = 1000, validation_data = validation)

   duration = len(test)
   test_predictions = []
   first_eval_batch = scaled_train[-n_input:]
   current_batch = first_eval_batch.reshape((1, n_input, n_features))
   for i in range(duration):
      current_pred = best_model_AMZN.predict(current_batch)[0]
      test_predictions.append(current_pred) 
      current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
   true_predictions = stage_transformer.inverse_transform(test_predictions)
   
   fig = plt.figure(figsize=(10,5))
   ax = fig.add_subplot(111)
   plt.plot(test, color = 'b', label = "AMZN")
   plt.plot(true_predictions, color = 'r', label = "LSTM Predictions")
   plt.legend()
   ax.set_ylabel("Response")
   ax.set_xlabel("Day's Past Training Data")
   ax.set_title("LSTM Predictions on Observed AMZN Stock")
   plt.savefig(f"Training/AMZN_model_{j}.png")
   plt.clf()
   
# Check when loss levels out
loss_per_epoch = best_model_AMZN.history.history["loss"]
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
plt.show()

# ---------------------------------------------------------------------
# CAT -----------------------------------------------------------------
# ---------------------------------------------------------------------

# Reading in data
CAT = pd.read_csv("Data/CAT.csv",index_col= "Date", parse_dates = True)
CAT['Close/Last'] = CAT['Close/Last'].str.replace('$', '')

# Splitting dataset for cross-validation
train_test_split = 0.9
train_size = int(len(CAT) * train_test_split) # Use 90% of data for training
train = CAT.iloc[0:train_size]['Close/Last'] # Selecting closing price as target
test = CAT.iloc[train_size:len(CAT)] ['Close/Last']
test = pd.to_numeric(test)

# Reshaping data sets from Panda Series to 1D Array
train = train.values.flatten()
train = train.reshape(-1,1)
test = test.values.flatten()
test = test.reshape(-1,1)

# Scaling time-series train and test values values
stage_transformer = RobustScaler()
stage_transformer = stage_transformer.fit(train)
scaled_train = stage_transformer.transform(train)
scaled_test = stage_transformer.transform(test)

# Define inputs  
n_input = 31 # use last months information to predict next day
n_features = 1 # only 1 feature (for now)

# Define train and test
training = TimeseriesGenerator(scaled_train, scaled_train, 
                                length = n_input,
                                batch_size = 5000)

validation = TimeseriesGenerator(scaled_test, scaled_test, 
                                length = n_input,
                                batch_size = 1000)

# Load best model
best_model_CAT = tf.keras.models.load_model('Models/Bayes_HT_CAT.keras')    

# Training Best CAT Model

for j in range(12):
    
   # Fitting model  
   with tf.device('/device:GPU:0'): 
        best_model_CAT.fit(training, epochs = 1000, validation_data = validation)

   duration = len(test)
   test_predictions = []
   first_eval_batch = scaled_train[-n_input:]
   current_batch = first_eval_batch.reshape((1, n_input, n_features))
   for i in range(duration):
      current_pred = best_model_CAT.predict(current_batch)[0]
      test_predictions.append(current_pred) 
      current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
   true_predictions = stage_transformer.inverse_transform(test_predictions)
   
   fig = plt.figure(figsize=(10,5))
   ax = fig.add_subplot(111)
   plt.plot(test, color = 'b', label = "CAT")
   plt.plot(true_predictions, color = 'r', label = "LSTM Predictions")
   plt.legend()
   ax.set_ylabel("Response")
   ax.set_xlabel("Day's Past Training Data")
   ax.set_title("LSTM Predictions on Observed CAT Stock")
   plt.savefig(f"Training/CAT_model_{j}.png")
   plt.clf()
   
# Check when loss levels out
loss_per_epoch = best_model_CAT.history.history["loss"]
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
plt.show()

# ---------------------------------------------------------------------
# NVDA ----------------------------------------------------------------
# ---------------------------------------------------------------------

# Reading in data
NVDA = pd.read_csv("Data/NVDA.csv",index_col= "Date", parse_dates = True)
NVDA['Close/Last'] = NVDA['Close/Last'].str.replace('$', '')

# Splitting dataset for cross-validation
train_test_split = 0.9
train_size = int(len(NVDA) * train_test_split) # Use 90% of data for training
train = NVDA.iloc[0:train_size]['Close/Last'] # Selecting closing price as target
test = NVDA.iloc[train_size:len(NVDA)] ['Close/Last']
test = pd.to_numeric(test)

# Reshaping data sets from Panda Series to 1D Array
train = train.values.flatten()
train = train.reshape(-1,1)
test = test.values.flatten()
test = test.reshape(-1,1)

# Scaling time-series train and test values values
stage_transformer = RobustScaler()
stage_transformer = stage_transformer.fit(train)
scaled_train = stage_transformer.transform(train)
scaled_test = stage_transformer.transform(test)

# Define inputs  
n_input = 31 # use last months information to predict next day
n_features = 1 # only 1 feature (for now)

# Define train and test
training = TimeseriesGenerator(scaled_train, scaled_train, 
                                length = n_input,
                                batch_size = 5000)

validation = TimeseriesGenerator(scaled_test, scaled_test, 
                                length = n_input,
                                batch_size = 1000)

# Load best model
best_model_NVDA = tf.keras.models.load_model('Models/Bayes_HT_NVDA.keras')    

# Training Best NVDA Model

for j in range(12):
    
   # Fitting model  
   with tf.device('/device:GPU:0'): 
        best_model_NVDA.fit(training, epochs = 1000, validation_data = validation)

   duration = len(test)
   test_predictions = []
   first_eval_batch = scaled_train[-n_input:]
   current_batch = first_eval_batch.reshape((1, n_input, n_features))
   for i in range(duration):
      current_pred = best_model_NVDA.predict(current_batch)[0]
      test_predictions.append(current_pred) 
      current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
   true_predictions = stage_transformer.inverse_transform(test_predictions)
   
   fig = plt.figure(figsize=(10,5))
   ax = fig.add_subplot(111)
   plt.plot(test, color = 'b', label = "NVDA")
   plt.plot(true_predictions, color = 'r', label = "LSTM Predictions")
   plt.legend()
   ax.set_ylabel("Response")
   ax.set_xlabel("Day's Past Training Data")
   ax.set_title("LSTM Predictions on Observed NVDA Stock")
   plt.savefig(f"Training/NVDA_model_{j}.png")
   plt.clf()
   
# Check when loss levels out
loss_per_epoch = best_model_NVDA.history.history["loss"]
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
plt.show()
