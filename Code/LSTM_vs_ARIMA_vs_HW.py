# ----------------------------------------------------------------------
# Libraries ------------------------------------------------------------
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
import tensorflow as tf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import RobustScaler
import datetime
from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima.arima import auto_arima


##########################################################################
#                                                                        #
#             Before using this script, run the                          #
#             data preprocessing script to read                          #
#             in the most current data from                              #
#                  Yahoo Finance                                         #
#                                                                        #
##########################################################################

# ------------------------------------------------------------------------
# AAPL -------------------------------------------------------------------
# ------------------------------------------------------------------------

# Reading in data
AAPL_Raw = pd.read_csv("Data/Stationary_AAPL.csv",index_col= "Date", parse_dates = True)

Last_Month = AAPL_Raw.iloc[-31:]
AAPL = AAPL_Raw.iloc[0:-32]

# Splitting dataset for cross-validation
train_test_split = 0.9
train_size = int(len(AAPL) * train_test_split) # Use 90% of data for training
train = AAPL.iloc[0:train_size] # Selecting closing price as target
test = AAPL.iloc[train_size:len(AAPL)]
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

# Whole time-series for forecasting
TS = AAPL.values.flatten()
TS = TS.reshape(-1,1)
TS_Scaled = stage_transformer.transform(TS)

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

# Fitting model without loop
with tf.device('/device:GPU:0'): 
   best_model_AAPL.fit(training, epochs = 5000, validation_data = validation)
   
# Whole time-series for forecasting
TS = AAPL.values.flatten()
TS = TS.reshape(-1,1)
TS_Scaled = stage_transformer.transform(TS)

# Forecasting
duration = 31 # two week predictions
test_predictions = []
first_eval_batch = TS_Scaled[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))
for i in range(duration):
   current_pred = best_model_AAPL.predict(current_batch)[0]
   test_predictions.append(current_pred) 
   current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
true_predictions = stage_transformer.inverse_transform(test_predictions)
  
# Current date 
start_date = Last_Month.index[0]
# Generate list of dates 
date_sequence = pd.date_range(start=start_date, periods=duration, freq='B')

# Fit ARIMA
model_arima = auto_arima(TS)
model_arima.fit(TS)
model_arima.summary()
arima_preds = model_arima.predict(n_periods = 31)
forecast = pd.DataFrame(arima_preds, index = AAPL.index[train_size:train_size + 31], columns=['Prediction'])

# Fit HW
HW = ExponentialSmoothing(TS, trend = 'add', seasonal = 'add', seasonal_periods = 365).fit()
HW_preds = HW.forecast(31)

# Comparison
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(Last_Month, color = 'b', label = "AAPL Stock")
plt.plot(date_sequence, true_predictions, color = 'r', label = "LSTM")
plt.plot(date_sequence, arima_preds, color = 'g', label = "ARIMA")
plt.plot(date_sequence, HW_preds, color = 'y', label = "Holt-Winters")
plt.legend()
ax.set_ylabel("Closing Price (Stationary)")
ax.set_xlabel("Date")
ax.set_title("Comparison of Forecasts on AAPL Stock")
plt.xticks(rotation = 45, ha = 'right', fontsize = 6)
ax.xaxis.set_major_locator(MaxNLocator(nbins=31))
plt.savefig("Figures/AAPL_LSTM_vs_ARIMA_vs_HW.png")
plt.clf()

# ------------------------------------------------------------------------
# AMZN -------------------------------------------------------------------
# ------------------------------------------------------------------------

# Reading in data
AMZN_Raw = pd.read_csv("Data/Stationary_AMZN.csv",index_col= "Date", parse_dates = True)

Last_Month = AMZN_Raw.iloc[-31:]
AMZN = AMZN_Raw.iloc[0:-32]

# Splitting dataset for cross-validation
train_test_split = 0.9
train_size = int(len(AMZN) * train_test_split) # Use 90% of data for training
train = AMZN.iloc[0:train_size] # Selecting closing price as target
test = AMZN.iloc[train_size:len(AMZN)]
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

# Whole time-series for forecasting
TS = AMZN.values.flatten()
TS = TS.reshape(-1,1)
TS_Scaled = stage_transformer.transform(TS)

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

# Fitting model without loop
with tf.device('/device:GPU:0'): 
   best_model_AMZN.fit(training, epochs = 5000, validation_data = validation)
   
# Whole time-series for forecasting
TS = AMZN.values.flatten()
TS = TS.reshape(-1,1)
TS_Scaled = stage_transformer.transform(TS)

# Forecasting
duration = 31 # two week predictions
test_predictions = []
first_eval_batch = TS_Scaled[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))
for i in range(duration):
   current_pred = best_model_AMZN.predict(current_batch)[0]
   test_predictions.append(current_pred) 
   current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
true_predictions = stage_transformer.inverse_transform(test_predictions)
  
# Current date 
start_date = Last_Month.index[0]
# Generate list of dates 
date_sequence = pd.date_range(start=start_date, periods=duration, freq='B')

# Fit ARIMA
model_arima = auto_arima(TS)
model_arima.fit(TS)
model_arima.summary()
arima_preds = model_arima.predict(n_periods = 31)
forecast = pd.DataFrame(arima_preds, index = AMZN.index[train_size:train_size + 31], columns=['Prediction'])

# Fit HW
HW = ExponentialSmoothing(TS, trend = 'add', seasonal = 'add', seasonal_periods = 365).fit()
HW_preds = HW.forecast(31)

# Comparison
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(Last_Month, color = 'b', label = "AMZN Stock")
plt.plot(date_sequence, true_predictions, color = 'r', label = "LSTM")
plt.plot(date_sequence, arima_preds, color = 'g', label = "ARIMA")
plt.plot(date_sequence, HW_preds, color = 'y', label = "Holt-Winters")
plt.legend()
ax.set_ylabel("Closing Price (Stationary)")
ax.set_xlabel("Date")
ax.set_title("Comparison of Forecasts on AMZN Stock")
plt.xticks(rotation = 45, ha = 'right', fontsize = 6)
ax.xaxis.set_major_locator(MaxNLocator(nbins=31))
plt.savefig("Figures/AMZN_LSTM_vs_ARIMA_vs_HW.png")
plt.clf()

# ------------------------------------------------------------------------
# CAT -------------------------------------------------------------------
# ------------------------------------------------------------------------

# Reading in data
CAT_Raw = pd.read_csv("Data/Stationary_CAT.csv",index_col= "Date", parse_dates = True)

Last_Month = CAT_Raw.iloc[-31:]
CAT = CAT_Raw.iloc[0:-32]

# Splitting dataset for cross-validation
train_test_split = 0.9
train_size = int(len(CAT) * train_test_split) # Use 90% of data for training
train = CAT.iloc[0:train_size] # Selecting closing price as target
test = CAT.iloc[train_size:len(CAT)]

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

# Fitting model without loop
with tf.device('/device:GPU:0'): 
   best_model_CAT.fit(training, epochs = 5000, validation_data = validation)
   
# Whole time-series for forecasting
TS = CAT.values.flatten()
TS = TS.reshape(-1,1)
TS_Scaled = stage_transformer.transform(TS)

# Forecasting
duration = 31 # two week predictions
test_predictions = []
first_eval_batch = TS_Scaled[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))
for i in range(duration):
   current_pred = best_model_CAT.predict(current_batch)[0]
   test_predictions.append(current_pred) 
   current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
true_predictions = stage_transformer.inverse_transform(test_predictions)
  
# Current date 
start_date = Last_Month.index[0]
# Generate list of dates 
date_sequence = pd.date_range(start=start_date, periods=duration, freq='B')

# Fit ARIMA
model_arima = auto_arima(TS)
model_arima.fit(TS)
model_arima.summary()
arima_preds = model_arima.predict(n_periods = 31)
forecast = pd.DataFrame(arima_preds, index = CAT.index[train_size:train_size + 31], columns=['Prediction'])

# Fit HW
HW = ExponentialSmoothing(TS, trend = 'add', seasonal = 'add', seasonal_periods = 365).fit()
HW_preds = HW.forecast(31)

# Comparison
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(Last_Month, color = 'b', label = "CAT Stock")
plt.plot(date_sequence, true_predictions, color = 'r', label = "LSTM")
plt.plot(date_sequence, arima_preds, color = 'g', label = "ARIMA")
plt.plot(date_sequence, HW_preds, color = 'y', label = "Holt-Winters")
plt.legend()
ax.set_ylabel("Closing Price (Stationary)")
ax.set_xlabel("Date")
ax.set_title("Comparison of Forecasts on CAT Stock")
plt.xticks(rotation = 45, ha = 'right', fontsize = 6)
ax.xaxis.set_major_locator(MaxNLocator(nbins=31))
plt.savefig("Figures/CAT_LSTM_vs_ARIMA_vs_HW.png")
plt.clf()

# ------------------------------------------------------------------------
# NVDA -------------------------------------------------------------------
# ------------------------------------------------------------------------

# Reading in data
NVDA_Raw = pd.read_csv("Data/Stationary_NVDA.csv",index_col= "Date", parse_dates = True)

Last_Month = NVDA_Raw.iloc[-31:]
NVDA = NVDA_Raw.iloc[0:-32]

# Splitting dataset for cross-validation
train_test_split = 0.9
train_size = int(len(NVDA) * train_test_split) # Use 90% of data for training
train = NVDA.iloc[0:train_size] # Selecting closing price as target
test = NVDA.iloc[train_size:len(NVDA)]
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

# Whole time-series for forecasting
TS = NVDA.values.flatten()
TS = TS.reshape(-1,1)
TS_Scaled = stage_transformer.transform(TS)

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

# Fitting model without loop
with tf.device('/device:GPU:0'): 
   best_model_NVDA.fit(training, epochs = 5000, validation_data = validation)
   
# Whole time-series for forecasting
TS = NVDA.values.flatten()
TS = TS.reshape(-1,1)
TS_Scaled = stage_transformer.transform(TS)

# Forecasting
duration = 31 # two week predictions
test_predictions = []
first_eval_batch = TS_Scaled[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))
for i in range(duration):
   current_pred = best_model_NVDA.predict(current_batch)[0]
   test_predictions.append(current_pred) 
   current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
true_predictions = stage_transformer.inverse_transform(test_predictions)
  
# Current date 
start_date = Last_Month.index[0]
# Generate list of dates 
date_sequence = pd.date_range(start=start_date, periods=duration, freq='B')

# Fit ARIMA
model_arima = auto_arima(TS)
model_arima.fit(TS)
model_arima.summary()
arima_preds = model_arima.predict(n_periods = 31)
forecast = pd.DataFrame(arima_preds, index = NVDA.index[train_size:train_size + 31], columns=['Prediction'])

# Fit HW
HW = ExponentialSmoothing(TS, trend = 'add', seasonal = 'add', seasonal_periods = 365).fit()
HW_preds = HW.forecast(31)

# Comparison
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(Last_Month, color = 'b', label = "NVDA Stock")
plt.plot(date_sequence, true_predictions, color = 'r', label = "LSTM")
plt.plot(date_sequence, arima_preds, color = 'g', label = "ARIMA")
plt.plot(date_sequence, HW_preds, color = 'y', label = "Holt-Winters")
plt.legend()
ax.set_ylabel("Closing Price (Stationary)")
ax.set_xlabel("Date")
ax.set_title("Comparison of Forecasts on NVDA Stock")
plt.xticks(rotation = 45, ha = 'right', fontsize = 6)
ax.xaxis.set_major_locator(MaxNLocator(nbins=31))
plt.savefig("Figures/NVDA_LSTM_vs_ARIMA_vs_HW.png")
plt.clf()