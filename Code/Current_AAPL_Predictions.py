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


##########################################################################
#                                                                        #
#             Before using this script, visit                            #
#             the NASDAQ website and download                            #
#             the most recent historical data                            #
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
   best_model_AAPL.fit(training, epochs = 7500, validation_data = validation)
   
# Whole time-series for forecasting
TS = AAPL.values.flatten()
TS = TS.reshape(-1,1)
TS_Scaled = stage_transformer.transform(TS)

# Forecasting
duration = 62 # two month predictions
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

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(date_sequence, true_predictions, color = 'r', label = "LSTM Predictions")
plt.plot(Last_Month, color = 'b', label = "AAPL Stock Last Month")
plt.legend()
ax.set_ylabel("Closing Price (stationary)")
ax.set_xlabel("Date")
ax.set_title("LSTM Predictions on AAPL Stock")
plt.xticks(rotation = 45, ha = 'right', fontsize = 6)
ax.xaxis.set_major_locator(MaxNLocator(nbins=31))
fig.tight_layout()
plt.savefig("Figures/AAPL_Last_Month_&_Next_Month.png")
plt.clf()
