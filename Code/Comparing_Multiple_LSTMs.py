# ----------------------------------------------------------------------
# Libraries ------------------------------------------------------------
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import RobustScaler
import tensorflow as tf

# Confirming GPU is being used
tf.test.is_gpu_available()
tf.config.list_physical_devices('GPU')

# ------------------------------------------------------------------------
# AAPL -------------------------------------------------------------------
# ------------------------------------------------------------------------

# Read in stationary data
AAPL = pd.read_csv("Data/Stationary_AAPL.csv",index_col= "Date", parse_dates = True)

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


# Train 5 different models with same structure
for j in range(5):
    model = tf.keras.models.load_model('Models/Bayes_HT_AAPL.keras') 
    
    for z in range(5):
      with tf.device('/device:GPU:0'): 
         model.fit(training, epochs = 500, validation_data = validation)

      duration = 14
      test_predictions = []
      first_eval_batch = scaled_train[-n_input:]
      current_batch = first_eval_batch.reshape((1, n_input, n_features))
      for i in range(duration):
         current_pred = model.predict(current_batch)[0]
         test_predictions.append(current_pred) 
         current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
      true_predictions = stage_transformer.inverse_transform(test_predictions)
      locals()[f'Model_{j}_Predictions_Gen_{z}'] = true_predictions

# Comparing models in generation 1
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "AAPL Stock")
plt.plot(Model_0_Predictions_Gen_0, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_0, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_0, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_0, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_0, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 1)")
plt.show()

# Comparing models in generation 2
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "AAPL Stock")
plt.plot(Model_0_Predictions_Gen_1, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_1, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_1, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_1, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_1, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 2)")
plt.show()

# Comparing models in generation 3
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "AAPL Stock")
plt.plot(Model_0_Predictions_Gen_2, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_2, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_2, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_2, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_2, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 3)")
plt.show()


# Comparing models in generation 4
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "AAPL Stock")
plt.plot(Model_0_Predictions_Gen_3, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_3, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_3, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_3, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_3, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 4)")
plt.show()


# Comparing models in generation 5
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "AAPL Stock")
plt.plot(Model_0_Predictions_Gen_4, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_4, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_4, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_4, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_4, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 5)")
plt.show()

# ---------------------------------------------------------------------
# AMZN ----------------------------------------------------------------
# ---------------------------------------------------------------------

# Read in stationary data
AMZN = pd.read_csv("Data/Stationary_AMZN.csv",index_col= "Date", parse_dates = True)

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


# Train 5 different models with same structure
for j in range(5):
    model = tf.keras.models.load_model('Models/Bayes_HT_AMZN.keras') 
    
    for z in range(5):
      with tf.device('/device:GPU:0'): 
         model.fit(training, epochs = 500, validation_data = validation)

      duration = 7
      test_predictions = []
      first_eval_batch = scaled_train[-n_input:]
      current_batch = first_eval_batch.reshape((1, n_input, n_features))
      for i in range(duration):
         current_pred = model.predict(current_batch)[0]
         test_predictions.append(current_pred) 
         current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
      true_predictions = stage_transformer.inverse_transform(test_predictions)
      locals()[f'Model_{j}_Predictions_Gen_{z}'] = true_predictions

# Comparing models in generation 1
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "AMZN Stock")
plt.plot(Model_0_Predictions_Gen_0, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_0, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_0, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_0, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_0, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 1)")
plt.show()

# Comparing models in generation 2
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "AMZN Stock")
plt.plot(Model_0_Predictions_Gen_1, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_1, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_1, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_1, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_1, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 2)")
plt.show()

# Comparing models in generation 3
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "AMZN Stock")
plt.plot(Model_0_Predictions_Gen_2, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_2, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_2, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_2, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_2, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 3)")
plt.show()


# Comparing models in generation 4
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "AMZN Stock")
plt.plot(Model_0_Predictions_Gen_3, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_3, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_3, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_3, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_3, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 4)")
plt.show()


# Comparing models in generation 5
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "AMZN Stock")
plt.plot(Model_0_Predictions_Gen_4, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_4, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_4, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_4, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_4, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 5)")
plt.show()

# ---------------------------------------------------------------------
# CAT -----------------------------------------------------------------
# ---------------------------------------------------------------------

# Read in stationary data
CAT = pd.read_csv("Data/Stationary_CAT.csv",index_col= "Date", parse_dates = True)

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


# Train 5 different models with same structure
for j in range(5):
    model = tf.keras.models.load_model('Models/Bayes_HT_CAT.keras') 
    
    for z in range(5):
      with tf.device('/device:GPU:0'): 
         model.fit(training, epochs = 500, validation_data = validation)

      duration = 7
      test_predictions = []
      first_eval_batch = scaled_train[-n_input:]
      current_batch = first_eval_batch.reshape((1, n_input, n_features))
      for i in range(duration):
         current_pred = model.predict(current_batch)[0]
         test_predictions.append(current_pred) 
         current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
      true_predictions = stage_transformer.inverse_transform(test_predictions)
      locals()[f'Model_{j}_Predictions_Gen_{z}'] = true_predictions

# Comparing models in generation 1
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "CAT Stock")
plt.plot(Model_0_Predictions_Gen_0, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_0, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_0, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_0, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_0, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 1)")
plt.show()

# Comparing models in generation 2
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "CAT Stock")
plt.plot(Model_0_Predictions_Gen_1, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_1, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_1, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_1, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_1, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 2)")
plt.show()

# Comparing models in generation 3
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "CAT Stock")
plt.plot(Model_0_Predictions_Gen_2, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_2, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_2, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_2, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_2, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 3)")
plt.show()


# Comparing models in generation 4
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "CAT Stock")
plt.plot(Model_0_Predictions_Gen_3, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_3, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_3, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_3, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_3, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 4)")
plt.show()


# Comparing models in generation 5
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "CAT Stock")
plt.plot(Model_0_Predictions_Gen_4, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_4, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_4, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_4, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_4, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 5)")
plt.show()

# ---------------------------------------------------------------------
# NVDA ----------------------------------------------------------------
# ---------------------------------------------------------------------

# Read in stationary data
NVDA = pd.read_csv("Data/Stationary_NVDA.csv",index_col= "Date", parse_dates = True)

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


# Train 5 different models with same structure
for j in range(5):
    model = tf.keras.models.load_model('Models/Bayes_HT_NVDA.keras') 
    
    for z in range(5):
      with tf.device('/device:GPU:0'): 
         model.fit(training, epochs = 500, validation_data = validation)

      duration = 7
      test_predictions = []
      first_eval_batch = scaled_train[-n_input:]
      current_batch = first_eval_batch.reshape((1, n_input, n_features))
      for i in range(duration):
         current_pred = model.predict(current_batch)[0]
         test_predictions.append(current_pred) 
         current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
      true_predictions = stage_transformer.inverse_transform(test_predictions)
      locals()[f'Model_{j}_Predictions_Gen_{z}'] = true_predictions

# Comparing models in generation 1
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "NVDA Stock")
plt.plot(Model_0_Predictions_Gen_0, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_0, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_0, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_0, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_0, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 1)")
plt.show()

# Comparing models in generation 2
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "NVDA Stock")
plt.plot(Model_0_Predictions_Gen_1, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_1, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_1, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_1, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_1, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 2)")
plt.show()

# Comparing models in generation 3
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "NVDA Stock")
plt.plot(Model_0_Predictions_Gen_2, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_2, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_2, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_2, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_2, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 3)")
plt.show()


# Comparing models in generation 4
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "NVDA Stock")
plt.plot(Model_0_Predictions_Gen_3, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_3, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_3, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_3, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_3, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 4)")
plt.show()


# Comparing models in generation 5
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(test[0:duration], color = 'b', label = "NVDA Stock")
plt.plot(Model_0_Predictions_Gen_4, color = 'r', label = "Model 1", linestyle = 'dashed')
plt.plot(Model_1_Predictions_Gen_4, color = 'y', label = "Model 2", linestyle = 'dashed')
plt.plot(Model_2_Predictions_Gen_4, color = 'g', label = "Model 3", linestyle = 'dashed')
plt.plot(Model_3_Predictions_Gen_4, color = 'c', label = "Model 4", linestyle = 'dashed')
plt.plot(Model_4_Predictions_Gen_4, color = 'm', label = "Model 5", linestyle = 'dashed')
plt.legend()
ax.set_ylabel("Closing Price ($)")
ax.set_xlabel("Day's Past Training Data")
ax.set_title("Comparison of Forecasts (Generation 5)")
plt.show()
