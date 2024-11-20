# ----------------------------------------------------------------------
# Libraries ------------------------------------------------------------
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
import statsmodels.api as sm
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
from tsbootstrap import MovingBlockBootstrap

# Confirming GPU is being used
tf.test.is_gpu_available()
tf.config.list_physical_devices('GPU')

# ------------------------------------------------------------------------
# AAPL -------------------------------------------------------------------
# ------------------------------------------------------------------------

# Reading in data
AAPL = pd.read_csv("Data/AAPL.csv",index_col= "Date", parse_dates = True)
AAPL['Close/Last'] = AAPL['Close/Last'].str.replace('$', '')

# Decomposing for stationarity
decomposition = sm.tsa.seasonal_decompose(AAPL['Close/Last'], model='additive', period = 365)

# Plot the components
decomposition.plot()
# plt.show()

# Extract stationary TS
AAPL = decomposition.seasonal

# Splitting dataset for cross-validation
train_test_split = 0.9
train_size = int(len(AAPL) * train_test_split) # Use 90% of data for training
true_train = AAPL.iloc[0:train_size]['Close/Last'] # Selecting closing price as target
true_test = AAPL.iloc[train_size:len(AAPL)] ['Close/Last']
true_test = pd.to_numeric(true_test)

# Reshaping data sets from Panda Series to 1D Array
true_train = true_train.values.flatten()
true_train = true_train.reshape(-1,1)
true_test = true_test.values.flatten()
true_test = true_test.reshape(-1,1)

# Scaling time-series train and test values values
stage_transformer = RobustScaler()
stage_transformer = stage_transformer.fit(true_train)
scaled_true_train = stage_transformer.transform(true_train)
scaled_true_test = stage_transformer.transform(true_test)

Boots = []

for j in range(100):
    
    # Generate bootstrapped samples
    return_indices = False
    bootstrapped_samples = mbb.bootstrap(
        AAPL, return_indices=return_indices)
        
    # Collect bootstrap samples
    X_bootstrapped = []
    for data in bootstrapped_samples:
        X_bootstrapped.append(data)

    X_bootstrapped = np.array(X_bootstrapped)

    AAPL_BS = pd.DataFrame(data=X_bootstrapped[0,:,0])

    # Splitting dataset for cross-validation
    train_size = int(len(AAPL_BS) * 0.9) # Use 90% of data for training
    train = AAPL_BS.iloc[0:train_size]
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

    # Define inputs  
    n_input = 31
    n_features = 1
    generator = TimeseriesGenerator(scaled_train, scaled_train, 
                                    length = n_input,
                                    batch_size = 5000)

    model = Sequential() # layers are added sequentially
    model.add(LSTM(40, 
                    activation = 'tanh',
                    input_shape = (n_input, n_features),
                    return_sequences=True,
                    kernel_regularizer=regularizers.L2(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.01))
    model.add(LSTM(30, 
                    activation = 'tanh', 
                    input_shape = (n_input, n_features),
                    return_sequences=True,
                    kernel_regularizer=regularizers.L2(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.1))
    model.add(LSTM(48, 
                    activation = 'tanh', 
                    input_shape = (n_input, n_features),
                    return_sequences=False,
                    kernel_regularizer=regularizers.L2(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.01))
    model.add(Dense(1))
    model.compile(optimizer = Adam(learning_rate=0.0001,
                                clipnorm = 1), 
                loss = 'mse')
    gen_output = TimeseriesGenerator(scaled_test, scaled_test, 
                                    length = n_input,
                                    batch_size = 1000)

    with tf.device('/device:GPU:0'): 
        model.fit(generator, epochs = 10000, validation_data = gen_output)

    duration = 7
    test_predictions = []
    first_eval_batch = true_scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))
    for i in range(duration):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred) 
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    true_predictions = stage_transformer.inverse_transform(test_predictions)
    Boots.append([z[0] for z in true_predictions])

# Ask Bryan
Plotting = [
    [i[e] for i in Boots] #... take the eth element from ith array
    for e in range(len(Boots[0])) # for each e in 0:30...
    ]

# Save as csv
df = pd.DataFrame(Plotting)
df.to_csv("AAPL_Pred_Intervals.csv")

# ------------------------------------------------------------------------
# AMZN -------------------------------------------------------------------
# ------------------------------------------------------------------------

# Reading in data
AMZN = pd.read_csv("Data/AMZN.csv",index_col= "Date", parse_dates = True)
AMZN['Close/Last'] = AMZN['Close/Last'].str.replace('$', '')

# Decomposing for stationarity
decomposition = sm.tsa.seasonal_decompose(AMZN['Close/Last'], model='additive', period = 365)

# Plot the components
decomposition.plot()
# plt.show()

# Extract stationary TS
AMZN = decomposition.seasonal

# Splitting dataset for cross-validation
train_test_split = 0.9
train_size = int(len(AMZN) * train_test_split) # Use 90% of data for training
true_train = AMZN.iloc[0:train_size]['Close/Last'] # Selecting closing price as target
true_test = AMZN.iloc[train_size:len(AMZN)] ['Close/Last']
true_test = pd.to_numeric(test)

# Reshaping data sets from Panda Series to 1D Array
true_train = true_train.values.flatten()
true_train = true_train.reshape(-1,1)
true_test = true_test.values.flatten()
true_test = true_test.reshape(-1,1)

# Scaling time-series train and test values values
stage_transformer = RobustScaler()
stage_transformer = stage_transformer.fit(true_train)
scaled_true_train = stage_transformer.transform(true_train)
scaled_true_test = stage_transformer.transform(true_test)

Boots = []

for j in range(100):
    
    # Generate bootstrapped samples
    return_indices = False
    bootstrapped_samples = mbb.bootstrap(
        AMZN, return_indices=return_indices)
        
    # Collect bootstrap samples
    X_bootstrapped = []
    for data in bootstrapped_samples:
        X_bootstrapped.append(data)

    X_bootstrapped = np.array(X_bootstrapped)

    AMZN_BS = pd.DataFrame(data=X_bootstrapped[0,:,0])

    # Splitting dataset for cross-validation
    train_size = int(len(AMZN_BS) * 0.9) # Use 90% of data for training
    train = AMZN_BS.iloc[0:train_size]
    test = AMZN.iloc[train_size:len(AMZN)] 

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
    n_input = 31
    n_features = 1
    generator = TimeseriesGenerator(scaled_train, scaled_train, 
                                    length = n_input,
                                    batch_size = 5000)

    model = Sequential() # layers are added sequentially
    model.add(LSTM(64, 
                    activation = 'tanh',
                    input_shape = (n_input, n_features),
                    return_sequences=True,
                    kernel_regularizer=regularizers.L2(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.01))
    model.add(LSTM(8, 
                    activation = 'tanh', 
                    input_shape = (n_input, n_features),
                    return_sequences=True,
                    kernel_regularizer=regularizers.L2(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.01))
    model.add(LSTM(64, 
                    activation = 'tanh', 
                    input_shape = (n_input, n_features),
                    return_sequences=False,
                    kernel_regularizer=regularizers.L2(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.01))
    model.add(Dense(1))
    model.compile(optimizer = Adam(learning_rate=0.0001,
                                clipnorm = 1), 
                loss = 'mse')
    gen_output = TimeseriesGenerator(scaled_test, scaled_test, 
                                    length = n_input,
                                    batch_size = 1000)

    with tf.device('/device:GPU:0'): 
        model.fit(generator, epochs = 10000, validation_data = gen_output)

    duration = 7
    test_predictions = []
    first_eval_batch = true_scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))
    for i in range(duration):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred) 
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    true_predictions = stage_transformer.inverse_transform(test_predictions)
    Boots.append([z[0] for z in true_predictions])

# Ask Bryan
Plotting = [
    [i[e] for i in Boots] #... take the eth element from ith array
    for e in range(len(Boots[0])) # for each e in 0:30...
    ]

# Save as csv
df = pd.DataFrame(Plotting)
df.to_csv("AMZN_Pred_Intervals.csv")


# ------------------------------------------------------------------------
# CAT --------------------------------------------------------------------
# ------------------------------------------------------------------------

# Reading in data
CAT = pd.read_csv("Data/CAT.csv",index_col= "Date", parse_dates = True)
CAT['Close/Last'] = CAT['Close/Last'].str.replace('$', '')

# Decomposing for stationarity
decomposition = sm.tsa.seasonal_decompose(CAT['Close/Last'], model='additive', period = 365)

# Plot the components
decomposition.plot()
# plt.show()

# Extract stationary TS
CAT = decomposition.seasonal

# Splitting dataset for cross-validation
train_test_split = 0.9
train_size = int(len(CAT) * train_test_split) # Use 90% of data for training
true_train = CAT.iloc[0:train_size]['Close/Last'] # Selecting closing price as target
true_test = CAT.iloc[train_size:len(CAT)] ['Close/Last']
true_test = pd.to_numeric(test)

# Reshaping data sets from Panda Series to 1D Array
true_train = true_train.values.flatten()
true_train = true_train.reshape(-1,1)
true_test = true_test.values.flatten()
true_test = true_test.reshape(-1,1)

# Scaling time-series train and test values values
stage_transformer = RobustScaler()
stage_transformer = stage_transformer.fit(true_train)
scaled_true_train = stage_transformer.transform(true_train)
scaled_true_test = stage_transformer.transform(true_test)

Boots = []

for j in range(100):
    
    # Generate bootstrapped samples
    return_indices = False
    bootstrapped_samples = mbb.bootstrap(
        CAT, return_indices=return_indices)
        
    # Collect bootstrap samples
    X_bootstrapped = []
    for data in bootstrapped_samples:
        X_bootstrapped.append(data)

    X_bootstrapped = np.array(X_bootstrapped)

    CAT_BS = pd.DataFrame(data=X_bootstrapped[0,:,0])

    # Splitting dataset for cross-validation
    train_size = int(len(CAT_BS) * 0.9) # Use 90% of data for training
    train = CAT_BS.iloc[0:train_size]
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
    n_input = 31
    n_features = 1
    generator = TimeseriesGenerator(scaled_train, scaled_train, 
                                    length = n_input,
                                    batch_size = 5000)

    model = Sequential() # layers are added sequentially
    model.add(LSTM(18, 
                    activation = 'tanh',
                    input_shape = (n_input, n_features),
                    return_sequences=True,
                    kernel_regularizer=regularizers.L2(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.01))
    model.add(LSTM(58, 
                    activation = 'tanh', 
                    input_shape = (n_input, n_features),
                    return_sequences=True,
                    kernel_regularizer=regularizers.L2(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.05))
    model.add(LSTM(36, 
                    activation = 'tanh', 
                    input_shape = (n_input, n_features),
                    return_sequences=False,
                    kernel_regularizer=regularizers.L2(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.01))
    model.add(Dense(1))
    model.compile(optimizer = Adam(learning_rate=0.0001,
                                clipnorm = 1), 
                loss = 'mse')
    gen_output = TimeseriesGenerator(scaled_test, scaled_test, 
                                    length = n_input,
                                    batch_size = 1000)

    with tf.device('/device:GPU:0'): 
        model.fit(generator, epochs = 10000, validation_data = gen_output)

    duration = 7
    test_predictions = []
    first_eval_batch = true_scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))
    for i in range(duration):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred) 
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    true_predictions = stage_transformer.inverse_transform(test_predictions)
    Boots.append([z[0] for z in true_predictions])

# Ask Bryan
Plotting = [
    [i[e] for i in Boots] #... take the eth element from ith array
    for e in range(len(Boots[0])) # for each e in 0:30...
    ]

# Save as csv
df = pd.DataFrame(Plotting)
df.to_csv("CAT_Pred_Intervals.csv")


# ------------------------------------------------------------------------
# NVDA -------------------------------------------------------------------
# ------------------------------------------------------------------------

# Reading in data
NVDA = pd.read_csv("Data/NVDA.csv",index_col= "Date", parse_dates = True)
NVDA['Close/Last'] = NVDA['Close/Last'].str.replace('$', '')

# Decomposing for stationarity
decomposition = sm.tsa.seasonal_decompose(NVDA['Close/Last'], model='additive', period = 365)

# Plot the components
decomposition.plot()
# plt.show()

# Extract stationary TS
NVDA = decomposition.seasonal

# Splitting dataset for cross-validation
train_test_split = 0.9
train_size = int(len(NVDA) * train_test_split) # Use 90% of data for training
true_train = NVDA.iloc[0:train_size]['Close/Last'] # Selecting closing price as target
true_test = NVDA.iloc[train_size:len(NVDA)] ['Close/Last']
true_test = pd.to_numeric(test)

# Reshaping data sets from Panda Series to 1D Array
true_train = true_train.values.flatten()
true_train = true_train.reshape(-1,1)
true_test = true_test.values.flatten()
true_test = true_test.reshape(-1,1)

# Scaling time-series train and test values values
stage_transformer = RobustScaler()
stage_transformer = stage_transformer.fit(true_train)
scaled_true_train = stage_transformer.transform(true_train)
scaled_true_test = stage_transformer.transform(true_test)

Boots = []

for j in range(100):
    
    # Generate bootstrapped samples
    return_indices = False
    bootstrapped_samples = mbb.bootstrap(
        NVDA, return_indices=return_indices)
        
    # Collect bootstrap samples
    X_bootstrapped = []
    for data in bootstrapped_samples:
        X_bootstrapped.append(data)

    X_bootstrapped = np.array(X_bootstrapped)

    NVDA_BS = pd.DataFrame(data=X_bootstrapped[0,:,0])

    # Splitting dataset for cross-validation
    train_size = int(len(NVDA_BS) * 0.9) # Use 90% of data for training
    train = NVDA_BS.iloc[0:train_size]
    test = NVDA.iloc[train_size:len(NVDA)] 

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
    n_input = 31
    n_features = 1
    generator = TimeseriesGenerator(scaled_train, scaled_train, 
                                    length = n_input,
                                    batch_size = 5000)

    model = Sequential() # layers are added sequentially
    model.add(LSTM(64, 
                    activation = 'tanh',
                    input_shape = (n_input, n_features),
                    return_sequences=True,
                    kernel_regularizer=regularizers.L2(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.01))
    model.add(LSTM(62, 
                    activation = 'tanh', 
                    input_shape = (n_input, n_features),
                    return_sequences=True,
                    kernel_regularizer=regularizers.L2(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.01))
    model.add(LSTM(31, 
                    activation = 'tanh', 
                    input_shape = (n_input, n_features),
                    return_sequences=False,
                    kernel_regularizer=regularizers.L2(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.01))
    model.add(Dense(1))
    model.compile(optimizer = Adam(learning_rate=0.0001,
                                clipnorm = 1), 
                loss = 'mse')
    gen_output = TimeseriesGenerator(scaled_test, scaled_test, 
                                    length = n_input,
                                    batch_size = 1000)

    with tf.device('/device:GPU:0'): 
        model.fit(generator, epochs = 10000, validation_data = gen_output)

    duration = 7
    test_predictions = []
    first_eval_batch = true_scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))
    for i in range(duration):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred) 
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    true_predictions = stage_transformer.inverse_transform(test_predictions)
    Boots.append([z[0] for z in true_predictions])

# Ask Bryan
Plotting = [
    [i[e] for i in Boots] #... take the eth element from ith array
    for e in range(len(Boots[0])) # for each e in 0:30...
    ]

# Save as csv
df = pd.DataFrame(Plotting)
df.to_csv("NVDA_Pred_Intervals.csv")
