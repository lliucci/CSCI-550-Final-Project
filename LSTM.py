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

# -----------------------------------------------------------------------
# Setting Up LSTM Architecture ------------------------------------------
# -----------------------------------------------------------------------

def build_model(hp):
    model = Sequential() # layers are added sequentially
    for i in range(1, hp.Int("num_layers", 2, 6)):
                model.add(LSTM(hp.Int('layer_neurons', min_value = 8, max_value = 64), 
                    activation = 'tanh', 
                    input_shape = (n_input, n_features),
                    return_sequences=True,
                    kernel_regularizer=regularizers.L1(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
                model.add(Dropout(hp.Choice('dropout_1', values = [0.01, 0.05, 0.1, 0.15])))
    model.add(LSTM(hp.Int('final_layer_neurons', min_value = 8, max_value = 64), 
                    activation = 'tanh', 
                    input_shape = (n_input, n_features),
                    return_sequences=False,
                    kernel_regularizer=regularizers.L1(0.001),
                    activity_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(hp.Choice('dropout_final', values = [0.01, 0.05, 0.1, 0.15])))
    model.add(Dense(1))
    model.compile(optimizer = Adam(learning_rate=0.001,
                                   clipnorm = 1,
                                   clipvalue = 0.5), 
                loss = 'mse')
    return(model)

# ------------------------------------------------------------------------
# AAPL Data --------------------------------------------------------------
# ------------------------------------------------------------------------

# Reading in data
AAPL = pd.read_csv("Data/AAPL.csv",index_col= "Date", parse_dates = True)
AAPL['Close/Last'] = AAPL['Close/Last'].str.replace('$', '')

# Splitting dataset for cross-validation
train_test_split = 0.90
train_size = int(len(AAPL) * train_test_split) # Use 90% of data for training
train = AAPL.iloc[0:train_size]['Close/Last'] # Selecting closing price as target
test = AAPL.iloc[train_size:len(AAPL)] ['Close/Last']

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

# Initiate folder for saving searched models
LOG_DIR = f"{int(time.time())}" 

# Settings for trials
tuner = BayesianOptimization(
        build_model,
        objective="val_loss",
        max_trials=50,
        executions_per_trial=1,
        directory = f"Models/AAPL/{LOG_DIR}")

# Searching
tuner.search(
    x = training,
    epochs = 100,
    validation_data = validation
    )

# Obtain best model from search
best_model_AAPL = tuner.get_best_models()[0]

# Dimensions of model
best_model_AAPL.summary()

# Getting layer specifications
for layer in best_model_AAPL.layers:
    layer_config = layer.get_config()
    print(layer_config)

# Save best model
best_model_AAPL.save("Models/Bayes_HT_AAPL.keras")

# Load best model
best_model_AAPL = tf.keras.models.load_model('Models/Bayes_HT_AAPL.keras')    

# Training Best SSM Model

for j in range(12):
    
   # Fitting model  
   with tf.device('/device:GPU:0'): 
        best_model_AAPL.fit(training, epochs = 1000, validation_data = validation)

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
   plt.savefig(f"Model Diagnostics/AAPL_model_{j}.png")
   plt.clf()
   
# Check when loss levels out
loss_per_epoch = best_model_AAPL.history.history["loss"]
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
plt.show()
