# ----------------------------------------------------------------------
# Libraries ------------------------------------------------------------
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
import yfinance as yf
from datetime import datetime, timedelta
from keras import regularizers


# --------------------------------------------------
# Read In Data -------------------------------------
# --------------------------------------------------

# initialize parameters 
end_date = datetime.now()
start_date = end_date - timedelta(days=3650)
  
# Read in most recent 10 year historic data
AAPL = yf.download('AAPL', start = start_date, 
                    end = end_date) 

input_length = 31

train = AAPL.iloc[:-62]
test = AAPL.iloc[-62:]

scaler = RobustScaler()
scaler = scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

scaled_data = scaler.transform(AAPL)

scaled_train.shape

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 2:6])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)

trainX,trainY=createXY(scaled_train,30)
testX,testY=createXY(scaled_test,30)

dataX, dataY = createXY(scaled_data, 30)

# First batch training input
trainX[0]
# First batch training response value
trainY.shape


model = Sequential() # layers are added sequentially
model.add(LSTM(8, 
                activation = 'tanh', 
                input_shape = (trainX.shape[1], trainX.shape[2]),
                return_sequences=True,
                kernel_regularizer=regularizers.L1(0.001),
                activity_regularizer=regularizers.L2(0.001)))
model.add(Dropout(0.05))
model.add(LSTM(8, 
                activation = 'tanh', 
                input_shape = (trainX.shape[1], trainX.shape[2]),
                return_sequences=True,
                kernel_regularizer=regularizers.L1(0.001),
                activity_regularizer=regularizers.L2(0.001)))
model.add(Dropout(0.05))
model.add(LSTM(64, 
                activation = 'tanh', 
                input_shape = (trainX.shape[1], trainX.shape[2]),
                return_sequences=False,
                kernel_regularizer=regularizers.L1(0.001),
                activity_regularizer=regularizers.L2(0.001)))
model.add(Dropout(0.15))
model.add(Dense(1))
model.compile(optimizer = Adam(learning_rate=0.001,
                               clipnorm = 1,
                               clipvalue = 0.5), 
            loss = 'mse')

with tf.device('/device:GPU:0'): 
   model.fit(trainX, trainY, epochs = 10, validation_split = 0.9)
   
predY = model.predict(testX)

predY_array = np.repeat(predY, 6, axis=-1)

true_predY = scaler.inverse_transform(np.reshape(predY_array,(len(predY),6)))[:,0]

plt.plot(true_predY, color = 'red')
plt.plot(test[0], color = 'blue')
plt.show()