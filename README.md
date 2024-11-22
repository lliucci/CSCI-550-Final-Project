# CSCI-550-Final-Project

## Directory
- Data: Contains all raw data from NASDAQ website, Stationary Data from Preprocessing Script, and Predictions for Prediction Intervals from the Tempest cluster.
- Models: Optimized models obtained from Tempest Cluster
- Training: Folder containing graphics of training process. Created in `Training_Loop.py`
- Comparing_Multiple_LSTMs.py: Trains multiple LSTMs on the same data with the same architecture to show variability
- Current_AAPL_Predictions.py: Trains optimized AAPL model on newest AAPL data for current forecasts
- Current_AMZN_Predictions.py: Trains optimized AMZN model on newest AMZN data for current forecasts
- Current_CAT_Predictions.py: Trains optimized CAT model on newest AMZN data for current forecasts
- Current_NVDA_Predictions.py: Trains optimized NVDA model on newest AMZN data for current forecasts
- Data_Preprocessing.py: Loads in raw data and removes trend, saving a stationary time series for each stock
- Figures.R: Used for creating higher quality plots using `ggplot2`
- Prediction_Intervals.py: Uses a moving block bootstrap to train the same model architecture on a different training set to obtain prediction intervals for observed stock data
- Training_Loop.py: Loads in models and trains tuned models, saving results at fixed intervals
- Training.py: Used to train models and obtain loss vs. epoch plots
- Tuning.py: Uses Bayesian Optimization to tune hyper-parameters for each data set