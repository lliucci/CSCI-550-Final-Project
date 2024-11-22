# ----------------------------------------------------------------------
# Libraries ------------------------------------------------------------
# ----------------------------------------------------------------------

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf 

# ------------------------------------------------------------------------
# AAPL -------------------------------------------------------------------
# ------------------------------------------------------------------------
  
# initialize parameters 
end_date = datetime.now()
start_date = end_date - timedelta(days=3650)
  
# Read in most recent 10 year historic data
AAPL = yf.download('AAPL', start = start_date, 
                    end = end_date) 

# Decomposing for stationarity
decomposition = sm.tsa.seasonal_decompose(AAPL['Close'], model='additive', period = 365)

# Plot the components
decomposition.plot()
plt.show()

# Extract stationary TS
AAPL = decomposition.seasonal

# Save stationary data
AAPL.to_csv("Data/Stationary_AAPL.csv")

# ------------------------------------------------------------------------
# AMZN -------------------------------------------------------------------
# ------------------------------------------------------------------------
  
# initialize parameters 
end_date = datetime.now()
start_date = end_date - timedelta(days=3650)
  
# Read in most recent 10 year historic data
AMZN = yf.download('AMZN', start = start_date, 
                    end = end_date) 

# Decomposing for stationarity
decomposition = sm.tsa.seasonal_decompose(AMZN['Close'], model='additive', period = 365)

# Plot the components
decomposition.plot()
plt.show()

# Extract stationary TS
AMZN = decomposition.seasonal

# Save stationary data
AMZN.to_csv("Data/Stationary_AMZN.csv")

# ------------------------------------------------------------------------
# CAT --------------------------------------------------------------------
# ------------------------------------------------------------------------
  
# initialize parameters 
end_date = datetime.now()
start_date = end_date - timedelta(days=3650)
  
# Read in most recent 10 year historic data
CAT = yf.download('CAT', start = start_date, 
                    end = end_date) 

# Decomposing for stationarity
decomposition = sm.tsa.seasonal_decompose(CAT['Close'], model='additive', period = 365)

# Plot the components
decomposition.plot()
plt.show()

# Extract stationary TS
CAT = decomposition.seasonal

# Save stationary data
CAT.to_csv("Data/Stationary_CAT.csv")

# ------------------------------------------------------------------------
# NVDA -------------------------------------------------------------------
# ------------------------------------------------------------------------
  
# initialize parameters 
end_date = datetime.now()
start_date = end_date - timedelta(days=3650)
  
# Read in most recent 10 year historic data
NVDA = yf.download('NVDA', start = start_date, 
                    end = end_date) 

# Decomposing for stationarity
decomposition = sm.tsa.seasonal_decompose(NVDA['Close'], model='additive', period = 365)

# Plot the components
decomposition.plot()
plt.show()

# Extract stationary TS
NVDA = decomposition.seasonal

# Save stationary data
NVDA.to_csv("Data/Stationary_NVDA.csv")
