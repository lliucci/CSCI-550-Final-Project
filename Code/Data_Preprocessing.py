# ----------------------------------------------------------------------
# Libraries ------------------------------------------------------------
# ----------------------------------------------------------------------

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------
# AAPL -------------------------------------------------------------------
# ------------------------------------------------------------------------

# Reading in data
AAPL = pd.read_csv("Data/AAPL.csv",index_col= "Date", parse_dates = True)
AAPL['Close/Last'] = AAPL['Close/Last'].str.replace('$', '')
AAPL = AAPL.reindex(index=AAPL.index[::-1])

# Decomposing for stationarity
decomposition = sm.tsa.seasonal_decompose(AAPL['Close/Last'], model='additive', period = 365)

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

# Reading in data
AMZN = pd.read_csv("Data/AMZN.csv",index_col= "Date", parse_dates = True)
AMZN['Close/Last'] = AMZN['Close/Last'].str.replace('$', '')
AMZN = AMZN.reindex(index=AMZN.index[::-1])

# Decomposing for stationarity
decomposition = sm.tsa.seasonal_decompose(AMZN['Close/Last'], model='additive', period = 365)

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

# Reading in data
CAT = pd.read_csv("Data/CAT.csv",index_col= "Date", parse_dates = True)
CAT['Close/Last'] = CAT['Close/Last'].str.replace('$', '')
CAT = CAT.reindex(index=CAT.index[::-1])

# Decomposing for stationarity
decomposition = sm.tsa.seasonal_decompose(CAT['Close/Last'], model='additive', period = 365)

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

# Reading in data
NVDA = pd.read_csv("Data/NVDA.csv",index_col= "Date", parse_dates = True)
NVDA['Close/Last'] = NVDA['Close/Last'].str.replace('$', '')
NVDA = NVDA.reindex(index=NVDA.index[::-1])

# Decomposing for stationarity
decomposition = sm.tsa.seasonal_decompose(NVDA['Close/Last'], model='additive', period = 365)

# Plot the components
decomposition.plot()
plt.show()

# Extract stationary TS
NVDA = decomposition.seasonal

# Save stationary data
NVDA.to_csv("Data/Stationary_NVDA.csv")
