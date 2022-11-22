#!/usr/bin/env python
# coding: utf-8

# # Technical indicators for trading
# https://blog.quantinsti.com/build-technical-indicators-in-python/

# In[12]:


# Load the necessary packages and modules
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# ## 1) Moving Average

# In[11]:


# Simple Moving Average 
def SMA(data, ndays): 
    SMA = data['Close'].rolling(ndays).mean() 
    data = data.join(SMA) 
    return data

# Exponentially-weighted Moving Average 
def EWMA(data, ndays): 
    EMA = pd.Series(data['Close'].ewm(span = ndays, min_periods = ndays - 1).mean(), 
                 name = 'EWMA_' + str(ndays)) 
    data = data.join(EMA) 
    return data

# Retrieve the Goolge stock data from Yahoo finance
data = yf.download('GOOGL', start="2020-01-01", end="2022-04-30")
close = data['Close']

# Compute the 50-day SMA
n = 50
SMA = SMA(data,n)
SMA = SMA.dropna()
SMA = SMA['SMA']

# Compute the 200-day EWMA
ew = 200
EWMA = EWMA(data,ew)
EWMA = EWMA.dropna()
EWMA = EWMA['EWMA_200']

# Plotting the Google stock Price Series chart and Moving Averages below
plt.figure(figsize=(10,7))

# Set the title and axis labels
plt.title('Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')

# Plot close price and moving averages
plt.plot(data['Close'],lw=1, label='Close Price')
plt.plot(SMA,'g',lw=1, label=str(n)+'D SMA')
plt.plot(EWMA,'r', lw=1, label=str(ew)+'D EMA')

# Add a legend to the axis
plt.legend()

plt.show()


# ## 2) Bollinger Bands

# Bollinger bands involve the following calculations:
# 
# - Middle Band: 30 Day moving average
# - Upper Band: Middle Band  + 2 x 30 Day Moving Standard Deviation
# - Lower Band: Middle Band  – 2 x 30 Day Moving Standard Deviation

# In[2]:


# Load the necessary packages and modules
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# In[10]:


# Compute the Bollinger Bands 
def BBANDS(data, window, std_value):
    MA = data.Close.rolling(window).mean()
    SD = data.Close.rolling(window).std()
    data['MiddleBand'] = MA
    data['UpperBand'] = MA + (std_value * SD) 
    data['LowerBand'] = MA - (std_value * SD)
    return data
 
# Retrieve the Goolge stock data from Yahoo finance
data = yf.download('GOOGL', start="2020-01-01", end="2022-04-30")

# Compute the Bollinger Bands for Google using the 50-day Moving average
n = 50
p=2 # STD multiple
BBANDS = BBANDS(data, n, p)

# Create the plot
# pd.concat([BBANDS.Close, BBANDS.UpperBB, BBANDS.LowerBB],axis=1).plot(figsize=(9,5),)

plt.figure(figsize=(10,7))

# Set the title and axis labels
plt.title('Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price')

plt.plot(BBANDS.Close,lw=1, label='Close Price')
plt.plot(data['UpperBand'],'g',lw=1, label='+'+str(p)+'std Upper band')
plt.plot(data['MiddleBand'],'r',lw=1, label='SMA ('+str(n)+'D)')
plt.plot(data['LowerBand'],'g', lw=1, label='-'+str(p)+'std Lower band')

# Add a legend to the axis
plt.legend()

plt.show()


# ## 3) Relative Strength Index (RSI)

# **Definiton:**   <br>
# Relative strength index (RSI) is a momentum oscillator to indicate overbought and oversold conditions in the market. It oscillates between 0 and 100 and its values are below a certain level.
# 
# Usually, if the RSI line goes below 30, it indicates an oversold market whereas the RSI going above 70 indicates overbought conditions. Typically, a lookback period of 14 days is considered for its calculation and can be changed to fit the characteristics of a particular asset or trading style.

# Calculation for RSI
# 
# - Average gain   =  sum of gains in the last 14 days/14
# - Average loss  =  sum of losses in the last 14 days/14
# - Relative Strength (RS)  = Average Gain / Average Loss
# - RSI =  100 – 100 / (1+RS)

# In[5]:


# Load the necessary packages and modules
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# In[13]:


# Returns RSI values
def rsi(close, periods):
    
    close_delta = close.diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()

    rsi = ma_up / ma_down
    #rsi = 100 - (100/(1 + rsi))
    data['RSI']=100 - (100/(1 + rsi))
    return data


# Retrieve the Apple Inc. data from Yahoo finance
data = yf.download("AAPL", start="2020-01-01", end="2022-04-30")

# Call RSI function from the talib library to calculate RSI
rsi(data['Close'],14)

# Plotting the Price Series chart and the RSI below
fig = plt.figure(figsize=(10, 7))

# Define position of 1st subplot
ax = fig.add_subplot(2, 1, 1)

# Set the title and axis labels
plt.title('Apple Price Chart')
plt.xlabel('Date')
plt.ylabel('Close Price')

plt.plot(data['Close'], label='Close price')

# Add a legend to the axis
plt.legend()

# Define position of 2nd subplot
bx = fig.add_subplot(2, 1, 2)

# Set the title and axis labels
plt.title('Relative Strength Index')
plt.xlabel('Date')
plt.ylabel('RSI values')

plt.plot(data['RSI'], 'm', label='RSI')

# Add a legend to the axis
plt.legend()

plt.tight_layout()
plt.show()


# ## 4) Money Flow Index

# The Money Flow Index (MFI) is the momentum indicator that is used to measure the inflow and outflow of money over a particular time period.
# 
# MFI is calculated by accumulating the positive and negative Money Flow values and then it creates the money ratio. 

# In[9]:


# Load the necessary packages and modules
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# In[10]:


def gain(x):
    return ((x > 0) * x).sum()


def loss(x):
    return ((x < 0) * x).sum()


# Calculate money flow index
def mfi(high, low, close, volume, n=14):
    typical_price = (high + low + close)/3
    money_flow = typical_price * volume
    mf_sign = np.where(typical_price > typical_price.shift(1), 1, -1)
    signed_mf = money_flow * mf_sign
    mf_avg_gain = signed_mf.rolling(n).apply(gain, raw=True)
    mf_avg_loss = signed_mf.rolling(n).apply(loss, raw=True)
    return (100 - (100 / (1 + (mf_avg_gain / abs(mf_avg_loss))))).to_numpy()


# In[11]:


# Retrieve the Apple Inc. data from Yahoo finance
data = yf.download("AAPL", start="2020-01-01", end="2022-04-30")

data['MFI'] = mfi(data['High'], data['Low'], data['Close'], data['Volume'], 14)

# Plotting the Price Series chart and the MFI below
fig = plt.figure(figsize=(10, 7))

# Define position of 1st subplot
ax = fig.add_subplot(2, 1, 1)

# Set the title and axis labels
plt.title('Apple Price Chart')
plt.xlabel('Date')
plt.ylabel('Close Price')

plt.plot(data['Close'], label='Close price')

# Add a legend to the axis
plt.legend()

# Define position of 2nd subplot
bx = fig.add_subplot(2, 1, 2)

# Set the title and axis labels
plt.title('Money flow index')
plt.xlabel('Date')
plt.ylabel('MFI values')

plt.plot(data['MFI'], 'm', label='MFI')

# Add a legend to the axis
plt.legend()

plt.tight_layout()
plt.show()


# The output shows the chart with the close price of the stock (Apple) and Money Flow Index (MFI) indicator’s result. You must see two observations in the output above:
# 
# - Oversold levels occur below 20 and overbought levels usually occur above 80. These levels may change depending on market conditions. Level lines should cut across the highest peaks and the lowest troughs.
# - If the underlying price makes a new high or low that isn't confirmed by the MFI, this divergence can signal a price reversal.
# 
# But, it is also important to note that, oversold/overbought levels are generally not enough of the reasons to buy/sell. The trader must consider some other technical indicators as well to confirm the asset’s position in the market.

# ## 5) Average True Range (ATR)

# The Average True Range (ATR) is a technical indicator that measures the volatility of the financial market by decomposing the entire range of the price of a stock or asset for a particular period.

# In[12]:


# Load the necessary packages and modules
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


# In[13]:


# Returns ATR values
def atr(high, low, close, n=14):
    tr = np.amax(np.vstack(((high - low).to_numpy(), (abs(high - close)).to_numpy(), (abs(low - close)).to_numpy())).T, axis=1)
    return pd.Series(tr).rolling(n).mean().to_numpy()


# In[14]:


# Retrieve the Apple Inc. data from Yahoo finance
data = yf.download("AAPL", start="2020-01-01", end="2022-04-30")

data['ATR'] = atr(data['High'], data['Low'], data['Close'], 14)

# Plotting the Price Series chart and the ATR below
fig = plt.figure(figsize=(10, 7))

# Define position of 1st subplot
ax = fig.add_subplot(2, 1, 1)

# Set the title and axis labels
plt.title('Apple Price Chart')
plt.xlabel('Date')
plt.ylabel('Close Price')

plt.plot(data['Close'], label='Close price')

# Add a legend to the axis
plt.legend()

# Define position of 2nd subplot
bx = fig.add_subplot(2, 1, 2)

# Set the title and axis labels
plt.title('Average True Range')
plt.xlabel('Date')
plt.ylabel('ATR values')

plt.plot(data['ATR'] , 'm', label='ATR')

# Add a legend to the axis
plt.legend()

plt.tight_layout()
plt.show()


# The above two graphs show the Apple stock's close price and ATR value.
# 
# The ATR is a moving average, generally using 14 days of the true ranges.
# 
# In the output above, you can see that the average true range indicator is the greatest of the following: current high less the current low; the absolute value of the current high less the previous close; and the absolute value of the current low less the previous close.

# ## 6) Force Index

# __Definition:__ <br>
# The force index was created by Alexander Elder. The force index takes into account the direction of the stock price, the extent of the stock price movement, and the volume. Using these three elements it forms an oscillator that measures the buying and the selling pressure.
# 
# Each of these three factors plays an important role in the determination of the force index. For example, a big advance in prices, which is given by the extent of the price movement, shows a strong buying pressure. A big decline in heavy volume indicates strong selling pressure.

# __Calculation for Force Index :__
# 
# Example: Computing Force index(1) and Force index(15) period.
# 
# __The Force index(1) = {Close (current period) - Close (prior period)} x Current period volume__
# 
# The Force Index for the 15-day period is an exponential moving average of the 1-period Force Index.
# 
# The force index uses price and volume to determine a trend and the strength of the trend. A shorter force index can be used to determine the short-term trend, while a longer force index, for example, a 100-day force index can be used to determine the long-term trend in prices.
# 
# A force index can also be used to identify corrections in a given trend. To do so, it can be used in conjunction with a trend following indicator. For example, one can use a 22-day EMA for trend and a 2-day force index to identify corrections in the trend.

# __Python code for Force Index__ <br>
# Let us now see how using Python, we can calculate the Force Index over the period of 13 days

# In[15]:


# Load the necessary packages and modules
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# In[16]:


# Returns the Force Index 
def ForceIndex(data, ndays): 
    FI = pd.Series(data['Close'].diff(ndays) * data['Volume'], name = 'ForceIndex') 
    data = data.join(FI) 
    return data


# In[17]:


# Retrieve the Apple Inc. data from Yahoo finance
data = yf.download("AAPL", start="2020-01-01", end="2022-04-30") 

# Compute the Force Index for AAPL
n = 1
AAPL_ForceIndex = ForceIndex(data,n)
AAPL_ForceIndex['ForceIndex']


# ## 7) Ease of Movement Value (EMV) 

# Developed by Richard Arms, Ease of Movement Value (EMV) is an oscillator that attempts to quantify both price and volume into one quantity. As it takes into account both price and volume, it is useful when determining the strength of a trend.
# 
# When the EMV rises over zero it means the price is increasing with relative ease. Whereas the fall of EMV means the price is on an easy decline.

# __Calculation for EVM :__ <br>
# 
# To calculate the EMV we first calculate the distance moved.
# 
# It is given by:
# __Distance moved = ((Current High + Current Low)/2 - (Prior High + Prior Low)/2)__
# 
# We then compute the Box ratio which uses the volume and the high-low range:
# __Box ratio = (Volume / 100,000,000) / (Current High – Current Low)__
# 
# __EMV = Distance moved / Box ratio__

# In[18]:


# Load the necessary packages and modules
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# In[19]:


# Ease of Movement 
def EMV(data, ndays): 
    dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
    br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
    EMV = dm / br 
    EMV_MA = pd.Series(EMV.rolling(ndays).mean(), name = 'EMV') 
    data = data.join(EMV_MA) 
    return data 


# In[20]:


# Retrieve the AAPL data from Yahoo finance
data = yf.download("AAPL", start="2020-01-01", end="2022-04-30")  

# Compute the 14-day Ease of Movement for AAPL
n = 14
AAPL_EMV = EMV(data, n)
EMV = AAPL_EMV['EMV']

# Plotting the Price Series chart and the Ease Of Movement below
fig = plt.figure(figsize=(10, 7))

# Define position of 1st subplot
ax = fig.add_subplot(2, 1, 1)

# Set the title and axis labels
plt.title('AAPL Price Chart')
plt.xlabel('Date')
plt.ylabel('Close Price')

# Plot the close price of the Apple
plt.plot(data['Close'], label='Close price')

# Add a legend to the axis
plt.legend()

# Define position of 2nd subplot
bx = fig.add_subplot(2, 1, 2)

# Set the title and axis labels
plt.title('Ease Of Movement Chart')
plt.xlabel('Date')
plt.ylabel('EMV values')

# Plot the ease of movement
plt.plot(EMV, 'm', label='EMV(14)')

# Add a legend to the axis
plt.legend()

plt.tight_layout()
plt.show()


# # Conclusion

# Python technical indicators are quite useful for traders to predict future stock values. Every indicator is useful for a particular market condition. For example, the Average True Range (ATR) is most useful when the market is too volatile.
# 
# Hence, ATR helps measure volatility on the basis of which a trader can enter or exit the market.

# In[ ]:





# In[ ]:




