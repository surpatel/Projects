# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 14:57:34 2019

@author: Suraj
"""

import pandas_datareader as pdr
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.core import datetools

stock_name = 'JPM'
stock = pdr.get_data_yahoo(stock_name,
                          start=dt.datetime(2006,1,1),
                          end=dt.datetime(2018,12,31))

##Time Series Plot
#stock['Close'].plot(grid=True,figsize=(15,10))
#plt.show()
#
##Daily, Monthly, Quarterly data pct change/mean
#daily_close = stock['Adj Close']
#daily_pct_change = daily_close.pct_change().fillna(0)
#daily_log_returns = np.log(daily_pct_change+1)
#
#monthly = stock.resample('BM').apply(lambda x: x[-1])
#monthly = monthly.pct_change().fillna(0)
#
#quarter = stock.resample('4M').mean()
#quarter = quarter.pct_change().fillna(0)
#
##Plot Distribution of Daily Pct Change
#daily_pct_change.hist(bins=50,figsize=(15,10))
#plt.show()
#
##Cumulative daily returns
#cum_daily_return = (1+daily_pct_change).cumprod()
#cum_daily_return.plot(figsize=(15,10))
#plt.show()

def get(tickers, start, end):
    def data(ticker):
        return (pdr.get_data_yahoo(ticker,start=start,end=end))
    datas = map(data,tickers)
    return pd.concat(datas,keys=tickers,names=['Ticker','Date'])

tickers = ['MSFT','CI','JPM']

all_data = get(tickers,dt.datetime(2006,1,1),dt.datetime(2018,12,31))

#1. Explore Returns
#Plot Daily Change Historgram
daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date','Ticker','Adj Close')
daily_pct_change = daily_close_px.pct_change()
daily_pct_change.hist(bins=50,sharex=True, figsize=(12,8))
plt.show()

#Plot Daily Change Scatter Matrix
pd.scatter_matrix(daily_pct_change, diagonal='kde',alpha=0.1,figsize=(12,12))
plt.show()

#2. Explore Moving Windows
#Moving Average
adj_close_px = stock['Adj Close']
stock['40'] = adj_close_px.rolling(window=40).mean()
stock['252'] = adj_close_px.rolling(window=252).mean()
stock[['Adj Close','40','252']].plot(figsize=(15,10))
plt.show()

#Moving Volatiltiy
min_periods = 75
vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods)
vol.plot(figsize=(15,10))
plt.show()

#Ordinary Least Squares Regression
all_adj_close = all_data[['Adj Close']]
all_returns = np.log(all_adj_close/all_adj_close.shift(1))
jpm_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker')=='JPM']
jpm_returns.index = jpm_returns.index.droplevel('Ticker')
msft_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker')=='MSFT']
msft_returns.index = msft_returns.index.droplevel('Ticker')
return_data = pd.concat([jpm_returns, msft_returns], axis=1)[1:]
return_data.columns = ['JPM','MSFT']
X = sm.add_constant(return_data['JPM'])
model = sm.OLS(return_data['MSFT'],X).fit()
print(model.summary())

#Plot Returns and Regression
plt.figure(figsize=(15,10))
plt.plot(return_data['JPM'], return_data['MSFT'], 'r.')
ax = plt.axis()
x = np.linspace(ax[0],ax[1]+0.01)
plt.plot(x,model.params[0] + model.params[1] * x, 'b', lw=2)
plt.grid(True)
plt.axis('tight')
plt.xlabel('JP Morgan Returns')
plt.ylabel('Microsoft Returns')
plt.show()

#Plot Rolling Correlations
return_data['MSFT'].rolling(window=252).corr(return_data['JPM']).plot(figsize=(15,10))
plt.show()

#3. Trading Strategies

#Moving Average Crossover
short_window = 40
long_window = 100

signals = pd.DataFrame(index=stock.index)
signals['signal'] = 0.0
signals['short_mavg'] = stock['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
signals['long_mavg'] = stock['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
signals['positions'] = signals['signal'].diff()
print(signals)

fig = plt.figure(figsize=(20,15))
ax1 = fig.add_subplot(111,ylabel='Price in $')
stock['Close'].plot(ax=ax1,color='b',lw=2.)
signals[['short_mavg','long_mavg']].plot(ax=ax1, lw=2.)
ax1.plot(signals.loc[signals.positions == 1.0].index, signals.short_mavg[signals.positions == 1.0], '^', markersize=10, color='g')
ax1.plot(signals.loc[signals.positions == -1.0].index, signals.short_mavg[signals.positions == -1.0], 'v', markersize=10, color='r')
plt.show()

#Backtesting Strategy with Pandas
initial_capital = float(100000.0)
positions = pd.DataFrame(index=signals.index).fillna(0.0)
positions[stock_name] = 100*signals['signal'] #Buy 100 shares
portfolio = positions.multiply(stock['Adj Close'], axis=0) #Initialize portfolio with value owned
pos_diff = positions.diff() #Store difference in shares owned
portfolio['holdings'] = (positions.multiply(stock['Adj Close'], axis=0)).sum(axis=1) #Add holdings to portfolio
portfolio['cash'] = initial_capital - (pos_diff.multiply(stock['Adj Close'], axis=0)).sum(axis=1).cumsum() #Add cash to portfolio
portfolio['total'] = portfolio['cash'] + portfolio['holdings']
portfolio['returns'] = portfolio['total'].pct_change()

fig = plt.figure(figsize=(20,15))
ax1 = fig.add_subplot(111, ylabel='Portfolio Value in $')
portfolio['total'].plot(ax=ax1, lw=2.)
ax1.plot(portfolio.loc[signals.positions == 1.0].index, portfolio.total[signals.positions == 1.0], '^', markersize=10, color='g')
ax1.plot(portfolio.loc[signals.positions == -1.0].index, portfolio.total[signals.positions == -1.0], 'v', markersize=10, color='r')
plt.show()

#Evaluating Trading Strategy
#Sharpe Ratio
returns = portfolio['returns']
sharpe_ratio = np.sqrt(252) * (returns.mean()/returns.std())
print(sharpe_ratio)

#Max Drawdown
window = 252
rolling_max = stock['Adj Close'].rolling(window, min_periods=1).max()
daily_drawdown = stock['Adj Close']/rolling_max - 1.0
max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()
daily_drawdown.plot(figsize=(20,15))
max_daily_drawdown.plot(figsize=(20,15))
plt.show()

#CAGR
days = (stock.index[-1] - stock.index[0]).days
cagr = ((((stock['Adj Close'][-1]) / stock['Adj Close'][1])) ** (365.0/days)) - 1
print(cagr)