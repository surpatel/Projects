# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:25:46 2019

@author: Suraj
"""

import quandl
import pandas as pd
import numpy as np
import datetime as dt
from yahoo_fin.stock_info import get_data
import matplotlib.pyplot as plt

def get_economic_data():
    authtoken = "qRy47RRsVxPSxUfSJgzy"
    
    # Real GDP Data
    real_gdp = pd.DataFrame(quandl.get("FRED/GDPC1", authtoken=authtoken)).rename(columns={'Value':'Real GDP'}).reset_index()
    real_gdp['Real GDP QTR Growth %'] = real_gdp['Real GDP'].pct_change()*100
    recession = pd.DataFrame(quandl.get("FRED/JHDUSRGDPBR", authtoken=authtoken)).rename(columns={'Value':'Recession'}).reset_index()
    real_gdp = pd.merge(real_gdp, recession, how='outer', left_on='Date', right_on='Date')
    
#    for i in range(len(real_gdp)-1):
#        if ((real_gdp.loc[i,'Real GDP QTR Growth %'] < 0) & (real_gdp.loc[i+1,'Real GDP QTR Growth %'] < 0)):
#            real_gdp.loc[i,'Recession'] = 1
#        elif ((real_gdp.loc[i,'Real GDP QTR Growth %'] > 0) & (real_gdp.loc[i+1,'Real GDP QTR Growth %'] > 0)):
#            real_gdp.loc[i,'Recession'] = 0
    real_gdp.fillna(method='ffill',inplace=True)
    
    # 10-Year Breakeven Inflation
    breakeven_inflation_10y = pd.DataFrame(quandl.get("FRED/T10YIE", authtoken=authtoken)).rename(columns={'Value':'10-Year Breakeven Inflation'}).reset_index()
    
    # Fed Funds Rate
    fed_funds_rate = pd.DataFrame(quandl.get("FRED/DFF", authtoken=authtoken)).rename(columns={'Value':'Federal Funds Rate'}).reset_index()
    
    # Treasury Rates & Spreads
    tsy_10y = pd.DataFrame(quandl.get("FRED/DGS10", authtoken=authtoken)).rename(columns={'Value':'10-Year TSY'}).reset_index()
    tsy_2y = pd.DataFrame(quandl.get("FRED/DGS2", authtoken=authtoken)).rename(columns={'Value':'2-Year TSY'}).reset_index()
    ted_spread = pd.DataFrame(quandl.get("FRED/TEDRATE", authtoken=authtoken)).rename(columns={'Value':'TED Spread'}).reset_index()
    tsy = pd.merge(tsy_10y,tsy_2y,how='outer',on='Date')
    tsy['10y-2y Spread'] = tsy['10-Year TSY'] - tsy['2-Year TSY']
    tsy = pd.merge(tsy,ted_spread,how='outer',on='Date')
    
    # Unemployment Rate
    unemployment_rate = pd.DataFrame(quandl.get("FRED/UNRATE", authtoken=authtoken)).rename(columns={'Value':'Unemployment Rate'}).reset_index()
    
    # Crude Oil
    crude_oil = pd.DataFrame(quandl.get("FRED/DCOILWTICO", authtoken=authtoken)).rename(columns={'Value':'Crude Oil (WTI)'}).reset_index()
    
    # Dollar Index
    dollar_index = pd.DataFrame(quandl.get("FRED/DTWEXM", authtoken=authtoken)).rename(columns={'Value':'U.S. Dollar Index'}).reset_index()
    
    # Create Final DataFrame of All Data
    index = pd.date_range(start=dt.date(1970,1,1),end=dt.datetime.now().date(),freq='D')
    df = pd.DataFrame(index=index)
    df = pd.merge(df,real_gdp,how='left',left_on=index, right_on='Date')
    df = pd.merge(df,fed_funds_rate,how='left',left_on='Date', right_on='Date')
    df = pd.merge(df,tsy,how='left',left_on='Date', right_on='Date')
    df = pd.merge(df,breakeven_inflation_10y,how='left',left_on='Date',right_on='Date')
    df = pd.merge(df,unemployment_rate,how='left',left_on='Date', right_on='Date')
    df = pd.merge(df,crude_oil,how='left',left_on='Date', right_on='Date')
    df = pd.merge(df,dollar_index,how='left',left_on='Date', right_on='Date')
    df.fillna(method='ffill',inplace=True)
    
    return df

def get_adj_prices(stock, start,end):
    prices = pd.DataFrame(get_data(stock,start_date=start, end_date=end)['adjclose']).rename(columns={'adjclose':'Adj Close Price'}).reset_index()
    return prices

def get_recession_periods(full_data):
    start = []
    end = []
    
    for i in range(len(full_data)-1):
        if ((full_data.loc[i,'Recession'] == 0) & (full_data.loc[i+1,'Recession'] == 1)):
            start.append(full_data.loc[i,'Date'])
        elif ((full_data.loc[i,'Recession'] == 1) & (full_data.loc[i+1,'Recession'] == 0)):
            end.append(full_data.loc[i,'Date'])
            
    return start,end

# Inputs
start = dt.date(1987,1,1)
end = dt.datetime.today().strftime('%Y-%m-%d')
ticker = '^DJI'
index_name = 'DJIA Index'

# Pull Economic Data
economic_data = get_economic_data()
economic_data['Date'] = economic_data['Date'].dt.date
economic_data = economic_data[economic_data['Date'] >= start]

# Pull Index Data
dji_prices = get_adj_prices(ticker,start,end).rename(columns={'index':'Date'})
dji_prices['Date'] = dji_prices['Date'].dt.date

# Merge Data Sets
full_data = pd.merge(economic_data,dji_prices,how='outer',left_on='Date',right_on='Date')
full_data = full_data.rename(columns={'Adj Close Price':index_name})
full_data.fillna(method='ffill',inplace=True)

# Recession Periods
recession_starts, recession_ends = get_recession_periods(full_data)

# Plot Variables to see pattern against market
fig, ax1 = plt.subplots(figsize=(15,10))
time = np.array(full_data['Date']).reshape(-1,1)
plot1 = np.array(full_data['DJIA Index'])
ax1.plot(time, plot1, 'b-')
ax1.set_xlabel('Date')
ax1.set_ylabel('DJIA Index')
ax1.set_ylim([0,30000])
for i in range(len(recession_starts)):
    ax1.axvspan(recession_starts[i],recession_starts[i], color='y', alpha=1)
ax2 = ax1.twinx()
variable = '10y-2y Spread'
plot2 = np.array(full_data[variable])
ax2.plot(time, plot2, 'g-')
ax2.set_ylabel(variable)
plt.show()

# Correlation of Variables
correlation_matrix = full_data.corr()


