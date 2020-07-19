
#yahoo and google sources

import pandas_datareader.data as web
import datetime
start = datetime.datetime(2018, 1, 1).strftime('%Y-%m-%d')

end = datetime.datetime(2020, 1, 1).strftime('%Y-%m-%d')


# try 'yahoo' if Google doesn't work. make sure to check the website mentioned above
# search QA forums if you have any issues on this, many questions have already been answered there!

def get_yahoo_data(Stock_name, start, end):
    Data = web.DataReader(Stock_name, 'yahoo', start, end)
    return Data


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import quandl
def get_quandl_data(Stock_name, start, end):
#mydata = quandl.get("EIA/PET_RWTC_D")
    mydata = quandl.get(Stock_name,start_date=start,end_date=end)
    return  mydata

"""
##Implementation

facebook = get_yahoo_data('FB',start, end)
Tesla = get_yahoo_data('TSLA',start, end)
Amazon=get_yahoo_data('AMZN',start, end)
Apple = get_yahoo_data('AAPL', start, end)
#nasdaq=get_yahoo_data('NQ=F',start,end)
AMD = get_yahoo_data('AMD',start, end)
INTEL = get_yahoo_data('INTC',start, end)
Google = get_yahoo_data('GOOG',start, end)

plt.plot(facebook['High'])
plt.plot(Tesla['High'])
plt.plot(Google['High'])
plt.plot(Amazon['High'])
plt.plot(Apple['High'])
plt.plot(INTEL['High'])
plt.plot(AMD['High'])

plt.legend(['facebook','Tesla','google','Amazon','Apple','intel','AMD'])
plt.xlabel('Date')
plt.ylabel('US dollars')

#nasqda=get_quandl_data('NASDAQOMX/NQDMEU2300LMGBPT', start, end)

#print(nasqda.head())
#plt.plot(nasqda['High'])
plt.show()

"""
quantum_companys=['GOOG','T','IBM','INTC','HON','AMZN']
stocks_names=['GOOGLE','AT&T','IBM','INTEL','Lockheed Martin','MICROSOFT','NOKIA']

def show_portfolio(companys,company_names,start,end):
    stocks={}
    for i in  range(len(companys)):
        stocks[company_names[i]]=get_yahoo_data(companys[i],start,end)
        plt.plot(stocks[company_names[i]]['High'])
    plt.legend(company_names)
    plt.xlabel('Date')
    plt.ylabel('US dollars')
    plt.show()

#value=show_portfolio(quantum_companys,stocks_names,start,end)

import yfinance as yf

def multistocks_dataframe_yfinance(stocks,start,end,state):
    cl_price=pd.DataFrame()
    for ticker in stocks:
        cl_price[ticker]=yf.download(ticker,start,end)[state]
    return cl_price
#example
#print(multistocks_dataframe_yfinance(quantum_companys,start,end,'High'))

from yahoofinancials import YahooFinancials

def multistocks_yahoofinancials(stocks,start,end,period_number,stock_value):
    periods=["daily", "weekly", "monthly"]
    stock_value=['high','adjclose']
    prices=pd.DataFrame()
    #end_date=(datetime.date.today()).strftime('%Y-%m-%d')
    #start_date=(datetime.date.today()-datetime.timedelta(1825)).strftime('%Y-%m-%d')
    for ticker in stocks:
        #create a object
        stock_name= YahooFinancials(ticker)
        data=stock_name.get_historical_price_data(start,end,periods[period_number])
        value=data[ticker]["prices"]
        value_period=periods[period_number]
        #print(value_period)
        temp=pd.DataFrame(value)[["formatted_date","high"]]
        temp.set_index("formatted_date",inplace=True)
        temp.dropna(inplace=True)
        prices[ticker]=temp["high"]
    return prices

from alpha_vantage.timeseries import TimeSeries

#no lo implemnte

import requests
from bs4 import BeautifulSoup

##getting info from a table
"""
url='https://finance.yahoo.com/quote/%5EIXIC/history?p=%5EIXIC'
page=requests.get(url)
page_content=page.content
soup=BeautifulSoup(page_content,'html.parser')
tabl=soup.find_all('table',{'class':"W(100%) M(0)"})
for t in tabl:
    rows=t.find_all('tr')
    for row in rows:
        print(row.get_text())
        print('pasa otra linea')
"""





