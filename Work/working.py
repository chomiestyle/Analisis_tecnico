import Pre_ProccessingData as data
import Technical_Analysis as T
import datetime
import data_visualization as V
import Strategy_functions as Sf


start = datetime.datetime(2018, 1, 1).strftime('%Y-%m-%d')

end = datetime.datetime(2020, 1, 1).strftime('%Y-%m-%d')


quantum_companys=['GOOG','T','IBM','INTC','HON','AMZN']
stocks_names=['GOOGLE','AT&T','IBM','INTEL','Lockheed Martin','Amazon']

#value=data.show_portfolio(quantum_companys,stocks_names,start,end)
#print(data.multistocks_dataframe_yfinance(quantum_companys,start,end,'High'))
#stuck=data.multistocks_yahoofinancials(quantum_companys,start,end,0,0)
#V.multistocks_show(stuck)

# Import necesary libraries
import yfinance as yf
import datetime as dt

# Download historical data for required stocks
ticker = "AMZN"
ohlcv = yf.download(ticker,dt.date.today()-dt.timedelta(1825),dt.datetime.today())
#print(ohlcv)

#Bollinger=T.BollBnd(ohlcv,20)
#print(Bollinger)
#V.BollBnd_visualization(Bollinger,ticker)
#ATR=T.ATR(ohlcv,20)
#print(ATR)
#V.ATR_visualization(ATR,ticker)
#MACD=T.MACD(ohlcv,12, 26, 9)
#print(MACD)
#V.MACD_visualization(MACD,ticker)
#RSI=T.RSI(ohlcv,20)
#print(RSI)
#V.RSI_visualization(RSI,ticker)

#slopes=T.slope(ohlcv,5)
#V.Slope_visualization(slopes,ticker)
#print(slopes)
#renko=T.renko_DF(ohlcv,120)
#print(renko)
#print(T.ADX(ohlcv,3))
#print(T.OBV(ohlcv))
#print(T.RSI(ohlcv,3))

print(Sf.CAGR(ohlcv,252))
print(Sf.volatility(ohlcv,252))
print(Sf.calmar(ohlcv,252))
print(Sf.max_dd(ohlcv,252))
print(Sf.sharpe(ohlcv,20,252))
print(Sf.sortino(ohlcv,30,252))

tickers = ["MMM","AXP","T","BA","CAT","CVX","CSCO","KO", "XOM","GE","GS","HD",
           "IBM","INTC","JNJ","JPM","MCD","MRK","MSFT","NKE","PFE","PG","TRV",
           "UTX","UNH","VZ","V","WMT","DIS"]


start = dt.datetime.today()-dt.timedelta(3650)
end = dt.datetime.today()

Sf.portfolio_rebalance(quantum_companys,start,end)
Sf.portfolio_rebalance(tickers,start,end)