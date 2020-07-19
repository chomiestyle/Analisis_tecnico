# =============================================================================
# Measuring the performance of a buy and hold strategy - CAGR
# Author : Mayank Rasu (http://rasuquant.com/wp/)

# Please report bug/issues in the Q&A section
# =============================================================================

# Import necesary libraries
import numpy as np
import yfinance as yf
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import copy
import matplotlib.pyplot as plt


#daily data
def CAGR(DF,n_d):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    if n_d==12:
        ret='mon_ret'
    elif n_d==252:
        ret='daily_ret'
        df[ret] = DF["Adj Close"].pct_change()

    df["cum_return"] = (1 + df[ret]).cumprod()
    n = len(df)/n_d
    #CAGR = (df["cum_return"][-1])**(1/n) - 1
    CAGR = (df["cum_return"].tolist()[-1]) ** (1 / n) - 1
    return CAGR

#Volatility of a strategy is represented by the standard deviation of the returns
#This capture the variability of returns of the mean return
#Annualization is achived by multiplying volatility by square root of the annualization factor
#widely used measure of risk. However this aproach asumes normal distributions of returns wich is not true
#Does not capture tail risk

def volatility(DF,n_d):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    if n_d==12:
        ret='mon_ret'
    elif n_d==252:
        ret='daily_ret'
        df[ret] = DF["Adj Close"].pct_change()
    vol = df[ret].std() * np.sqrt(n_d)
    return vol


def sharpe(DF, rf,n_d):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df,n_d) - rf) / volatility(df,n_d)
    return sr


def sortino(DF, rf,n_d):
    "function to calculate sortino ratio ; rf is the risk free rate"
    if n_d==12:
        ret='mon_ret'
    elif n_d==252:
        ret='daily_ret'
    df = DF.copy()
    df[ret] = DF["Adj Close"].pct_change()
    df["neg_ret"] = np.where(df[ret] < 0, df[ret], 0)
    neg_vol = df["neg_ret"].std() * np.sqrt(252)
    sr = (CAGR(df,n_d) - rf) / neg_vol
    return sr


def max_dd(DF,n_d):
    "function to calculate max drawdown"
    df = DF.copy()
    if n_d==12:
        ret='mon_ret'
    elif n_d==252:
        ret='daily_ret'
        df[ret] = DF["Adj Close"].pct_change()

    df["cum_return"] = (1 + df[ret]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"] / df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd


def calmar(DF,n_d):
    "function to calculate calmar ratio"
    df = DF.copy()
    clmr = CAGR(df,n_d) / max_dd(df,n_d)
    return clmr
###monthly portfolio rebalancing
#Choose any universe of stocks*Large cap, mind cap, small cap, industry specific, factor specific, etc) and stick to this grou of stocks as the source fo your portfolio
#for the entire duration of backtesting
#Build fixed of individual position sized long only portfolio by picking m number of stocks based on monthly returns (or any other suitable criterio)(any othe momentum idicator).
#Rebalance the portfolio every month by removing worse x stocks and replacing them with top  x stocks from the universe of stocks (can existing stock be picked again?)
#Then backstest the strategy and compare the KPIs with that of simple buy and hold strategy of corresponding index
def portfolio_rebalance(portfolio,start,end):
    ohlc_mon = {}  # directory with ohlc value for each stock

    # looping over tickers and creating a dataframe with close prices
    for ticker in portfolio:
        ohlc_mon[ticker] = yf.download(ticker, start, end, interval='1mo')
        ohlc_mon[ticker].dropna(inplace=True, how="all")

    tickers = ohlc_mon.keys()
    ################################Backtesting####################################

    # calculating monthly return for each stock and consolidating return info by stock in a separate dataframe
    ohlc_dict = copy.deepcopy(ohlc_mon)
    return_df = pd.DataFrame()
    for ticker in tickers:
        print("calculating monthly return for ", ticker)
        ohlc_dict[ticker]["mon_ret"] = ohlc_dict[ticker]["Adj Close"].pct_change()
        return_df[ticker] = ohlc_dict[ticker]["mon_ret"]
    # calculating overall strategy's KPIs
    CAGR(pflio(return_df, 6, 3),12)
    sharpe(pflio(return_df, 6, 3), 0.025,12)
    max_dd(pflio(return_df, 6, 3),12)

    # calculating KPIs for Index buy and hold strategy over the same period
    DJI = yf.download("^DJI", dt.date.today() - dt.timedelta(1900), dt.date.today(), interval='1mo')
    DJI["mon_ret"] = DJI["Adj Close"].pct_change()
    CAGR(DJI,12)
    sharpe(DJI, 0.025,12)
    max_dd(DJI,12)
    # visualization
    fig, ax = plt.subplots()
    plt.plot((1 + pflio(return_df, 6, 3)).cumprod())
    plt.plot((1 + DJI["mon_ret"][2:].reset_index(drop=True)).cumprod())
    plt.title("Index Return vs Strategy Return")
    plt.ylabel("cumulative return")
    plt.xlabel("months")
    ax.legend(["Strategy Return", "Index Return"])
    plt.show()

# function to calculate portfolio return iteratively
def pflio(DF,m,x):
    """Returns cumulative portfolio return
    DF = dataframe with monthly return info for all stocks
    m = number of stock in the portfolio
    x = number of underperforming stocks to be removed from portfolio monthly"""
    df = DF.copy()
    portfolio = []
    monthly_ret = [0]
    for i in range(1,len(df)):
        if len(portfolio) > 0:
            monthly_ret.append(df[portfolio].iloc[i,:].mean())
            bad_stocks = df[portfolio].iloc[i,:].sort_values(ascending=True)[:x].index.values.tolist()
            portfolio = [t for t in portfolio if t not in bad_stocks]
        fill = m - len(portfolio)
        new_picks = df.iloc[i,:].sort_values(ascending=False)[:fill].index.values.tolist()
        portfolio = portfolio + new_picks
        print(portfolio)
    monthly_ret_df = pd.DataFrame(np.array(monthly_ret),columns=["mon_ret"])
    return monthly_ret_df