import matplotlib.pyplot as plt

#stocks is a dataframe with stocks values
def multistocks_show(stocks):
    # Handling NaN Values
    stocks.fillna(method='bfill', axis=0,inplace=True)  # Replaces NaN values with the next valid value along the column
    daily_return = stocks.pct_change()  # Creates dataframe with daily return for each stock

    # Data vizualization
    stocks.plot()  # Plot of all the stocks superimposed on the same chart

    cp_standardized = (stocks - stocks.mean()) / stocks.std()  # Standardization
    cp_standardized.plot()  # Plot of all the stocks standardized and superimposed on the same chart

    stocks.plot(subplots=True, layout=(3, 2), title="Tech Stock Price Evolution",
                      grid=True)  # Subplots of the stocks

    # Pyplot demo
    fig, ax = plt.subplots()
    plt.style.available
    plt.style.use('ggplot')
    ax.set(title="Daily return on tech stocks", xlabel="Tech Stocks", ylabel="Daily Returns")
    plt.bar(daily_return.columns, daily_return.mean())
    plt.show()

def MACD_visualization(df,ticker):
    plt.subplot(311)
    plt.plot(df.iloc[-100:, 4])
    plt.title(ticker+' Stock Price')
    plt.xticks([])

    plt.subplot(312)
    plt.bar(df.iloc[-100:, 5].index, df.iloc[-100:, 5].values)
    plt.title('Volume')
    plt.xticks([])

    plt.subplot(313)
    plt.plot(df.iloc[-100:, [-2, -1]])
    plt.title('MACD')
    plt.legend(('MACD', 'Signal'), loc='lower right')

    plt.show()
def ATR_visualization(df,ticker):
    plt.subplot(211)
    plt.plot(df['Open'])
    plt.title(ticker + ' Stock Price')
    plt.xticks([])

    plt.subplot(212)
    plt.plot(df['ATR'])
    plt.title('ATR')
    plt.legend(('ATR', 'Signal'), loc='lower right')
    plt.show()

def BollBnd_visualization(df,ticker):
    df['BB_up'].plot()
    df['BB_dn'].plot()
    df['Adj Close'].plot()
    plt.title('Bollinger band '+ticker + ' Stock Price')
    plt.xticks([])
    plt.legend(('UP BAND','DOWN BAND', 'Signal'), loc='lower right')
    plt.show()

def RSI_visualization(df,ticker):
    plt.subplot(211)
    plt.plot(df['Adj Close'])
    plt.title(ticker + ' Stock Price')
    plt.xticks([])

    plt.subplot(212)
    plt.plot(df['RSI'])
    plt.title('RSI')
    plt.legend(('RSI', 'Signal'), loc='lower right')
    plt.show()
def Slope_visualization(df,ticker):
    plt.subplot(211)
    plt.plot(df['Adj Close'])
    plt.title(ticker + ' Stock Price')
    plt.xticks([])

    plt.subplot(212)
    plt.plot(df['Slope Angle'])
    plt.title('Slope Angle')
    plt.legend(('Slope Angle', 'Signal'), loc='lower right')
    plt.show()