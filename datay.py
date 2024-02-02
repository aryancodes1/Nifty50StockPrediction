import yfinance as yf

ticker = yf.Ticker('^NSEI')
df = ticker.history(period='10y')
df = df[['Close']]

