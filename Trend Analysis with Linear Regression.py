import numpy as np
from sklearn.linear_model import LinearRegression
from binance.client import Client
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

binance_api_key = ''
binance_api_secret = ''
client = Client(binance_api_key, binance_api_secret)

symbol = 'BTCUSDT'
#n is the length of data
n = 24

#DATA FETCH
marketData = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR)
dfMarketData = pd.DataFrame(marketData)

#DATA PREPARE
closePrice = dfMarketData[4].tail(n)
closePrice = closePrice.to_numpy()
time = dfMarketData[0].tail(n)
time = time.to_numpy()

t = []
for i in range(n):
    ts = int(time[i])
    ts = int(str(ts)[:10])
    ts = int(ts)
    time2 = datetime.fromtimestamp(ts).strftime('%d/%m/%Y %H:%M:%S')
    t.append(time2)

t = np.array(t)

x = [list(range(n))]
x = np.array(x)
x = x.reshape((-1, 1))

y = np.array([x.split() for x in closePrice], dtype=float)

#MODEL
model = LinearRegression()
model.fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('slope:', model.coef_)
y_pred = model.predict(x)
plt.scatter(t, y)
plt.plot(t, y_pred, color='red')
plt.xticks(rotation=12)
plt.xticks(fontsize=6)
plt.show()
