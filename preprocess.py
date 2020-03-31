import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
plt.style.use('ggplot')

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Grabbing the most Recent Data from yahoo finance
url = "https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=1554046796&period2=1585669196&interval=1d&events=history"
r = requests.get(url, allow_redirects=True)
print("Importing Data from Yahoo Finance...")
open('btc_data.csv', 'wb').write(r.content)

# reading in CSV
data = pd.read_csv("btc_data.csv")


#Data preprocessing
print("Setting datetime index as date...")
data = data.set_index("Date")[['Close']].tail(1000)
data = data.set_index(pd.to_datetime(data.index))

#Normalizing/Scaling
scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
