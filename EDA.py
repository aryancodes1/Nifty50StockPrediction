import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datay import df

scaler = MinMaxScaler(feature_range=(0,1))

def create_dataset(dataset):
	dataX, dataY = [],[]
	for i in range(0,len(dataset)-1):
		a = [dataset[i]]
		b = [dataset[i+1]]
		dataY.append(b)
		dataX.append(a)
	
	
	return dataX,dataY

df = np.array(df)
df = df.reshape(-1,1)
df = scaler.fit_transform(df)

length = int(len(df))
train_data, test_data = df[0:int(length*0.85)],df[int(length*0.85):length]

train_data = np.array(train_data)
test_data = np.array(test_data)


X_in, X_out = create_dataset(train_data)
Y_in, Y_out = create_dataset(test_data)


X_in = np.array(X_in)
Y_in = np.array(Y_in)
X_out = np.array(X_out)
Y_out = np.array(Y_out)

X_in = X_in.reshape(X_in.shape[0],1,X_in.shape[1])
Y_in = Y_in.reshape(Y_in.shape[0],1,Y_in.shape[1])
