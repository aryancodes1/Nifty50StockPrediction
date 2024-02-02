from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from EDA import X_in,X_out,Y_in,Y_out

model = Sequential()

model.add(LSTM(256,return_sequences = True,input_shape = (1,1,)))
model.add(LSTM(256,return_sequences = False))
model.add(Dense(1))


model.summary()
model.compile(loss = "mean_squared_error",optimizer = 'adam')
model.fit(X_in,X_out,epochs = 25,validation_data = (Y_in,Y_out),batch_size = 32)
model.save('model.h5')