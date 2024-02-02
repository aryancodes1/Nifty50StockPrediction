import numpy as np
from keras.models import load_model
from EDA import scaler
import matplotlib.pyplot as plt
import tensorflow
class StockPrice:
    def __init__(self,model_path,df):
        self.model_path = model_path
        self.df = df
    def predict_next_day(self):
        df = np.array(self.df)
        model = load_model(self.model_path)
        model.compile(loss="mean_squared_error", optimizer= tensorflow.keras.optimizers.Adam(0.001))
        df = scaler.fit_transform(df.reshape(-1, 1))
        pred_data = df[-1]
        pred_data = np.array(pred_data)
        pred_data = pred_data.reshape(1, pred_data.shape[0], 1)
        price = model.predict(pred_data, verbose=0)
        price = scaler.inverse_transform(price)
        print("Predicted Closing Price For Tomorrow Is:", float(price))
        return float(price)
    def show_accuracy(self,graph = True):
        real_val = []
        self.df = np.array(self.df)
        self.df = self.df.reshape(-1,1)
        for i in range(1,len(self.df)):
            real_val.append(float(self.df[i]))     
        length = len(self.df)
        model = load_model(self.model_path)
        model.compile(loss="mean_squared_error", optimizer=tensorflow.keras.optimizers.Adam(0.001))
        predictions = []
        for i in range(0,length):
            df = np.array(self.df)
            df = scaler.fit_transform(df.reshape(-1, 1))
            pred_data = df[i]
            pred_data = np.array(pred_data)
            pred_data = pred_data.reshape(1, pred_data.shape[0], 1)
            price = model.predict(pred_data, verbose=0)
            price = scaler.inverse_transform(price)
            print(float(price))
            predictions.append(float(price))
        print("All Done")
        if graph == True:
            plt.title("PREDICTION VS REAL VALUES (1O YEARS)")
            plt.plot(real_val, label="Real Values")
            plt.plot(predictions, label="Predictions")
            plt.legend()
            plt.show()
        return [predictions,real_val]







