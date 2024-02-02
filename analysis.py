from prediction import StockPrice
from datay import df

nifty = StockPrice(model_path='model1.h5',df = df)
nifty.show_accuracy()

