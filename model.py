
import pickle
import urllib.request
from model import EnergyModel as Model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from datetime import date, datetime
from fbprophet import Prophet

class EnergyModel:

    def __init__(self):

        self.model = None

    def preprocess_training_data(self, df):
        
        X = pd.DatetimeIndex(df['day'])
        return X, df.consumption


     

    def fit(self, X, y):
        
        df = pd.DataFrame({'ds': X, 'y': y})

        self.model = Prophet(growth='linear',
                             holidays=bank_holidays,
                             weekly_seasonality=True,
                             yearly_seasonality=True,
                             seasonality_mode='additive')

        self.model.fit(df)

    def preprocess_unseen_data(self, df):

   
        X = pd.DatetimeIndex(df['day'])

        return X

    def predict(self, X):

        #raise NotImplementedError
        df_dates = self.model.make_future_dataframe(periods=X.shape[0],
                                                    include_history=False)
        predictions = self.model.predict(df_dates)

        return pd.Series(predictions.yhat.values, predictions.ds)
