import pickle
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from datetime import date, datetime
from fbprophet import Prophet


HOLIDAYS = [
    '26-Dec-2011', '27-Dec-2011', '2-Jan-2012', '6-Apr-2012', '9-Apr-2012',
    '7-May-2012', '4-Jun-2012', '5-Jun-2012', '27-Aug-2012', '25-Dec-2012',
    '26-Dec-2012', '1-Jan-2013', '29-Mar-2013', '1-Apr-2013', '6-May-2013',
    '27-May-2013', '26-Aug-2013', '25-Dec-2013', '26-Dec-2013', '1-Jan-2014'
]

class EnergyModel:

    def __init__(self):

        self.model = None

    def preprocess_training_data(self, df):
        
        X = pd.DatetimeIndex(df['day'])
        return X, df.consumption


     

    def fit(self, X, y):
        
       bank_holidays = pd.DataFrame({
            'holiday': 'BankHoliday',
            'ds': pd.to_datetime(HOLIDAYS)
        })

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
