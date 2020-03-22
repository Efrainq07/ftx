import schedule as scd
import time
import rest.client as cl
from secret.keys import api_key,api_secret
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates

class FTXTrader:
    def __init__(self,**kwargs):
        self.market = kwargs['market']
        self.api_key = kwargs['api_key']
        self.api_secret = kwargs['api_secret']
        self.lookback = kwargs['lookback']
        self.resolution = kwargs['resolution']
        self.client = cl.FtxClient(api_key = self.api_key, api_secret = self.api_secret)
        self.ax = kwargs['ax']
        scd.every(kwargs['dt']).seconds.do(self.trading) 

    def make_timeseries(self,data):
        df = {}
        for datum in data:
            for key in datum:
                if key in df:
                    df[key].append(datum[key])
                else:
                    df[key] = [datum[key]]
        df = pd.DataFrame(df)
        return df

    def trading(self):
        maxspan = 26
        historical = self.client.get_window(self.market,self.resolution,self.lookback)
        data = self.make_timeseries(historical)
        indicators = pd.DataFrame({'date':data['startTime']})
        for col in ['open','high','low','close']:
            indicators[col+'_EMA26'] = data[col].ewm(span=26).mean()
            indicators[col+'_EMA9'] = data[col].ewm(span=9).mean()
        print(indicators)

fig, ax = plt.subplots()
t = FTXTrader(dt=10,api_key=api_key,api_secret=api_secret,resolution=60,lookback='day',market='ETHBULL/USD',ax = ax)


while True: 
    scd.run_pending() 
    time.sleep(1) 