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
        self.state = kwargs['state']
        self.demo = kwargs['demo']
        self.currency = {'volatile':self.market.split('/')[0],
                        'stable':self.market.split('/')[1]}

        self.minimum_roi = kwargs['minimum_roi']
        if(self.demo):
            self.market_price = 0
        self.last_buy_price = -1
        if('balance' in kwargs):
            self.balance = kwargs['balance']

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
        df['avg'] = (df['open']+df['close'])*0.5
        return df



    def trading(self):
        maxspan = 26
        historical = self.client.get_window(self.market,self.resolution,self.lookback)
        data = self.make_timeseries(historical)
        indicators = pd.DataFrame({'date':data['startTime']})
        for col in ['open','high','low','close','avg']:
            indicators[col+'_EMA26'] = data[col].ewm(span=26).mean()
            indicators[col+'_EMA9'] = data[col].ewm(span=9).mean()
        res = indicators[['date','avg_EMA26','avg_EMA9']]
        res['short'] = res['avg_EMA26']<=res['avg_EMA9']
        res['market'] = data['close']
        datum = res.iloc[-1]
        if(self.demo):
            self.market_price = datum['market']
            if(datum['short'] and self.state == 'stable'):
                self.buy_volatile(self.balance[self.currency['stable']])
                self.state = 'volatile'
            elif ((datum['market']-self.last_buy_price)/self.last_buy_price > self.minimum_roi ):
                self.sell_volatile(self.balance[self.currency['volatile']])
                self.state = 'stable'
            else:
                print('HODL',self.balance)

    def buy_volatile(self,stable_amount):
        self.balance[self.currency['volatile']] += stable_amount/self.market_price
        self.balance[self.currency['stable']] = 0
        print('Bought {} {} for {} {}'.format(stable_amount/self.market_price,self.currency['volatile'],stable_amount,self.currency['stable']))
        print(self.balance)
        self.last_buy_price = self.market_price

    def sell_volatile(self,volatile_amount):
        self.balance[self.currency['stable']] += volatile_amount*self.market_price
        self.balance[self.currency['volatile']] = 0
        print('Sold {} {} for {} {}'.format(volatile_amount,self.currency['volatile'],volatile_amount*self.market_price,self.currency['stable']))
        print(self.balance)

       

    def graph(self,dataframe):
        for col in dataframe.columns:
            if(col == 'date'):
                continue
            plt.plot(dataframe[col],label = col)
        plt.pause(0.01)

fig, ax = plt.subplots()

settings = {'dt':10,
'api_key':api_key,
'api_secret':api_secret,
'resolution':300,
'lookback':'day',
'market':'ETHBULL/USD',
'state' : 'stable',
'balance': {'USD':24.0,'ETHBULL':0},
'demo':True,
'minimum_roi':0.1}

t = FTXTrader(**settings)

while True: 
    scd.run_pending() 
    time.sleep(1) 