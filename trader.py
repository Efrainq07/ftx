import schedule as scd
import time
import rest.client as cl
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as dates
import mpl_finance as mpf
import datetime
import matplotlib.dates as dates
import matplotlib.cm as cm


class FTXTrader:
    """
    FTX Trader class. 

    Stores the trading variables and parameters
    """

    # Initialize variables
    def __init__(self, **kwargs):
        self.market = kwargs['market']  # Coin pair to trade
        self.api_key = kwargs['api_key']  # API key for ftx
        self.api_secret = kwargs['api_secret']  # API secret for ftx
        if('subaccount' in kwargs):
            self.subaccount = kwargs['subaccount']
        else:
            self.subaccount = None
        self.window = kwargs['window']  # Window / lookback of data to consider
        # Granularity of data requested from ftx API
        self.resolution = kwargs['resolution']
        self.client = cl.FtxClient(
            api_key=self.api_key, api_secret=self.api_secret,subaccount_name=self.subaccount)  # FTX API Client
        self.state = kwargs['state']  # Initial coin state
        if('demo' in kwargs):
            self.demo = kwargs['demo']  # Boolean to run off the market
        else:
            self.demo = False
        self.currency = {'volatile': self.market.split('/')[0],
                         'stable': self.market.split('/')[1]}  # Dictionary of coin pairs
        self.maximum_loss = kwargs['maximum_loss']  # Minimum roi threshold
        
        # Initialize graph 
        self.last_datum = None
        plt.style.use('dark_background')
        plt.ion()
        self.line_dict = {}
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(211)
        self.axslope = self.fig.add_subplot(212)


        initial_data =self.client.get_window(self.market,self.resolution,'quarterday')
        initial_data = self.make_timeseries_df(initial_data)
        initial_indicators = self.make_indicators_df(initial_data)
        initial_indicators['index'] = list(range(0,len(initial_indicators)))
        self.x, self.avg_EMA26, self.avg_EMA9, self.ohlc_list = initial_indicators['index'].tolist(), initial_indicators['avg_EMA26'].tolist(),\
             initial_indicators['avg_EMA9'].tolist(), [[date,o,h,l,c] for date,o,h,l,c in initial_indicators[['index','open','high','low','close']].values]
        self.graph_dict= {
            'x':list(range(0,len(initial_indicators))),
            'date':initial_indicators['date'].tolist(),
            'avg_EMA26':initial_indicators['avg_EMA26'].tolist(),
            'avg_EMA9': initial_indicators['avg_EMA9'].tolist(),
            'avg_EMA20':initial_indicators['avg_EMA20'].tolist(),
            'avg_EMA20_slope':initial_indicators['avg_EMA20_slope'].tolist(),
            'avg_20_bollinger_bottom':initial_indicators['avg_20_bollinger_bottom'].tolist(),
            'avg_20_bollinger':initial_indicators['avg_20_bollinger'].tolist(),
            'avg_20_bollinger_slope':initial_indicators['avg_20_bollinger_slope'].tolist(),
            'ohlc_list':[[date,o,h,l,c] for date,o,h,l,c in initial_indicators[['index','open','high','low','close']].values]
        }
        self.update_graph()
 

        if(self.demo):
            self.market_price = 0

        if('last_buy_price' in kwargs):
            # Last price at which a coin was bought
            self.last_buy_price = kwargs['last_buy_price']
        else:
            self.last_buy_price = -1
        if(self.demo):
            if('balance' in kwargs):
                # Initial value for the account balance of the coin pair
                self.balance = kwargs['balance']
            else:
                self.demo = False
                self.balance = self.get_balance()
                self.demo = True

        # Scheduler runs the trading function every dt seconds
        scd.every(kwargs['dt']).seconds.do(self.trading)
        scd.every(kwargs['dt']).seconds.do(self.update_graph)

    def make_timeseries_df(self, data):
        """
        Generates dataframe from data with added mean column 
        """

        df = pd.DataFrame(data)
        df['avg'] = df[['open', 'close']].mean(axis=1, skipna=True)

        return df

    def update_graph(self):
        last_row = self.last_datum
        ## Update graph
        self.ax.cla()
        self.axslope.cla()
        if(last_row is not None):
            if(last_row['date'] == self.graph_dict['date'][-1]):
                self.graph_dict['ohlc_list'][-1] = [self.graph_dict['ohlc_list'][-1][0],last_row['open'],last_row['high'],last_row['low'],last_row['close']]
                for key in self.graph_dict:
                    if(key in ['x','ohlc_list','date']):
                        continue
                    self.graph_dict[key][-1] = last_row[key]
            else:
                self.graph_dict['x'].pop(0)
                self.graph_dict['ohlc_list'].pop(0)

                self.graph_dict['x'].append(self.graph_dict['x'][-1]+1)
                self.graph_dict['ohlc_list'].append([self.graph_dict['ohlc_list'][-1][0]+1,last_row['open'],last_row['high'],last_row['low'],last_row['close']])
                for key in self.graph_dict:
                    if(key in ['x','ohlc_list']):
                        continue
                    self.graph_dict[key].append(last_row[key])
                    self.graph_dict[key].pop(0)
            
        # Update line graph
        colors = cm.rainbow(np.linspace(0,1,len(self.graph_dict)))
        self.line_dict['ohlc_list'] = mpf.candlestick_ohlc(self.ax, self.graph_dict['ohlc_list'], width=0.4, colorup='#77d879', colordown='#db3f3f')
        for color,key in zip(colors,self.graph_dict):
            if(key in ['x','ohlc_list','date']):
                continue
            if key.split('_')[-1]=='slope':
                self.line_dict[key] = self.axslope.plot(self.graph_dict['x'], self.graph_dict[key], color = color[:-1],label = key) 
            elif key.split('_')[-1]=='bollinger':
                self.line_dict[key] =  self.ax.fill_between(self.graph_dict['x'],self.graph_dict[key+'_bottom'],self.graph_dict[key],color = 'grey') 
            elif key.split('_')[-1]=='bottom':
                continue  
            else:
                self.line_dict[key] = self.ax.plot(self.graph_dict['x'], self.graph_dict[key], color = color[:-1],label = key)

    
        
        self.ax.legend()
        self.ax.set_xlabel('Últimas 6 horas')
        self.ax.set_ylabel(self.market)

        self.axslope.legend()
        self.axslope.set_ylabel('Slope')
        self.axslope.set_xlabel('Últimas 6 horas')

        # Redraw figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def make_indicators_df(self,data):
        ### Sets indicators dataframe
        indicators = pd.DataFrame({'date': pd.to_datetime(data['startTime'])})
        # Calculates EMAs and adds them as indicators 
        for col in ['open', 'high', 'low', 'close', 'avg']:
            indicators[col] = data[col]
            indicators[col + '_EMA26'] = data[col].ewm(span=26).mean()
            indicators[col + '_EMA20'] = data[col].ewm(span=20).mean()
            indicators[col + '_EMA9'] = data[col].ewm(span=9).mean()
            indicators[col+'_std20'] = data[col].rolling(window=20).std() 
            indicators[col+'_20_bollinger'] = indicators[col+'_EMA20']+2*indicators[col+'_std20']
            indicators[col+'_20_bollinger_bottom'] = indicators[col+'_EMA20']-2*indicators[col+'_std20']
            indicators[col+'_EMA20_slope'] = indicators[col+'_EMA20'].diff()
            indicators[col+'_20_bollinger_slope'] = indicators[col+'_20_bollinger'].diff()
        
        indicators['market'] = indicators['close']
        return indicators


    def trading(self):
        """
        Trading functions. 
        Calculates EMAs, checks for crosses, predicts a state and buys or sells 
        """
        try:
            # Gets historical data from a certain time window and resolution
            historical = self.client.get_window(self.market, self.resolution, self.window)

            # Gets data into dataframe from the windowed historical 
            data = self.make_timeseries_df(historical)
            
            ### Sets indicators dataframe
            results = self.make_indicators_df(data)

            # Creates results DataFrame with the EMA indicators 
            results['short'] = results['avg_EMA26'] <= results['avg_EMA9'] # Determines times when EMAs cross
            results['separate'] = np.abs(results['avg_EMA9'] - results['avg_EMA26']) > 0.5 # Determines times with differences greater than 0.65 units 

            # Gets the last row of the results
            last_row = results.iloc[-1]
            self.last_datum = last_row


            self.market_price = last_row['market']
            roi = (self.market_price-self.last_buy_price) / self.last_buy_price
            zscore = (self.market_price - last_row['avg_EMA20'])/last_row['avg_std20']

            print(f'''{datetime.datetime.today()}''')
            if(self.demo):
                print('##DEMO MODE##')
            print('State: {}'.format(self.state))
            print('Balance: ', self.get_balance())
            print(f'Zscore: {zscore}')
            print('')
            

            if(self.state == 'volatile'):
                if(roi > 0 and zscore>1.2):
                    self.sell_volatile(self.balance[self.currency['volatile']])
                    self.state = 'stable HODL'

                elif ((not last_row['short']) and last_row['separate']):

                    if(roi < -self.maximum_loss and zscore>-1.2):
                        self.sell_volatile(
                            self.balance[self.currency['volatile']])
                        self.state = 'stable'

            elif self.state == 'stable HODL':

                if ((not last_row['short']) and last_row['separate']):
                    self.state = 'stable'

            elif self.state == 'stable':

                if((last_row['short'] and last_row['separate'] and zscore < 0.5) or zscore < -1.7 ):
                    self.buy_volatile(self.balance[self.currency['stable']])
                    self.state = 'volatile'
        except Exception as e:
            print('ERROR:',e)

    def get_balance(self):
        if(self.demo):
            return self.balance
        else:
            balances = self.client.get_balances()
            balances = pd.DataFrame(balances)
            bal_volatile = balances[balances['coin']==self.currency['volatile']]
            bal_stable = balances[balances['coin']==self.currency['stable']]
            if(len(bal_volatile)!=0):
                bal_volatile = bal_volatile.loc[0]['free']
            else:
                bal_volatile = 0
            if(len(bal_stable)!=0):
                bal_stable = bal_stable.loc[0]['free']
            else:
                bal_stable = 0
        return {self.currency['volatile']:bal_volatile,self.currency['stable']:bal_stable}

    def buy_volatile(self, stable_amount):
        """
        This function buys the volatile coin and sells all the stable one
        """
        if self.demo:
            buy_intend = stable_amount/self.market_price
            self.get_balance()[self.currency['volatile']] += buy_intend
            self.get_balance()[self.currency['stable']] = 0
            buy_price = stable_amount/buy_intend
        else:
            buy_intend = stable_amount/self.market_price

            order_start = datetime.datetime.today().timestamp()
            order_data = self.client.place_order(self.market,'buy',None,buy_intend,'market')
            time.sleep(0.2)
            history_data = self.client.get_order_history(self.market,'buy','market',order_start)
            order_status = next(order for order in history_data if order['id'] == order_data['id'])

            buy_price = order_status['avgFillPrice']


        print(f'Bought {buy_intend} {self.currency["volatile"]} for {buy_price*buy_intend} {self.currency["stable"]}')
        print(f'({buy_price} {self.market})')
        print(self.get_balance())
        print('')
        self.last_buy_price = buy_price

    def sell_volatile(self, volatile_amount):
        """
        This function sells the volatile coin and buys all the stable one
        """
        if(self.demo):
            sell_intend = volatile_amount
            self.balance[self.currency['stable']] += volatile_amount*self.market_price
            self.balance[self.currency['volatile']] = 0
            sell_price = self.market_price
        else:
            sell_intend = volatile_amount
            order_start = datetime.datetime.today.timestamp()
            order_data = self.client.place_order(self.market,'sell',None,sell_intend,'market')
            time.sleep(0.2)
            history_data = self.client.get_order_history(self.market,'sell','market',order_start)
            order_status = next(order for order in history_data if order['id'] == order_data['id'])
            sell_price = order_status['avgFillPrice']

        print(f'Sold {sell_intend} {self.currency["volatile"]} for {sell_price*sell_intend} {self.currency["stable"]}')
        print(f'({sell_price} {self.market})')
        print(self.balance)
        print('')



from settings import settings_dict

t = FTXTrader(**settings_dict)

while True:
    scd.run_pending()
    plt.pause(1)
