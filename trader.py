import schedule as scd
import time
import rest.client as cl
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as dates
from datetime import datetime 


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
        self.window = kwargs['window']  # Window / lookback of data to consider
        # Granularity of data requested from ftx API
        self.resolution = kwargs['resolution']
        self.client = cl.FtxClient(
            api_key=self.api_key, api_secret=self.api_secret)  # FTX API Client
        self.state = kwargs['state']  # Initial coin state
        if('demo' in kwargs):
            self.demo = kwargs['demo']  # Boolean to run off the market
        else:
            self.demo = False
        self.currency = {'volatile': self.market.split('/')[0],
                         'stable': self.market.split('/')[1]}  # Dictionary of coin pairs
        self.minimum_roi = kwargs['minimum_roi']  # Minimum roi threshold

        # Initialize graph 
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.x, self.avg_EMA26, self.avg_EMA9, self.close = [], [], [], []
 

        if(self.demo):
            self.market_price = 0

        if('last_buy_price' in kwargs):
            # Last price at which a coin was bought
            self.last_buy_price = kwargs['last_buy_price']
        else:
            self.last_buy_price = 0

        if('balance' in kwargs):
            # Initial value for the account balance of the coin pair
            self.balance = kwargs['balance']

        # Scheduler runs the trading function every dt seconds
        scd.every(kwargs['dt']).seconds.do(self.trading)

    def make_timeseries_df(self, data):
        """
        Generates dataframe from data with added mean column 
        """

        df = pd.DataFrame(data)
        df['avg'] = df[['open', 'close']].mean(axis=1, skipna=True)

        return df

    def update_graph(self, last_row):
        ## Update graph
        if (self.x == []):
            self.x.append(1)
        else: 
            self.x.append(self.x[-1] + 1)

        self.avg_EMA26.append(last_row.iloc[1])
        self.avg_EMA9.append(last_row.iloc[2])
        self.close.append(last_row.iloc[4])

        # Update line graph
        self.line1, = self.ax.plot(self.x, self.avg_EMA26, 'r-', label = "avg_EMA26")
        self.line2, = self.ax.plot(self.x, self.avg_EMA9, 'b-', label = "avg_EMA9")
        self.line2, = self.ax.plot(self.x, self.close, 'g-', label = "close")

        if (self.x == [1]):
            self.ax.legend()
        else:
            pass

        # Redraw figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def trading(self):
        """
        Trading functions. 
        Calculates EMAs, checks for crosses, predicts a state and buys or sells 
        """

        # Gets historical data from a certain time window and resolution
        historical = self.client.get_window(self.market, self.resolution, self.window)

        # Gets data into dataframe from the windowed historical 
        data = self.make_timeseries_df(historical)
        
        ### Sets indicators dataframe
        indicators = pd.DataFrame({'date': data['startTime']})
        
        # Calculates EMAs and adds them as indicators 
        for col in ['open', 'high', 'low', 'close', 'avg']:
            indicators[col + '_EMA26'] = data[col].ewm(span=26).mean()
            indicators[col + '_EMA9'] = data[col].ewm(span=9).mean()

        # Creates results DataFrame with the EMA indicators 
        results = indicators[['date', 'avg_EMA26', 'avg_EMA9']]
        results['short'] = results['avg_EMA26'] <= results['avg_EMA9'] # Determines times when EMAs cross
        results['market'] = data['close']
        results['separate'] = np.abs(results['avg_EMA9'] - results['avg_EMA26']) > 0.65 # Determines times with differences greater than 0.65 units 

        # Gets the last row of the results
        last_row = results.iloc[-1]

        self.update_graph(last_row)


        if(self.demo):
            self.market_price = last_row['market']
            print(f'''{datetime.datetime.today()}''')
            print('State: {}'.format(self.state))
            print('Balance: ', self.get_balance())

            if(self.state == 'volatile'):
                roi = (self.market_price-self.last_buy_price) / \
                    self.last_buy_price

                if(roi > self.minimum_roi):
                    self.sell_volatile(self.balance[self.currency['volatile']])
                    self.state = 'stable HODL'

                elif ((not last_row['short']) and last_row['separate']):

                    if(roi < -3*self.minimum_roi):
                        self.sell_volatile(
                            self.balance[self.currency['volatile']])
                        self.state = 'stable'

            elif self.state == 'stable HODL':

                if ((not last_row['short']) and last_row['separate']):
                    self.state = 'stable'

            elif self.state == 'stable':

                if(last_row['short'] and last_row['separate']):
                    self.buy_volatile(self.balance[self.currency['stable']])
                    self.state = 'volatile'

    def get_balance(self):
        balances = client.get_balances()
        balances = pd.DataFrame(balances)
        bal_volatile = balances[balances['coin']==self.currency['volatile']]['free']
        bal_stable = balances[balances['coin']==self.currency['stable']]['free']
        return {self.currency['volatile']:bal_volatile,self.currency['stable']:bal_stable}

    def buy_volatile(self, stable_amount):
        """
        This function buys the volatile coin and sells all the stable one
        """
        if self.demo:
            buy_intend = stable_amount/self.market_price
            self.balance[self.currency['volatile']] += buy_intend
            self.balance[self.currency['stable']] = 0
            buy_price = buy_intend/stable_amount
        else:
            buy_intend = stable_amount/self.market_price
            order_start = datetime.datetime.today().timestamp()
            order_data = self.client.place_order(self.market,'buy',None,buy_intend,'market')
            time.sleep(0.2)
            history_data = self.client.get_order_history(self.market,'buy','market',order_start)
            order_status = next(order for order in history_data if order['id'] == order_data['id'])
            buy_price = order_status['avgFillPrice']


        print(f'Bought {buy_intend} {self.currency['volatile']} for {buy_price*buy_intend} {self.currency['stable']}')
        print(f'({buy_price} {self.market})')
        print(self.balance)
        print('')
        self.last_buy_price = buy_price

    def sell_volatile(self, volatile_amount):
        """
        This function sells the volatile coin and buys all the stable one
        """
        if(self.demo):
            self.balance[self.currency['stable']] += volatile_amount*self.market_price
            self.balance[self.currency['volatile']] = 0
        else:
            sell_intend = volatile_amount
            order_start = datetime.datetime.today.timestamp()
            order_data = self.client.place_order(self.market,'sell',None,sell_intend,'market')
            time.sleep(0.2)
            history_data = self.client.get_order_history(self.market,'sell','market',order_start)
            order_status = next(order for order in history_data if order['id'] == order_data['id'])
            sell_price = order_status['avgFillPrice']

        print(f'Sold {sell_intend} {self.currency['volatile']} for {sell_price*sell_intend} {self.currency['stable']}')
        print(f'({sell_price} {self.market})')
        print(self.balance)
        print('')



from settings import settings_dict

t = FTXTrader(**settings_dict)

while True:
    scd.run_pending()
    time.sleep(1)
