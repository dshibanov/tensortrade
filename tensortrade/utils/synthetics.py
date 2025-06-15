import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta, MO
import quantutils.constants as constants
import warnings

from pprint import pprint

# SYNTHETIC PROCESSES
SIN = 'SIN'
FLAT = 'FLAT'
VOLATILITY_K = 1
VOLUME_FROM_RANGE_K = 1

def make_ohlcv(df):

    opens = df['close'].shift(1)
    opens.iloc[0] = df['close'].iloc[0]
    df['open'] = opens

    def get_body(row):
        return row['close'] - row['open']

    def get_high(row):
        value = np.random.normal(loc=abs(row['body']), scale=1.0, size=None) * VOLATILITY_K
        body_high = row['close'] if row['close'] > row['open'] else row['open']
        return body_high + value if value > 0 else body_high

    def get_low(row):
        value = np.random.normal(loc=abs(row['body']), scale=1.0, size=None) * VOLATILITY_K
        body_low = row['close'] if row['close'] < row['open'] else row['open']
        return body_low + value if value < 0 else body_low

    def get_volume(row):
        value = np.random.normal(loc=abs(row['close'] - row['open'])) * VOLUME_FROM_RANGE_K
        return 0 if value < 0 else value

    bodies = df.apply(get_body, axis=1)
    df['body'] = bodies
    df['high'] = df.apply(get_high, axis=1)
    df['low'] = df.apply(get_low, axis=1)
    df['volume'] = df.apply(get_volume, axis=1)

    # TODO: remove 'body' from final df
    return df

def get_episodes_lengths(feed):
    lens = []
    steps_in_this_episode=0
    for i,row in feed.iterrows():
        steps_in_this_episode+=1
        if row.loc["end_of_episode"] == True:
            lens.append(steps_in_this_episode)
            steps_in_this_episode = 0
    return lens

def make_sin_feed(length=1000):
    x = np.arange(0, 2*np.pi, 2*np.pi / (length + 1))
    y = 50*np.sin(3*x) + 100
    xy = pd.DataFrame(data=np.transpose([y]), index=x)
    xy.columns=['close']
    xy.index.name = "datetime"
    return xy

def make_flat_feed(length=1000, price_value=100):
    x = np.arange(0, 2*np.pi, 2*np.pi / (length + 1))
    y = np.full(np.shape(x), float(price_value))
    xy = pd.DataFrame(data=np.transpose([y]), index=x)
    xy.columns=['close']
    xy.index.name = "datetime"
    return xy

def get_num_samples(_from, _to, timeframe):
    f = pd.to_datetime(_from)
    t = pd.to_datetime(_to)
    return int((t-f).total_seconds() / (60*constants.TIMEFRAMES[timeframe]))

def make_synthetic_symbol(config):

    if 'timeframes' not in config:
        warnings.warn("Timeframes not specified")
        timeframes = ['1h']
    else:
        timeframes = config['timeframes']

    timeframes = config.get('timeframes', ['1h'])
    config['from'] = config.get('from', '1.1.1970')

    if 'to' in config:
        config['to'] = config.get('to', '1.1.1970')
        config['num_of_samples'] = get_num_samples(config['from'], config['to'], timeframes[0])
    else:
        config['num_of_samples'] = config.get('num_of_samples', 101)


    config["process"] = config.get('process', SIN)
    config["code"] = config.get('code', 0)

    symbol = config
    end_of_episode = pd.Series(np.full(config["num_of_samples"]+1, False))
    print('end_of_episode ', end_of_episode)

    if config["process"] == SIN:
        feed = make_sin_feed(symbol["num_of_samples"])
    elif config["process"] == FLAT:
        feed = make_flat_feed(symbol["num_of_samples"])
    else:
        raise Exception("Wrong process name")

    min_timeframe = timeframes[0]
    feed.index = pd.date_range(start=config.get('from', '1/1/2018'), freq=f'{constants.TIMEFRAMES[min_timeframe]}min',
                               periods=len(feed.index))

    if symbol.get('ohlcv', False):
        print('ohlcv')
        feed = make_ohlcv(feed)

    feed[['close', 'open', 'low', 'high']] = feed[['close', 'open', 'low', 'high']] / config.get('y_scale', 10000)

    symbol['quotes'] = {}
    symbol['quotes'][f'{min_timeframe}'] = feed

    return symbol



def test_make_synthetic_symbol():

    s = make_synthetic_symbol({
         'name': 'AST0',
         'from': '2020-1-1',
         'to': '2020-1-2',
         'synthetic': True,
         'ohlcv': True,
         'code': 0,
         'timeframes': ['1h']
        })


    pprint(s)


if __name__ == "__main__":
    test_make_synthetic_symbol()
