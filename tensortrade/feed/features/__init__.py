import abc, os, time, ray
import sys
sys.path.append('../')
sys.path.append('../../')
import talib
import talib as ta
from talib import stream
from talib import abstract
from talib.abstract import *

from ray.actor import ActorHandle
from ray import tune
import pandas as pd
import numpy as np
import collections
from pprint import pprint
import timeit
from collections import OrderedDict


# from quantlib.utils import *
import math
import pandas as pd
import numpy as np
import collections


def call(name, params, seria):
    f = get_func(name)
    return f(seria, **params)

def get_func(name):
    if name == 'bbwp_caretaker':
        func = bbwp_caretaker
        params = collections.OrderedDict([('lookback', 252),('bbands_len', 13)])
    elif name == 'bbwp':
        func = bbwp
        params = collections.OrderedDict([('lookback', 252),('bbands_len', 13)])
    else:
        func = abstract.Function(name)
        params = func.parameters
    # TODO: return func only
    # fix it later
    # return (func, params)
    return func





# It shows the percentage of bars over a specified lookback period that the Bollinger Band Width was less than the current Bollinger Band Width. 
def bbwp(seria, lookback=252, bbands_len=13):
    if len(seria) < lookback:
        values = np.empty((1,len(seria.index)))[0]
        values[:] = np.nan
        return pd.Series(data=values, index=seria.index, name='bbwp')

    upperband, middleband, lowerband = call('bbands', dict([('timeperiod', bbands_len), ('nbdevup', 2.0), ('nbdevdn', 2.0), ('matype', 0)]), seria)
    width = pd.Series(data=(upperband - lowerband), index=seria.index, name='bbwp2')

    # rewind Nan's
    i = 0
    while i < len(width) and math.isnan(width.iloc[i]) == True:
        i+=1
    values = np.empty((1,lookback + i))[0]
    values[:] = np.nan
    values = values.tolist()
    j = len(values)
    while j < len(width):
        lback = width.iloc[j-lookback:j]
        less = lback < width.iloc[j]
        less_ratio = less.values.tolist().count(True)/lookback
        values.append(less_ratio*100)
        j+=1

    return pd.Series(data=values, index=seria.index, name='bbwp')


# Caretaker version, by default doesn't use multyplier for sdtdev in width
# calculation. Also uses width normalizing (norm_width flag)
def bbwp_caretaker(seria, lookback=252, bbands_len=13):
    norm_width = True
    upperband, middleband, lowerband = call('bbands', dict([('timeperiod', bbands_len), ('nbdevup', 1.0), ('nbdevdn', 1.0), ('matype', 0)]), seria)

    # TODO: look at compare_features_time_calc() in features.py
    # looks like bbwp is about 10 times slower than bbands
    # some maybe next section of code is possible to speedup somehow

    width = pd.Series(data=(upperband - lowerband), index=seria.index, name='bbwp_caretaker')
    if norm_width == True:
        width = width / middleband

    # rewind Nan's in width
    i = 0
    while i < len(width) and math.isnan(width.iloc[i]) == True:
        i+=1

    values = np.empty((1,lookback + i))[0]
    values[:] = np.nan
    j = len(values)
    while j < len(width):
        lback = width.iloc[j-lookback:j]
        less = lback < width.iloc[j]
        less_ratio = less.values.tolist().count(True)/lookback
        values = np.append(values, [less_ratio*100])
        j+=1
    return values



# get headers for indicators from here
# https://ta-lib.github.io/ta-lib-python/func_groups/overlap_studies.html

class Parameter:
    def __init__(self, name, value=None, search_space=None):
        # TODO: specify type for search space
        self.name = name
        if value == None:
            self.value = search_space.sample()
        else:
            self.value = value
        self.search_space = search_space

class Feature(abc.ABC):
    def __init__(self, name, params):
        self.name = name
        # self.func = indicators.get_func(name)
        self.func = get_func(name)
        self.func.parameters = params
        self.params = params
        self.depth = self.get_depth()

    def __reduce__(self):
        globals_dict = globals()  # or use a specific module's namespace
        deserializer = globals_dict[self.__class__.__name__]
        serialized_data = (self.params, ) # self.config ? 
        return deserializer, serialized_data

    def get_function_params(self):
        return self.func.info['parameters']

    def get_params_dict(self):
        params = {}
        paramslist=[]
        for i in self.params:
            params[i.name] = i.value
            paramslist.append((i.name, i.value))
        return collections.OrderedDict(paramslist)

    @abc.abstractmethod
    # TODO: make test for this
    def calc(self, seria): #, params = self.params):
        raise NotImplementedError

    def get_header(self):
        suffix=''
        for k,v in self.func.parameters.items():
            suffix+=f'{v}_'

        items = self.func.parameters.items()
        suffix = ''.join(f'_{str(x)}' for i,x in self.func.parameters.items())
        header=[f'{self.func.info["name"]}_{n}{suffix}' for n in self.func.output_names.copy()]
        return header

    def self(self, param_name, param_val=None):
        if param_val is None:
            return getattr(self, param_name)
        else:
            setattr(self, param_name, param_val)

    def get_depth(self, start_length = 100, max_length = 1000):
        length = start_length
        result_len = 10
        cleaned_len = 10
        while length < max_length and cleaned_len == result_len:
            inputs = {
                'open': np.random.random(100),
                'high': np.random.random(100),
                'low': np.random.random(100),
                'close': np.random.random(100),
                'volume': np.random.random(100)
            }

            df = pd.DataFrame(inputs)
            result = self.calc2(df)
            raw_len = len(result)
            result = result.dropna(how='any')
            cleaned_len = len(result)
            self.depth = raw_len - cleaned_len + 1
            if cleaned_len == 0:
                start_length *= 10

        return self.depth

    def latest(self, data):
        if type(data) == pd.DataFrame:
            sub_df = data.iloc[:self.depth]
        else:
            sub_df = data[:self.depth]

        return self.func(sub_df)

class Indicator(Feature):
    def __init__(self, name, params):
        super().__init__(name=name, params=params)

    # TODO: rename to calc_last_one
    def calc(self, seria):
        subseria = seria[-self.depth:]
        return self.func(subseria)

    def calc2(self, data):
        return self.func(data)

    def calc_all(self, seria):
        params = self.get_params_dict()
        return self.func(seria, **params)

    # calc_remote
    def calcr(self, seria, pba: ActorHandle):
        pba.update.remote(1)
        return self.calc(seria)

    # calc vertical
    def calcv(self, quotes, feed0, timeframe, pba: ActorHandle):
        state=[]
        t = timeframe
        counter=quotes[timeframe].index.get_loc(feed0.index[0])
        seria = quotes[t].iloc[:counter]
        for i,row in enumerate(feed0.itertuples(), 0):
            current_time = getattr(row,'index')
            if feed0[f'{t}_volume'].iloc[i] == 0: # 0 volume on bar opened tick 
                counter += 1
                seria = quotes[t].iloc[:counter].copy()

            lb = feed0[[f'{t}_open', f'{t}_high',f'{t}_low',f'{t}_close',f'{t}_volume']].iloc[i].to_frame().t
            lb.rename(columns={f'{t}_open':'open', f'{t}_high':'high',f'{t}_low':'low',f'{t}_close':'close',f'{t}_volume':'volume'}, inplace=true)

            seria.iloc[-1] = lb.iloc[-1]
            value = self.calc(seria['close'])
            s=[]
            if len(np.shape(value)) > 1:
                state.append([i[-1] for i in value])
            else:
                state.append(value[-1])

        pba.update.remote(1)
        return state



class BBWP_Caretaker(Indicator):
    def __init__(self, params):
        super().__init__(name='bbwp_caretaker', params=params)
        self.header = ['bbwp_caretaker']


@ray.remote
class BBWP_Caretaker_remote(BBWP_Caretaker):
    def __init__(self, params):
        super().__init__(params=params)


class BBANDS(Indicator):
    def __init__(self, params):
        super().__init__(name='bbands', params=params)
        self.header = ['bbands_upperband', 'bbands_middleband', 'bbands_lowerband']

@ray.remote
class BBANDS_remote(BBANDS):
    def __init__(self, params):
        super().__init__(params=params)


class SMA(Indicator):
    def __init__(self, params):
        super().__init__(name='sma', params=params)
        self.header = ['sma']

@ray.remote
class SMA_remote(SMA):
    def __init__(self, params):
        super().__init__(params=params)


class MACD(Indicator):
    def __init__(self, params):
        super().__init__(name='macd', params=params)
        self.header = ['sma']

@ray.remote
class MACD_remote(MACD):
    def __init__(self, params):
        super().__init__(params=params)

class Stochastic(Indicator):
    def __init__(self, params):
        super().__init__(name='stoch', params=params)
        self.header = ['sma']

@ray.remote
class Stochastic_remote(Stochastic):
    def __init__(self, params):
        super().__init__(params=params)

class RSI(Indicator):
    def __init__(self, params):
        super().__init__(name='rsi', params=params)
        self.header = ['sma']

@ray.remote
class RSI_remote(RSI):
    def __init__(self, params):
        super().__init__(params=params)

def will_frac(df: pd.DataFrame, period: int = 2) -> tuple[pd.Series, pd.Series]:
    """Indicate bearish and bullish fractal patterns using shifted Series.

    :param df: OHLC data
    :param period: number of lower (or higher) points on each side of a high (or low)
    :return: tuple of boolean Series (bearish, bullish) where True marks a fractal pattern
    """

    periods = [p for p in range(-period, period + 1) if p != 0] # default [-2, -1, 1, 2]

    hghg = [p for p in periods]
    highs = [df['high'] > df['high'].shift(p) for p in periods]
    bears = pd.Series(np.logical_and.reduce(highs), index=df.index)

    lows = [df['low'] < df['low'].shift(p) for p in periods]
    bulls = pd.Series(np.logical_and.reduce(lows), index=df.index)

    result = pd.concat([bears, bulls], axis=1).rename(columns={0:'bears', 1:'bulls'})
    return result

class RSI_Divergence(RSI):
    def __init__(self, params):
        super().__init__(params=params)
        # super().__init__(name='rsi', params=params)
        self.header = ['sma']

    def calc(self, df):
        seria = df['close']
        subseria = seria[-self.depth:]

        # get values
        values = self.func(subseria)
        values2 = self.func(seria)
        fractals = will_frac(df, 2)
        last_two_highs = fractals[fractals['bears'] == True]['bears']
        last_two_lows = fractals[fractals['bulls'] == True]['bulls']
        highs = df['high'].loc[last_two_highs[-2:].index]

        rsi_values = values2[last_two_highs[-2:].index]
        if highs.iloc[0] < highs.iloc[1] and rsi_values[0] > rsi_values[1]:
            bearish = True
        else:
            bearish = False

        lows = df['low'].loc[last_two_lows[-2:].index]
        rsi_values = values2[last_two_lows[-2:].index]
        if lows.iloc[0] > lows.iloc[1] and rsi_values[0] < rsi_values[1]:
            bullish = True
        else:
            bullish = False

        return [bearish, bullish]


def compare_features_time_calc():
    seria = get_quotes('BTCUSDT', '15m')['close']
    sma266 = SMA(params=[Parameter(name='timeperiod', value=266, search_space=tune.randint(11,121))])
    sma11 = SMA(params=[Parameter(name='timeperiod', value=11, search_space=tune.randint(11,121))])
    bbwp = BBWP_Caretaker(params=[Parameter(name='lookback', value=252, search_space=tune.randint(100,300)),
                                Parameter(name='bbands_len', value=13, search_space=tune.randint(11,30))])
    bbands13 = BBANDS(params=[Parameter(name='timeperiod', value=13)])
    bbands50 = BBANDS(params=[Parameter(name='timeperiod', value=50)])
    def check_feature_calc_time(feature, seria, num=1000):
        t0 = time.time()
        for i in range(num):
            values = feature.calc(seria)
        t1 = time.time()
        return (t1-t0)/num

    print('sma11: ', check_feature_calc_time(sma11, seria))
    print('sma266: ',check_feature_calc_time(sma266, seria))
    print('bbands13: ',check_feature_calc_time(bbands13, seria))
    print('bbands50: ',check_feature_calc_time(bbands50, seria))
    print('bbwp: ',check_feature_calc_time(bbwp, seria))

def ray_remote_features_test():
    seria = get_quotes('BTCUSDT', '15m')['close']
    sma266 = SMA_remote.remote(params=[Parameter(name='timeperiod', value=266, search_space=tune.randint(11,121))])
    sma11 = SMA_remote.remote(params=[Parameter(name='timeperiod', value=11, search_space=tune.randint(11,121))])
    bbwp = BBWP_Caretaker_remote.remote(params=[Parameter(name='lookback', value=252, search_space=tune.randint(100,300)),
                                Parameter(name='bbands_len', value=13, search_space=tune.randint(11,30))])
    bbands13 = BBANDS_remote.remote(params=[Parameter(name='timeperiod', value=13)])
    bbands50 = BBANDS_remote.remote(params=[Parameter(name='timeperiod', value=50)])

    features=[sma266, sma11, bbwp, bbands13, bbands50]

    def get_header(features):
        header = []
        for f in features:
            header.extend(ray.get(f.get_header.remote()))

        return header

    header = get_header(features)
    print(header)
    print('len(header) ', len(header))
    t0 = time.time()
    tasks_pre_launch = [f.calc.remote(seria) for f in features]
    tasks = ray.get(tasks_pre_launch)

    state=[]
    for t in tasks:
        if type(t) == list:
            for i in t:
                state.append(i[-1])
        else:
                state.append(t[-1])

    feed = pd.DataFrame(data=[state], columns=header, index=['2020-11-11'])
    print(feed.to_string())

def test_jose_feature_set(remote=False):
    if remote:
        bbwp  = BBWP_Caretaker_remote.remote(params=[])
        macd  = MACD_remote.remote(params=[])
        rsi   = RSI_remote.remote(params=[])
        stoch = Stochastic_remote.remote(params=[])
    else:
        bbwp  = BBWP_Caretaker(params=[])
        macd  = MACD(params=[])
        rsi   = RSI(params=[])
        stoch = Stochastic(params=[])

    features=[sma266, sma11, bbwp, bbands13, bbands50]

    def get_header(features):
        header = []
        for f in features:
            header.extend(ray.get(f.get_header.remote()))

        return header

    header = get_header(features)
    print(header)
    print('len(header) ', len(header))
    t0 = time.time()
    tasks_pre_launch = [f.calc.remote(seria) for f in features]
    tasks = ray.get(tasks_pre_launch)

    state=[]
    for t in tasks:
        if type(t) == list:
            for i in t:
                state.append(i[-1])
        else:
                state.append(t[-1])

    feed = pd.DataFrame(data=[state], columns=header, index=['2020-11-11'])
    print(feed.to_string())


def test_stream_api():

    # How I found out stream API doesn't work actually
    # this test demonstrate it
    # also there are mentions about this problem on talib repo

    close = np.random.random(100000)


    def function_api():
        # the Function API
        # output = talib.SMA(close, timeperiod=25)
        output = talib.STOCHRSI(close, timeperiod=25)

    def streaming_api():
        # the Streaming API
        # latest = stream.SMA(close, timeperiod=25)
        latest = stream.STOCHRSI(close, timeperiod=25)

    def abstract_api():
        # output = abstract.SMA(close, timeperiod=25)#, price='open')
        output = abstract.STOCHRSI(close, timeperiod=25)#, price='open')

    # check execution time
    time1 = timeit.timeit(function_api, number=100)
    time2 = timeit.timeit(abstract_api, number=100)
    time3 = timeit.timeit(streaming_api, number=100)

    # print(time1, time2, time3)



    latest = stream.STOCHRSI(close, timeperiod=25)
    # print('latest: ',latest)
    # output = abstract.STOCHRSI(close, timeperiod=25)#, price='open')
    output = talib.STOCHRSI(close, timeperiod=25)#, price='open')
    print(len(output))
    div0 = abs(latest[0] - output[0][-1])
    div1 = abs(latest[1] - output[1][-1])
    print(latest[0], '  ',output[0][-1], ' div ', div0)
    print(latest[1], '  ',output[1][-1], ' div ', div1)

    # the latest value is the same as the last output value
    # assert (output[0][-1] - latest[0]) < 0.00001
    # assert (output[1][-1] - latest[1]) < 0.00001

    assert div0 < 0.00001
    assert div1 < 0.00001
    # assert 5 < 0.00001

def test_stream_api_inputs():
    import numpy as np

    # note that all ndarrays must be the same length!
    inputs = {
        'open': np.random.random(100),
        'high': np.random.random(100),
        'low': np.random.random(100),
        'close': np.random.random(100),
        'volume': np.random.random(100)
    }

    df = pd.DataFrame(inputs)
    print(df)

    def streaming_api():
        # the Streaming API
        latest = stream.SMA(df, timeperiod=25)

    streaming_api()

def test_abstract_api():

    inputs = {
        'open': np.random.random(100),
        'high': np.random.random(100),
        'low': np.random.random(100),
        'close': np.random.random(100),
        'volume': np.random.random(100)
    }

    df = pd.DataFrame(inputs)
    print(df)

    print(abstract.Function('sma').info)


    def abstract_api():
        output = abstract.SMA(df, timeperiod=25)#, price='open')
        return output



    sma = abstract_api()
    print(type(sma), sma)

    print(abstract.BBANDS.info)
    bbands = abstract.BBANDS(df)
    print(type(bbands),bbands)


def test_get_depth():
    sma = SMA({})
    # print('depth ', sma.depth)
    assert sma.depth == 30


def test_func_params():
    from pprint import pprint
    for i, name in enumerate(talib.get_functions(), 0):
        # print(name)
        f = abstract.Function(name)
        # pprint(f.info)
        print(f'#{i}  n: {name}, inputs: {f.input_names} params: {f.parameters}')
        # return

    # for name in abstract.get_functions():
    #     print(name)

def test_latest():
    sma = SMA({})

    inputs = {
        'open': np.random.random(100),
        'high': np.random.random(100),
        'low': np.random.random(100),
        'close': np.random.random(100),
        'volume': np.random.random(100)
    }

    df = pd.DataFrame(inputs)['close']
    # r = sma.latest(df)
    # pprint(r)

    ndr = df.to_numpy()


    def to_numpy():
        re = df.to_numpy()

    print(ndr)
    # return
    def latest():
        # r = sma.latest(df)
        r = sma.latest(ndr)

    def streaming_api():
        # the Streaming API
        r = stream.SMA(ndr)

    time1 = timeit.timeit(latest, number=1000)
    time2 = timeit.timeit(streaming_api, number=1000)
    time3 = timeit.timeit(to_numpy, number=1000)

    print(time1, time2, time1/time2)
    print(time3)

def test_rsi_divergence():

    div = RSI_Divergence({})

    inputs = {
        'open': np.random.random(100),
        'high': np.random.random(100),
        'low': np.random.random(100),
        'close': np.random.random(100),
        'volume': np.random.random(100)
    }

    df = pd.DataFrame(inputs)['close'].to_numpy()

    result = div.calc(pd.DataFrame(inputs))
    print(result)


    # def latest():
    #     # r = sma.latest(df)
    #     r = div.latest(ndr)

    # def streaming_api():
    #     # the Streaming API
    #     r = stream.S(ndr)

    # time1 = timeit.timeit(latest, number=1000)
    # time2 = timeit.timeit(streaming_api, number=1000)



if __name__ == "__main__":
    ## compare_features_time_calc()
    # ray_remote_features_test()
    # test_jose_feature_set()
    # test_stream_api()
    # test_stream_api_inputs()
    # test_abstract_api()
    # test_get_depth()
    # test_func_params()
    # test_latest()
    test_rsi_divergence()

