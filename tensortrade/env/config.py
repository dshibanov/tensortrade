import sys, os, time
import datetime, itertools
# from alive_progress import alive_bar
# from quantlib.agents.default.features import SMA, BBANDS, BBWP_Caretaker, Parameter
# from ray import tune
# import modin.pandas
import pandas as pd
import numpy as np
# import quantlib.utils.constants
from quantutils.parameters import get_param
import  quantutils.constants as constants
from tensortrade.feed.core import Stream, DataFeed
import importlib as ilib
import tensortrade.env.default as default
# import api.datamart.datamart as dm

from tensortrade.oms.instruments import Instrument
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.services.execution.simulated import execute_order
# from ray.tune.tune import _Config
from pprint import pprint
import tensortrade.env.default as default
from tensortrade.env.default import split
import math
import copy

FEED_MODE_NORMAL = 'normal'
FEED_MODE_MODIN = 'modin'
FEED_MODE_RAY = 'ray'
FEED_MODE_RAY_VERTICAL = 'ray_vertical'
FEED_MODE_RAY_AND_MODIN = 'ray&modin'
TICKS_PER_BAR = 4


TIMEFRAMES = {'0': 0, # this is for ticks
              '1m': 1,
              '3m': 3,
              '5m': 5,
              '15m':15,
              '30m':30,
              '1h': 60,
              '2h':120,
              '4h': 240,
              '6h': 360,
              '8h': 480,
              '12h': 720,
              '1d': 1440,
              '3d': 4320,
              '1w': 10080,
              '1M': 43200}

pd.options.mode.chained_assignment = None  # default='warn'


# def get_search_space(params):
#     search_space = {}
#     for p in params:
#         if p.get('optimize', False) == True:
#             search_space[p['name']] = get_distribution(p)
#     return search_space


def get_distribution(config):
    # func_name = config.get('distribution', 'uniform')
    func_name = config.get('distribution', 'randint')
    module_path = 'ray.tune'
    module = ilib.import_module(module_path)
    f_name = getattr(module, func_name)
    keys_to_remove = {'name', 'distribution', 'optimize', 'value'}
    params = {k: v for k, v in config.items() if k not in keys_to_remove}
    return f_name(**params)


def get_agent(config):
    agent_name = config["name"]
    agent_params = config["params"]
    class_name = agent_name.rpartition('.')[-1]
    module_path, class_name = agent_name.rsplit('.', 1)
    module = ilib.import_module(module_path)
    class_name = getattr(module, class_name)
    return class_name(agent_params)

def prepare(config):

    find_and_normalize(config, ['from', 'to'], normalize_datetime)
    if 'timeframes' not in config['env']['data']:
        raise Exception("Timeframes is not specified. Please specify timeframes")
    else:
        # config['env']['data']['timeframes'] = sorted(config['env']['data']['timeframes'], key=lambda x: constants.TIMEFRAMES[x], reverse=False)
        config['env']['data']['timeframes'] = sorted(config['env']['data']['timeframes'], key=lambda x: TIMEFRAMES[x], reverse=False)

    config['env']['data']['features'] = config['env']['data'].get('features', {})

    for s in config['env']['data']['symbols']:
        if 'timeframes' not in s:
            s['timeframes'] = config['env']['data']['timeframes']
        else:
            # s['timeframes'] = sorted(s['timeframes'], key=lambda x: constants.TIMEFRAMES[x], reverse=False)
            s['timeframes'] = sorted(s['timeframes'], key=lambda x: TIMEFRAMES[x], reverse=False)

    config['env']['use_force_sell'] = config['env'].get('use_force_sell', True)
    config['env']['make_folds'] = config['env'].get('make_folds', False)
    config['env']['data']['load_feed_from'] = config['env']['data'].get('load_feed_from', '')
    config['env']['data']['num_folds'] = config['env']['data'].get('num_folds', 1)
    config['env']['data']['test_fold_num'] = config['env']['data'].get('test_fold_num', 0)
    if config['env']['data'].get('symbols', {}) == {}:
        config['env']['data']['symbols'] = (config['datamart'].get_symbols(config['env']['data']['symbols']))
    config['env']['multy_symbol_env'] = get_param(config['env']['params'], 'multy_symbol_env')['value']

    return config

class Bar:
    def __init__(self, open=0, low=0, high=0, close=0, volume=0):
        self.open = open
        self.low = low
        self.high = high
        self.close = close
        self.volume = volume

    def __repr__(self):
        return f'BAR: i:{self.index}, o:{self.open}, l:{self.low}, h:{self.high}, c:{self.close}, v:{self.volume}'

    def opened(self,index,price,volume):
        self.index = index
        self.close = self.high = self.low = self.open = price
        self.volume = volume
        return self

    def update(self, price, volume):
        self.close = price
        if price > self.high:
            self.high = price

        if price < self.low:
            self.low = price

        self.volume += volume

    def df(self):
        df = pd.DataFrame(data=[[self.open, self.high, self.low, self.close, self.volume]]
                            ,index = [self.index]
                            ,columns = ['open', 'high', 'low', 'close', 'volume'])
        df.index.name = 'date'
        return df


def date_from_str(t):
    t = str(t)
    if len(t) > 10:
        return datetime.datetime.strptime(t,'%Y-%m-%d  %H:%M:%S')
    else:
        return datetime.datetime.strptime(t,'%Y-%m-%d')

def get_timestamp(t):
    t = str(t)
    print(t)
    if len(t) > 10:
        return datetime.datetime.strptime(t,'%Y-%m-%d  %H:%M:%S.%f').timestamp()
    else:
        return datetime.datetime.strptime(t,'%Y-%m-%d').timestamp()

def get_bar_on_time(seria, current_time):
    # return bar index which consisted time 'current_time'
    i = 0
    current_timestamp = get_timestamp(current_time)

    for i, row in enumerate(seria.itertuples(), 1):

        if get_timestamp(getattr(row,'Index')) > current_timestamp:
            return i - 2
    return -1


def normalize_datetime(dt):
    # if len(dt) < 11:
    if len(str(dt)) < 11:
        return dt + ' 00:00:00'
    return dt

def find_and_normalize(d, keys, norm_func):
    if isinstance(d, dict):
        for k,v in d.items():
            if isinstance(v,dict) or isinstance(v, list):
                find_and_normalize(v, keys, norm_func)
            else:
                if k in keys:
                    d[k] = norm_func(v)
    elif isinstance(d, list):
        for i in d:
            find_and_normalize(i, keys, norm_func)





def get_train_test_feed(config, train_only=False, test_only=False):
    if train_only == True and test_only == True:
            raise ValueError('Wrong settings, no folds will be retuned    ¯\_(ツ)_/¯')

    for s in config["symbols"]:
        print(s["feed"].to_markdown())
        lengths = get_episodes_lengths(s["feed"])
        # print(f'before make_folds {lengths=}')
        assert min(lengths) > 3


    test_fold_index = config.get("test_fold_index",0)
    symbols = config["symbols"]
    train_feed = pd.DataFrame()
    test_feed = pd.DataFrame()
    for s in symbols:
        # print('s_feed ', s["feed"].to_markdown())
        train_episodes=[]
        test_episodes=[]
        for fold_num in range(config["num_folds"]):
            if fold_num == test_fold_index:
                test_episodes.extend(s["episodes"][s["folds"][fold_num][0]: s["folds"][fold_num][1]])
            else:
                train_episodes.extend(s["episodes"][s["folds"][fold_num][0]: s["folds"][fold_num][1]])

        if not test_only:
            for e in train_episodes:
                train_feed = pd.concat([train_feed, s["feed"].iloc[e[0]: e[1]]])
                train_feed.iloc[-1, train_feed.columns.get_loc('end_of_episode')] = True

        if not train_only:
            print(f'  {test_episodes=}')
            for e in test_episodes:
                # print(f'     test_feed {e[0]} {e[1]}')
                ic(f'     test_feed {e[0]} {e[1]}')
                # print(f'>>>> s_feed {s["feed"].iloc[e[0]: e[1]]}')
                test_feed = pd.concat([test_feed, s["feed"].iloc[e[0]: e[1]]])
                # print(test_feed.head(20))
                ic(test_feed.head(20))
                test_feed.iloc[-1, test_feed.columns.get_loc('end_of_episode')] = True
                # print(test_feed.head(20))

    return train_feed, test_feed


def get_dataset(config):
    if get_param(config['params'], "make_folds") == True:
        if config["test"] == True:
            return (get_train_test_feed(config, test_only=True)[1])
        else:
            return (get_train_test_feed(config, train_only=True)[0])
    else:
        quotes = {}
        for t in config['data']['timeframes']:
            for s in config['data']['symbols']:
                s['quotes'][t]['end_of_episode'] = False
                s['quotes'][t].iloc[-1, s['quotes'][t].columns.get_loc('end_of_episode')] = True
                s['quotes'][t]["symbol_code"] = s['code']
            quotes[t] = pd.concat([s['quotes'][t] for s in config['data']['symbols']])
        return config['data']['feed']

def make_folds(config):
    # NFOLDCV_MODE_PROPORTIONAL
    for i, s in enumerate(config["data"]["symbols"], 0):
        feed_length = len(s["feed"])
        fold_length = int(feed_length / config['data']["num_folds"])
        _,raw_folds = split(feed_length, config['data']["num_folds"])
        # ic(f'AFTER raw_folds {raw_folds}')
        # print(f'AFTER raw_folds {raw_folds}')
        all_episodes=[]
        folds=[]
        last_episode_end_index=0
        for start, end in raw_folds:
            num_of_episodes = math.ceil((end - start) / config['data']["max_episode_length"])
            _,episodes = split(end - start, num_of_episodes)
            # print(f'   {episodes=}')
            episodes = [[t + last_episode_end_index for t in e] for e in episodes]
            # print(f'   after {episodes=}')
            last_episode_end_index = episodes[-1][-1] #- last_episode_end_index
            a = len(all_episodes)
            b = a + len(episodes)
            all_episodes = [*all_episodes, *episodes]
            folds.append([a,b])
            for e in episodes:
                if e[1] - e[0] < config['data'].get("min_episode_length",1):
                    # print('end - start ', end - start, ' max_episode_length ', config['data']["max_episode_length"], ' min_episode_length ', config['data']["min_episode_length"])
                    warnings.warn("some episode length is less then min_episode_length  ¯\_(ツ)_/¯. Try to fix your config", Warning)

        s["folds"] = folds
        s["episodes"] = all_episodes

    # return config

# class EnvConfig(_Config):
class EnvConfig:
    def __init__(self, config = {}):
        self.feed_calc_mode = get_param(config['params'], 'feed_calc_mode', FEED_MODE_NORMAL)['value']
        if self.feed_calc_mode == FEED_MODE_MODIN or self.feed_calc_mode == FEED_MODE_RAY_AND_MODIN:
            self.pd=modin.pandas
        else:
            self.pd=pd

        self.config = config
        self.timeframes = config['data']['timeframes']

        # if 'timeframes' in config['data']:
        #     self.timeframes = config['data']['timeframes']
        # else:
        #     raise Exception("Timeframes are not specified. Please specify timeframes")

        if ('symbols' in self.config['data']):
            self.symbols = self.config['data']['symbols']
            for s in self.symbols:
                for t in self.timeframes:
                    if (t in s['quotes']) == False:
                        raise Exception(f"Some quotes are not specified. Please specify quotes of timeframe: {t}")
        else:
            raise Exception("Symbols are not specified. Please specify symbols")

        self.config['data']['current_symbol_index'] = config['data'].get('current_symbol_index',0)
        self.multy_symbol_env = self.gp('multy_symbol_env')['value']
        self.features = config['data'].get('features', {})
        self.feed = {}
        return


    def get_header(self, config):
        local_header=[]
        for t in self.features:
            for f in self.features[t]:
                if get_param(config['params'], 'feed_calc_mode') == FEED_MODE_RAY or get_param(config['params'], 'feed_calc_mode') == FEED_MODE_RAY_AND_MODIN or get_param(config['params'], 'feed_calc_mode') == FEED_MODE_RAY_VERTICAL:
                    local_header.extend(f'{t}_'+ray.get(f.get_header.remote()))
                else:
                    local_header.extend([f'{t}_'+ i for i in f.get_header()])

        return local_header


    def gp(self, p):
        # shortcut for get_param
        return get_param(self.config['params'], p)

    def get_reward_scheme(self):
        name = self.config['reward_scheme']['name']
        params = self.config['reward_scheme']['params']
        module_path, class_name = name.rsplit('.', 1)
        module = ilib.import_module(module_path)
        _class = getattr(module, class_name)
        scheme = _class(**{p['name']: p['value'] for p in params})
        return scheme

    def get_action_scheme(self):
        name = self.config['action_scheme']['name']
        params = self.config['action_scheme']['params']
        module_path, class_name = name.rsplit('.', 1)
        module = ilib.import_module(module_path)
        _class = getattr(module, class_name)

        if class_name == 'BSH':
            params = {'cash': self.cash, 'asset': self.asset}
        elif class_name == 'MultySymbolBSH':
            return _class(self.config)
        elif class_name == 'MSBSCMT':
            return _class(self.config)
        else:
            params = {p['name']: p['value'] for p in params}

        scheme = _class(**params)
        return scheme

    def make_feed0(self, config = {}):

        if config == {}:
            config = self.config

        _from = config['data']['from']
        _to = config['data']['to']
        # if 'load_feed_from' in config and config['load_feed_from'] != '':
        # if 'load_feed_from' in config and config['load_feed_from'] != '':
        #     df = self.pd.read_csv(config['load_feed_from'], index_col='date')
        #     return df

        self.feed0 = self.pd.DataFrame()
        for t in config['data']['timeframes']:
            for s in config['data']['symbols']:
                s['quotes'][t]['end_of_episode'] = False #end_of_episode #pd.Series(np.full(len(s['quotes'][t])-1, False))
                s['quotes'][t].iloc[-1, s['quotes'][t].columns.get_loc('end_of_episode')] = True
                s['quotes'][t]["symbol_code"] = s['code']

        if self.multy_symbol_env == True:
            symbols = config['data']['symbols']
        else:
            symbols = [config['data']['symbols'][self.config['data']['current_symbol_index']]]

        for s in symbols:
            smallesttmf = self.timeframes[0]
            volume = []
            ticks = []
            index = []
            new_bar = []
            quotes = s['quotes']
            seria = quotes[smallesttmf]
            bar_width = seria.index[-1] - seria.index[-2]# bar width in seconds
            step = bar_width / 3
            just_started = True
            last_ohlc={}
            for t in self.timeframes:
                last_ohlc[t] = {f'{t}_open':[], f'{t}_high':[], f'{t}_low':[], f'{t}_close':[], f'{t}_volume':[]}

            last_bars = {}
            counters={}
            start_bar = seria.index.get_loc(config['data']['from'])
            end_bar = seria.index.get_loc(config['data']['to'])
            num_of_ticks =  end_bar - start_bar #ticks.index.get_loc(stop_point) - ticks.index.get_loc(start_point)
            started = False

            for i, row in enumerate(seria.iloc[start_bar:end_bar].itertuples(),1):
                if started == False:
                    started = True
                    for t in self.timeframes:
                        counters[t] = get_bar_on_time(quotes[t], _from)
                        last_bars[t] = Bar().opened(getattr(row, 'Index'), getattr(row, 'open'), 0)

                current_time = getattr(row,'Index')
                index.append(getattr(row, 'Index'))
                if abs(row.high - row.close) < abs(row.low - row.open):
                    ticks.extend([row.open, row.low, row.high, row.close])
                    volume.extend([0, row.volume/3, row.volume/3, row.volume/3])
                    index.extend([index[-1] + step, index[-1] + 2*step, index[-1] + 3*step - datetime.timedelta(seconds=1)])
                else:
                    ticks.extend([row.open, row.high, row.low, row.close])
                    volume.extend([0, row.volume/3, row.volume/3, row.volume/3])
                    index.extend([index[-1] + step, index[-1] + 2*step, index[-1] + 3*step - datetime.timedelta(seconds=1)])

                new_bar.extend([True, False, False, False])

                # FIXME: '-4' is magic number.. we need constant here
                for tick, vol, i in zip(ticks[-4:], volume[-4:], index[-4:]):
                    for t in self.timeframes:
                        if counters[t] < len(quotes[t]) - 1 and current_time == quotes[t].index[
                                counters[t] + 1]:
                            counters[t]+=1
                            last_bars[t].opened(i, tick, vol)
                        else:
                            last_bars[t].update(tick, vol)

                        last_ohlc[t][f'{t}_open'].append(last_bars[t].open)
                        last_ohlc[t][f'{t}_high'].append(last_bars[t].high)
                        last_ohlc[t][f'{t}_low'].append(last_bars[t].low)
                        last_ohlc[t][f'{t}_close'].append(last_bars[t].close)
                        last_ohlc[t][f'{t}_volume'].append(last_bars[t].volume)
                # bar()

            df = self.pd.DataFrame(data=np.transpose([ticks, volume, new_bar]), index=index, columns=['price', 'volume', 'new_bar'])
            df.index.name ='date'
            dfs = [df]
            for t in self.timeframes:
                dfs.append(self.pd.DataFrame(last_ohlc[t], index = index))

            sc = seria['symbol_code'].iloc[0]
            length = len(last_ohlc[smallesttmf][f'{smallesttmf}_volume'])
            addditional_cols = {'end_of_episode': np.full(length, False),
                                'symbol_code': np.full(length, sc)}

            addditional_cols['end_of_episode'][-1] = True
            dfs.append(self.pd.DataFrame(addditional_cols, index=index))
            self.feed0  = self.pd.concat([self.feed0, self.pd.concat(dfs, axis=1)], axis=0)
        return self.feed0

    def make_feed(self, config = {}):
        if config == {}:
            config = self.config

        if self.features == {}:
            self.feed  = self.feed0
            return self.feed

        def is_new_bar(ticks, ticks_per_bar):
            if ticks == 0 or ticks % ticks_per_bar == 0:
                return True
            return False

        self.feed = []
        counters={}
        series={}

        current_symbol_code = self.feed0.iloc[0]['symbol_code']
        quotes = copy.deepcopy(config['data']['symbols'][current_symbol_code]['quotes'])
        for t in self.timeframes:
            counters[t] = quotes[t].index.get_loc(self.feed0.index[0])
            series[t] = quotes[t].iloc[:counters[t]]

        header = self.get_header(config)
        self.feed_calc_mode = get_param(config['params'], 'feed_calc_mode')['value']
        if self.feed_calc_mode == FEED_MODE_RAY_VERTICAL:
            num_of_features = len(self.timeframes)*len(self.features)
            pb = RayProgressBar(num_of_features)
            actor = pb.actor
            t0 = time.time()
            tasks_pre_launch=[]
            for t in self.timeframes:
                tasks_pre_launch.extend([f.calcv.remote(self.quotes[self.symbol], self.feed0, t, actor) for f in self.features])

            pb.print_until_done()
            tasks = ray.get(tasks_pre_launch)

            state=[]
            for t in tasks:
                if len(np.shape(t)) == 2:
                    a = np.squeeze(t).T
                    for i in a:
                        state.append(i)
                else:
                    state.append(t)

        else:
            num_of_ticks = len(self.feed0)
            for i,row in enumerate(self.feed0.itertuples(), 0):
                if current_symbol_code != row.symbol_code:
                    current_symbol_code = row.symbol_code
                    quotes = copy.deepcopy(config['data']['symbols'][current_symbol_code]['quotes'])
                    for t in self.timeframes:
                        counters[t] = quotes[t].index.get_loc(self.feed0.index[0])
                        series[t] = quotes[t].iloc[:counters[t]]

                state = []
                for t in self.timeframes:
                    if self.feed0[f'new_bar'].iloc[i] == 1:  # 0 volume on bar opened tick
                        counters[t] += 1
                        series[t] = quotes[t].iloc[:counters[t]]

                    last_bar = self.feed0[[f'{t}_open', f'{t}_high',f'{t}_low',f'{t}_close',f'{t}_volume']].iloc[i].to_frame().T
                    last_bar.rename(columns={f'{t}_open':'open', f'{t}_high':'high',f'{t}_low':'low',f'{t}_close':'close',f'{t}_volume':'volume'}, inplace=True)
                    last_last_bar  = last_bar.iloc[-1]
                    # series[t].iloc[-1] = last_last_bar.to_dict() #.iloc[-1]
                    # series[t].iloc[-1] = last_bar.iloc[0]
                    for col in last_bar.columns:
                        series[t].loc[series[t].index[-1], col] = last_bar[col].iloc[0]

                if self.feed_calc_mode == FEED_MODE_NORMAL:
                    for t in self.features:
                        for f in self.features[t]:
                            # TODO: send all quotes not only 'close'
                            value = f.calc(series[t]['close'])

                            if type(value) == np.ndarray or type(value) == self.pd.Series:
                                state.append(value[-1])
                            else:
                                for v in value:
                                    state.append(v[-1])

                elif self.feed_calc_mode == FEED_MODE_RAY or self.feed_calc_mode == FEED_MODE_RAY_AND_MODIN:
                    def tasks_to_state(tasks):
                        state=[]
                        for t in tasks:
                            if type(t) == list:
                                for i in t:
                                    state.append(i[-1])
                            else:
                                state.append(t[-1])
                        return state

                    tasks_pre_launch = [f.calc.remote(series[t]['close']) for f in self.features]
                    tasks = ray.get(tasks_pre_launch)
                    state = tasks_to_state(tasks)
                # bar()
                self.feed.append(state)

        header = self.get_header(config)
        df = self.pd.DataFrame(self.feed, columns=header, index=self.feed0.index)
        self.feed = self.pd.concat([self.feed0, df], axis=1).dropna()
        return self.feed

    def build(self, config = {}):

        if config == {}:
            config = self.config
        # else:
        #     self.config = config

        if config.get("make_folds", False) == True and config.get('test_fold_index',0) >= config.get('num_folds',3):
            raise ValueError(' test_fold_index is bigger then num_folds ¯\_(ツ)_/¯')


        def get_feed():
            self.feed0 = self.make_feed0()
            self.feed = self.make_feed()
            config['data']['feed'] = self.feed

            # for s in self.config['data']['symbols']:
            #     s['feed'] = config['data']['feed'].loc[config['data']['feed']['symbol_code'] == s['symbol_code']]

            if self.gp('save_feed')['value']:
                self.feed.to_csv('feed.csv', index=True)

        def set_folds():
            if config['data'].get('num_folds', 1) > 1:
                print('We need to make folds')
                make_folds(config)

                print(f"... and reconfigure feed for test_fold_num: {config['data']['test_fold_num']}")
                # get feed 
                # config['data']['feed'] <-- 
                # from folds and test_fold_num




        # load_feed_from = self.gp("load_feed_from")['value']
        load_feed_from = self.config["data"]["load_feed_from"]
        if  load_feed_from != '':
            try:
                self.feed = pd.read_csv(load_feed_from, index_col=0)
                config['data']['feed'] = self.feed
                print("DataFrame loaded successfully.")
            except FileNotFoundError:
                print("Error: The specified CSV file was not found.")
                get_feed()
        else:
            get_feed()

        # write feed to symbols



        config['symbols'] = []
        for s in config['data']['symbols']:
            s['quotes'][self.timeframes[0]]['end_of_episode'] = False
            s['quotes'][self.timeframes[0]].iloc[-1, s['quotes'][self.timeframes[0]].columns.get_loc('end_of_episode')] = True
            s['quotes'][self.timeframes[0]]["symbol_code"] = s['code']
            config['symbols'].append({'name': s['name'], 'feed': s['quotes'][self.timeframes[0]], 'commission': 0, 'spread': 0.0001})

            s['feed'] = config['data']['feed'].loc[config['data']['feed']['symbol_code'] == s['code']]

        # FIXME:
        # set_folds only when we need it!!
        set_folds()

        exchanges=[]
        wallets=[]
        # exchange_options = ExchangeOptions(commission=config["symbols"][-1]["commission"], config=config)
        exchange_options = ExchangeOptions(config=config['exchange_config'])


        prices=[]
        for i,s in enumerate(config["data"]["symbols"],0):
            price = self.feed['price'].copy()
            indexes = self.feed['symbol_code'] == s['code']
            indexes.reset_index(drop=True, inplace=True)
            no_price_indexes = indexes == True
            price.loc[indexes.values == False] = -666
            symbol_name = s["name"]
            base_symbol_name = config.get("base_symbol", "USDT")
            prices.append(Stream.source(price, dtype="float").rename(f"{base_symbol_name}/{symbol_name}"))

        exchange = Exchange('binance', service=execute_order, options=exchange_options)(*prices)

        # FIXME: USDT --> to some base account currency from config
        USDT = Instrument("USDT", 5, "USD Tether")
        cash = Wallet(exchange, 1000 * USDT)  # This is the starting cash we are going to use
        wallets.append(cash)

        # create assets wallets
        for i,s in enumerate(config["symbols"],0):
            asset_name = s['name']
            asset = Instrument(f'{asset_name}', 5, s['name'])
            asset = Wallet(exchange, 0 * asset)  # And we will start owning 0 stocks of TTRD
            wallets.append(asset)

        portfolio = Portfolio(USDT, wallets)
        features = []

        dataset = self.feed
        col0 = dataset.columns[0:]
        # if self.config['data'].get('add_index',False):
        #     s = Stream.source(list(dataset.index), dtype="bool").rename('index')
        #     features += [s]

        for c in dataset.columns[0:]:
            if c != 'symbol' and c != 'end_of_episode':
                dataset_c = dataset[c]
                name = dataset_c.name
                s = Stream.source(list(dataset[c]), dtype="float").rename(dataset[c].name)
            elif c == 'end_of_episode':
                s = Stream.source(list(dataset[c]), dtype="bool").rename(dataset[c].name)
            features += [s]

        feed = DataFeed(features)
        feed.compile()
        reward_scheme = self.get_reward_scheme() #rewards.SimpleProfit(window_size=config["reward_window_size"])
        action_scheme = self.get_action_scheme() #actions.MultySymbolBSH(config)

        env = default.create(
                feed=feed
                ,portfolio=portfolio
                ,action_scheme=action_scheme
                ,reward_scheme=reward_scheme
                ,renderer=[]
                , window_size=self.gp("window_size")['value']
                ,max_allowed_loss=self.gp("max_allowed_loss")['value']
                ,config=config
                ,informer=default.informers.MultySymbolEnvInformer()
        )

        env = default.StepAPICompatibility(default.FlattenObservation(default.EnvCompatibility(env)))
        return env

# if __name__ == "__main__":
    # test_get_reward_scheme()
    # test_get_action_scheme()

