
from typing import Union

from . import actions
from . import rewards
from . import observers
from . import stoppers
from . import informers
from . import renderers

from tensortrade.env.generic import TradingEnv
from tensortrade.env.generic.components.renderer import AggregateRenderer

import pandas as pd
import numpy as np
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.wallets import Wallet, Portfolio
from gymnasium.wrappers import EnvCompatibility, FlattenObservation, StepAPICompatibility
from pprint import pprint
import random
import math
import warnings
from icecream import ic

import static_frame as sf
from quantutils.parameters import get_param


# SYNTHETIC PROCESSES
SIN = 'SIN'
FLAT = 'FLAT'

# data = sf.Frame.from_csv(sf.WWW.from_file('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'), columns_depth=0)
# turn off pandas SettingWithCopyWarning 
pd.set_option('mode.chained_assignment', None)


def close(obs):
    return obs[0]

def get_feed(env):
    return get_env(env).config['data']['feed']

def get_observer(env):
    return env.env.env.env.observer

def get_info(env):
        return env.env.env.env.informer.info(env.env.env.env)

def get_env(env):
    return env.env.env.env


def get_exchange_pair(env):
    self.current_symbol_code

def get_obs_header(env):
    env = get_env(env)
    one_point_header = list(env.config['data']['feed'].columns)
    if 'symbol' in one_point_header:
        one_point_header.remove('symbol')

    if get_param(env.config['params'], 'multy_symbol_env')['value'] == True:
        one_point_header.remove('end_of_episode')
        one_point_header.remove('symbol_code')

    window_size = get_param(env.config['params'],'window_size')['value']
    header=[]
    # add lag indexing in the beginning of name
    for i in range(window_size-1,-1,-1):
        for h in one_point_header:
            header.append(f'{i}_{h}')

    return header

def get_info(env):
        return env.env.env.env.informer.info(env.env.env.env)

def get_action_scheme(env):
    return env.env.env.env.action_scheme

def create(portfolio: 'Portfolio',
           action_scheme: 'Union[actions.TensorTradeActionScheme, str]',
           reward_scheme: 'Union[rewards.TensorTradeRewardScheme, str]',
           feed: 'DataFeed',
           window_size: int = 1,
           min_periods: int = None,
           random_start_pct: float = 0.00,
           **kwargs) -> TradingEnv:
    """Creates the default `TradingEnv` of the project to be used in training
    RL agents.

    Parameters
    ----------
    portfolio : `Portfolio`
        The portfolio to be used by the environment.
    action_scheme : `actions.TensorTradeActionScheme` or str
        The action scheme for computing actions at every step of an episode.
    reward_scheme : `rewards.TensorTradeRewardScheme` or str
        The reward scheme for computing rewards at every step of an episode.
    feed : `DataFeed`
        The feed for generating observations to be used in the look back
        window.
    window_size : int
        The size of the look back window to use for the observation space.
    min_periods : int, optional
        The minimum number of steps to warm up the `feed`.
    random_start_pct : float, optional
        Whether to randomize the starting point within the environment at each
        observer reset, starting in the first X percentage of the sample
    **kwargs : keyword arguments
        Extra keyword arguments needed to build the environment.

    Returns
    -------
    `TradingEnv`
        The default trading environment.
    """

    action_scheme = actions.get(action_scheme) if isinstance(action_scheme, str) else action_scheme

    action_scheme.portfolio = portfolio

    config = kwargs.get("config", {})
    observer = observers.TensorTradeObserver(
        portfolio=portfolio,
        feed=feed,
        renderer_feed=kwargs.get("renderer_feed", None),
        window_size=window_size,
        min_periods=min_periods,
        num_service_cols = 0 if config.get("multy_symbol_env", False) == False else config.get('num_service_cols', 2)
    )

    stopper = stoppers.MaxLossStopper(
        max_allowed_loss=kwargs.get("max_allowed_loss", 0.5)
    )

    renderer_list = kwargs.get("renderer", renderers.EmptyRenderer())

    if isinstance(renderer_list, list):
        for i, r in enumerate(renderer_list):
            if isinstance(r, str):
                renderer_list[i] = renderers.get(r)
        renderer = AggregateRenderer(renderer_list)
    else:
        if isinstance(renderer_list, str):
            renderer = renderers.get(renderer_list)
        else:
            renderer = renderer_list



    env = TradingEnv(
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        observer=observer,
        stopper=kwargs.get("stopper", stopper),
        informer=kwargs.get("informer", informers.TensorTradeInformer()),
        renderer=renderer,
        min_periods=min_periods,
        random_start_pct=random_start_pct,
        config=kwargs.get('config', {})
    )
    return env




def make_sin_feed(length=1000):
    x = np.arange(0, 2*np.pi, 2*np.pi / (length + 1))
    y = 50*np.sin(3*x) + 100
    xy = pd.DataFrame(data=np.transpose([y]), index=x)#.assign(symbol=pd.Series(np.full(len(x), symbol_name)).values).assign(symbol_code=pd.Series(np.full(len(x), symbol_code)).values)
    xy.columns=['close'] #, 'symbol', 'symbol_code']
    xy.index.name = "datetime"
    return xy

def make_flat_feed(length=1000, price_value=100):
    x = np.arange(0, 2*np.pi, 2*np.pi / (length + 1))
    y = np.full(np.shape(x), float(price_value))
    xy = pd.DataFrame(data=np.transpose([y]), index=x)#.assign(symbol=pd.Series(np.full(len(x), symbol_name)).values)#.assign(symbol_code=pd.Series(np.full(len(x), symbol_code)).values)
    xy.columns=['close'] #, 'symbol', 'symbol_code']
    xy.index.name = "datetime"
    return xy




def get_episodes_lengths(feed):
    lens = []
    steps_in_this_episode=0
    for i,row in feed.iterrows():
        steps_in_this_episode+=1
        if row.loc["end_of_episode"] == True:
            lens.append(steps_in_this_episode)
            steps_in_this_episode = 0
    return lens

def split(N, num_parts):
    a = np.arange(0, N, 1)
    # print('a: ', a)
    b = np.array_split(a, num_parts)

    d = [len(c) for c in b]
    random.shuffle(d)

    result=[]
    start = 0
    for i,v in enumerate(d):
        result.append([start, start+v])
        start += v

    return d, result

def test_split():

    a = np.arange(0, 95, 1)
    r,t  = split(95,3)
    print(r, t)
    assert max(r) - min(r) <= 1
    for j,v in enumerate(t,0):
        # print(j,v)
        chunk = a[v[0]:v[1]]
        # print(len(chunk), chunk)
        assert len(chunk) == r[j]


    a = np.arange(0, 95, 1)
    r,t = split(95,7)
    print(r, t)
    assert max(r) - min(r) <= 1
    for j,v in enumerate(t,0):
        # print(j,v)
        chunk = a[v[0]:v[1]]
        # print(len(chunk), chunk)
        assert len(chunk) == r[j]

    a = np.arange(0, 95, 1)
    r,t = split(95,11)
    print(r, t)
    assert max(r) - min(r) <= 1
    for j,v in enumerate(t,0):
        # print(j,v)
        chunk = a[v[0]:v[1]]
        # print(len(chunk), chunk)
        assert len(chunk) == r[j]

    a = np.arange(0, 666, 1)
    r,t = split(666,36)
    print(r, t)
    assert max(r) - min(r) <= 1
    for j,v in enumerate(t,0):
        # print(j,v)
        chunk = a[v[0]:v[1]]
        # print(len(chunk), chunk)
        assert len(chunk) == r[j]

def make_folds(config):
    # NFOLDCV_MODE_PROPORTIONAL
    for i, s in enumerate(config["symbols"], 0):
        feed_length = len(s["feed"])
        fold_length = int(feed_length / config["num_folds"])
        _,raw_folds = split(feed_length, config["num_folds"])
        ic(f'AFTER raw_folds {raw_folds}')
        all_episodes=[]
        folds=[]
        last_episode_end_index=0
        for start, end in raw_folds:
            num_of_episodes = math.ceil((end - start) / config["max_episode_length"])
            _,episodes = split(end - start, num_of_episodes)
            ic(f'   {episodes=}')
            episodes = [[t + last_episode_end_index for t in e] for e in episodes]
            ic(f'   after {episodes=}')
            last_episode_end_index = episodes[-1][-1] #- last_episode_end_index
            a = len(all_episodes)
            b = a + len(episodes)
            all_episodes = [*all_episodes, *episodes]
            folds.append([a,b])
            for e in episodes:
                if e[1] - e[0] < config.get("min_episode_length",1):
                    ic('end - start ', end - start, ' max_episode_length ', config["max_episode_length"], ' min_episode_length ', config["min_episode_length"])
                    warnings.warn("some episode length is less then min_episode_length  ¯\_(ツ)_/¯. Try to fix your config", Warning)

        s["folds"] = folds
        s["episodes"] = all_episodes

    return config

def test_make_folds():
    print('hey')

    config = {
              "max_episode_length": 15, # smaller is ok
              "min_episode_length": 5, # bigger is ok, smaller is not
              "make_folds":True,
              "num_folds": 3,
              # "symbols": make_symbols(5, 410),
              "symbols": make_symbols(2, 160),
              "cv_mode": 'proportional',
              "test_fold_index": 3,
              "reward_window_size": 1,
              "window_size": 2,
              "max_allowed_loss": 0.9,
              "use_force_sell": True,
              "multy_symbol_env": True,
              "test": False
             }


    # dataset = pd.concat([config["symbols"][i]["feed"] for i in range(len(config["symbols"]))])

    # env = create_multy_symbol_env(config)
    action = 0 # 0 - buy asset, 1 - sell asset
    config["nn_topology_a"] = 7
    config["nn_topology_b_to_a_ratio"] = 0.3
    config["nn_topology_c_to_b_ratio"] = 0.7
    config["nn_topology_h_to_l_ratio"] = 2
    config = make_folds(config)


    for s in config["symbols"]:
        assert len(s["folds"]) == config["num_folds"]
        assert len(s["episodes"]) >= config["num_folds"]
        assert s["episodes"][-1][-1] == len(s["feed"])
        # print(s["folds"])
        # print(s["episodes"])


def make_symbols(num_symbols=5, num_of_samples=666, shatter_on_episode_on_creation = False, process = SIN):
    # TODO: rename --> make_synthetic_symbols

    symbols=[]
    for i in range(num_symbols):
        spread = 0.01
        commission=0
        if i == 2:
            commission=0
            spread=1.13
        elif i == 4:
            commission=0
            spread=3.66

        config = {"name": "AST"+str(i),
              "spread": spread,
              "commission": commission,
              "code": i,
              "num_of_samples": num_of_samples,
              "max_episode_steps": 11,
              # "max_episode_steps": 152,
              # "process": 'flat',
              "process": process,
              "price_value": 100,
              "shatter_on_episode_on_creation": shatter_on_episode_on_creation}

        symbols.append(make_synthetic_symbol(config))

    return symbols

def get_wallets_volumes(wallets):
    volumes = []
    for w in wallets:
        balance = w.total_balance
        volumes.append(float(balance.size))
    return volumes

def get_observer(env):
    return env.env.env.env.observer

def is_end_of_episode(env):
    return env.env.env.env.end_of_episode

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
    if config["make_folds"] == True:
        # print('should make folds')
        if config["test"] == True:
            return (get_train_test_feed(config, test_only=True)[1])
        else:
            return (get_train_test_feed(config, train_only=True)[0])
    else:
        return (pd.concat([config["symbols"][i]["feed"] for i in range(len(config["symbols"]))]))


def create_multy_symbol_env(config):

    ic.disable()
    # do some parameters check here
    if config["make_folds"] == True and config.get('test_fold_index',0) >= config.get('num_folds',3):
        raise ValueError(' test_fold_index is bigger then num_folds ¯\_(ツ)_/¯')

    # i = [0 if config["test"] == False else 1]
    # print('i ', i)
    dataset = get_dataset(config)#.drop('symbol', axis=1) # [ 0 if config["test"] == False else 1]

    exchanges=[]
    wallets=[]
    exchange_options = ExchangeOptions(commission=config["symbols"][-1]["commission"], config=config)

    ends = dataset.loc[dataset['end_of_episode'] == True]
    print(ends)
    prices=[]
    for i,s in enumerate(config["symbols"],0):
        price=[]
        for j in range(len(config["symbols"])):
            # FIXME: symbols should not have feed, only quotes and timeframes
            #        we make feed for 'create' method only
            #
            values = config["symbols"][j]["feed"]["close"].values
            if j == i:
                price.extend(values)
            else:

                price.extend(np.ones(len(values)))


        symbol_name = s["name"]
        base_symbol_name = config.get("base_symbol", "USDT")
        prices.append(Stream.source(price, dtype="float").rename(f"{base_symbol_name}/{symbol_name}"))

    exchange = Exchange('binance', service=execute_order, options=exchange_options)(*prices)

    # FIXME: USDT --> to some base account curtrency from config
    USDT = Instrument("USDT", 2, "USD Tether")
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

    for c in dataset.columns[0:]:
        if c != 'symbol' and c != 'end_of_episode':
            s = Stream.source(list(dataset[c]), dtype="float").rename(dataset[c].name)
        elif c == 'end_of_episode':
            s = Stream.source(list(dataset[c]), dtype="bool").rename(dataset[c].name)
        features += [s]

    feed = DataFeed(features)
    feed.compile()
    # reward_scheme = default.rewards.SimpleProfit(window_size=config["reward_window_size"])
    # action_scheme = default.actions.MultySymbolBSH(config)
    reward_scheme = rewards.SimpleProfit(window_size=config["reward_window_size"])
    action_scheme = actions.MultySymbolBSH(config)

    env = create(
            feed=feed
            ,portfolio=portfolio
            ,action_scheme=action_scheme
            ,reward_scheme=reward_scheme
            ,renderer=[]
            ,window_size=config["window_size"]
            ,max_allowed_loss=config["max_allowed_loss"]
            ,config=config
            ,informer=informers.MultySymbolEnvInformer()
    )


    env = StepAPICompatibility(FlattenObservation(EnvCompatibility(env)))
    return env
