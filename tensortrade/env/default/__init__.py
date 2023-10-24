
from typing import Union

from . import actions
from . import rewards
from . import observers
from . import stoppers
from . import informers
from . import renderers

from tensortrade.env.generic import TradingEnv, MultySymbolTradingEnv
from tensortrade.env.generic.components.renderer import AggregateRenderer

import pandas as pd
import numpy as np
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.wallets import Wallet, Portfolio
from gymnasium.wrappers import EnvCompatibility
from pprint import pprint
import random

# turn off pandas SettingWithCopyWarning 
pd.set_option('mode.chained_assignment', None)

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
    reward_scheme = rewards.get(reward_scheme) if isinstance(reward_scheme, str) else reward_scheme

    action_scheme.portfolio = portfolio

    observer = observers.TensorTradeObserver(
        portfolio=portfolio,
        feed=feed,
        renderer_feed=kwargs.get("renderer_feed", None),
        window_size=window_size,
        min_periods=min_periods
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


    if kwargs["config"]["multy_symbol_env"] == True:
        environment = MultySymbolTradingEnv
        print("multy_symbol_env == True")
    else:
        environment = TradingEnv

    env = environment(
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        observer=observer,
        stopper=kwargs.get("stopper", stopper),
        informer=kwargs.get("informer", informers.TensorTradeInformer()),
        renderer=renderer,
        min_periods=min_periods,
        random_start_pct=random_start_pct,
        config=kwargs["config"]
    )
    return env

def get_episode_lengths(num_rows, max_episode_length):
    n = 1
    while int(num_rows/n) > max_episode_length:
        n+=1

    rem = num_rows%n
    ep_size = int(num_rows/n)
    ep_lengths = np.empty(n, dtype = int)
    ep_lengths.fill(ep_size)
    while rem > 0:
        ep_lengths[random.choice(np.where(ep_lengths == ep_size)[0])] += 1
        rem -= 1

    return ep_lengths

def make_sin_feed(symbol_name='AssetX', symbol_code = 0, length=1000):
    x = np.arange(0, 2*np.pi, 2*np.pi / (length + 1))
    y = 50*np.sin(3*x) + 100
    xy = pd.DataFrame(data=np.transpose([y]), index=x).assign(symbol=pd.Series(np.full(len(x), symbol_name)).values).assign(symbol_code=pd.Series(np.full(len(x), symbol_code)).values)
    xy.columns=['close', 'symbol', 'symbol_code']
    xy.index.name = "datetime"
    return xy

def make_flat_feed(symbol_name='AssetX', symbol_code = 0, length=1000, price_value=100):
    x = np.arange(0, 2*np.pi, 2*np.pi / (length + 1))
    y = np.full(np.shape(x), float(price_value))
    xy = pd.DataFrame(data=np.transpose([y]), index=x).assign(symbol=pd.Series(np.full(len(x), symbol_name)).values).assign(symbol_code=pd.Series(np.full(len(x), symbol_code)).values)
    xy.columns=['close', 'symbol', 'symbol_code']
    xy.index.name = "datetime"
    return xy

def make_sin_symbol(name, symbol_code=0, spread=0.01, commission=0.005, length=40):
    symbol = {'name': name,
               'spread': spread,
               'commission': commission
              }

    end_of_episode = pd.Series(np.full(length+1, False))
    symbol["feed"] = make_sin_feed(symbol["name"], symbol_code, length).assign(end_of_episode=end_of_episode.values)
    symbol["feed"]["end_of_episode"].iloc[-1] = True

    return symbol


def make_flat_symbol(name, symbol_code=0, spread=0.01, commission=0.005, length=40, price=100):
    symbol = {'name': name,
               'spread': spread,
               'commission': commission
              }

    end_of_episode = pd.Series(np.full(length+1, False))
    symbol["feed"] = make_flat_feed(symbol["name"], symbol_code, length, price).assign(end_of_episode=end_of_episode.values)
    symbol["feed"]["end_of_episode"].iloc[-1] = True
    return symbol

def make_synthetic_symbol(config):
    print(config)
    symbol = config
    end_of_episode = pd.Series(np.full(config["length"]+1, False))

    if config["process"] == 'sin':
        symbol["feed"] = make_sin_feed(symbol["name"], config["code"], config["length"]).assign(end_of_episode=end_of_episode.values)
    elif config["process"] == 'flat':
        symbol["feed"] = make_flat_feed(symbol["name"], config["code"], config["length"], config["price_value"]).assign(end_of_episode=end_of_episode.values)

    ep_lengths = get_episode_lengths(config["length"], config["max_episode_steps"])
    end_of_episode_index=0
    for i, l in enumerate(ep_lengths,0):
        end_of_episode_index += l
        # FIXME: next line produces SettingWithCopyWarning, maybe somebody will
        # be so nice to fix it
        symbol["feed"]["end_of_episode"].iloc[end_of_episode_index] = True
    return symbol


def get_wallets_volumes(wallets):
    volumes = []
    for w in wallets:
        balance = w.total_balance
        volumes.append(float(balance.size))
    return volumes

def is_end_of_episode(obs):
    return obs[-1][-1]


def create_multy_symbol_env(config):
    dataset = pd.concat([config["symbols"][i]["feed"] for i in range(len(config["symbols"]))])
    print(dataset.to_markdown())
    exchanges=[]
    wallets=[]
    # exchange_options = ExchangeOptions(commission=config["symbols"][-1]["commission"], spread=config["symbols"][-1]["spread"])
    exchange_options = ExchangeOptions(commission=config["symbols"][-1]["commission"], config=config)

    price = Stream.source(list(dataset["close"]), dtype="float").rename("USDT-ASSET")
    prices=[]
    for i,s in enumerate(config["symbols"],0):
        price=[]
        for j in range(len(config["symbols"])):
            if j == i:
                price.extend(config["symbols"][i]["feed"]["close"].values)
            else:
                price.extend(np.ones(len(config["symbols"][j]["feed"]["close"])))

        prices.append(Stream.source(price, dtype="float").rename(f"USDT-AST{i}"))

    exchange = Exchange('binance', service=execute_order, options=exchange_options)(*prices)
    USDT = Instrument("USDT", 2, "USD Tether")
    cash = Wallet(exchange, 1000 * USDT)  # This is the starting cash we are going to use
    wallets.append(cash)

    # create assets wallets
    for i,s in enumerate(config["symbols"],0):
        asset = Instrument(f'AST{i}', 5, s['name'])
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


    env.num_service_cols = 2
    env = EnvCompatibility(env)
    return env
