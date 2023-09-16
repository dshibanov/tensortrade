import pandas as pd
import numpy as np
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
import tensortrade.env.default as default
import ray.rllib.utils as rlutils
from gymnasium.wrappers import EnvCompatibility

def make_sin_feed(symbol_name='AssetX', symbol_code = 0, length=1000):
    x = np.arange(0, 2*np.pi, 2*np.pi / (length + 1))
    y = 50*np.sin(3*x) + 100
    xy = pd.DataFrame(data=np.transpose([y]), index=x).assign(symbol=pd.Series(np.full(len(x), symbol_name)).values).assign(symbol_code=pd.Series(np.full(len(x), symbol_code)).values)
    xy.columns=['close', 'symbol', 'symbol_code']
    xy.index.name = "datetime"
    return xy

def make_flat_feed(symbol_name='AssetX', symbol_code = 0, length=1000, price=100):
    x = np.arange(0, 2*np.pi, 2*np.pi / (length + 1))
    y = np.full(np.shape(x), float(price))
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
    reward_scheme = default.rewards.SimpleProfit(window_size=config["reward_window_size"])
    action_scheme = default.actions.MultySymbolBSH(config)

    env = default.create(
            feed=feed,
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            renderer=[],
            window_size=config["window_size"],
            max_allowed_loss=config["max_allowed_loss"],
            config=config)

    env = EnvCompatibility(env)
    return env

