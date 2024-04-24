import pandas as pd
import sys, os, time
sys.path.append(os.getcwd())
import pytest
import ta

import tensortrade.env.default as default

from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.instruments import USD, BTC, ETH, LTC
from tensortrade.oms.wallets import Portfolio, Wallet
from tensortrade.env.default.actions import ManagedRiskOrders
from tensortrade.env.default.rewards import SimpleProfit
from tensortrade.feed import DataFeed, Stream, NameSpace
from tensortrade.oms.services.execution.simulated import execute_order

import numpy as np
import ray
from ray import tune
from ray.tune.registry import register_env
from tensortrade.env.default import *
import copy
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray import air, tune
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.rllib.utils import check_env
from ray.rllib.algorithms.algorithm import Algorithm
from icecream import ic
from plot import plot_history, set_gruvbox

from ray.tune.schedulers import PopulationBasedTraining

register_env("multy_symbol_env", create_multy_symbol_env)

@pytest.fixture
def portfolio():

    df1 = pd.read_csv("tests/data/input/bitfinex_(BTC,ETH)USD_d.csv").tail(100)
    df1 = df1.rename({"Unnamed: 0": "date"}, axis=1)
    df1 = df1.set_index("date")

    df2 = pd.read_csv("tests/data/input/bitstamp_(BTC,ETH,LTC)USD_d.csv").tail(100)
    df2 = df2.rename({"Unnamed: 0": "date"}, axis=1)
    df2 = df2.set_index("date")

    ex1 = Exchange("bitfinex", service=execute_order)(
        Stream.source(list(df1['BTC:close']), dtype="float").rename("USD-BTC"),
        Stream.source(list(df1['ETH:close']), dtype="float").rename("USD-ETH")
    )

    ex2 = Exchange("binance", service=execute_order)(
        Stream.source(list(df2['BTC:close']), dtype="float").rename("USD-BTC"),
        Stream.source(list(df2['ETH:close']), dtype="float").rename("USD-ETH"),
        Stream.source(list(df2['LTC:close']), dtype="float").rename("USD-LTC")
    )

    p = Portfolio(USD, [
        Wallet(ex1, 10000 * USD),
        Wallet(ex1, 10 * BTC),
        Wallet(ex1, 5 * ETH),
        Wallet(ex2, 1000 * USD),
        Wallet(ex2, 5 * BTC),
        Wallet(ex2, 20 * ETH),
        Wallet(ex2, 3 * LTC),
    ])
    return p


def test_runs_with_external_feed_only(portfolio):

    df = pd.read_csv("tests/data/input/bitfinex_(BTC,ETH)USD_d.csv").tail(100)
    df = df.rename({"Unnamed: 0": "date"}, axis=1)
    df = df.set_index("date")

    bitfinex_btc = df.loc[:, [name.startswith("BTC") for name in df.columns]]
    bitfinex_eth = df.loc[:, [name.startswith("ETH") for name in df.columns]]

    ta.add_all_ta_features(
        bitfinex_btc,
        colprefix="BTC:",
        **{k: "BTC:" + k for k in ['open', 'high', 'low', 'close', 'volume']}
    )
    ta.add_all_ta_features(
        bitfinex_eth,
        colprefix="ETH:",
        **{k: "ETH:" + k for k in ['open', 'high', 'low', 'close', 'volume']}
    )

    streams = []
    with NameSpace("bitfinex"):
        for name in bitfinex_btc.columns:
            streams += [Stream.source(list(bitfinex_btc[name]), dtype="float").rename(name)]
        for name in bitfinex_eth.columns:
            streams += [Stream.source(list(bitfinex_eth[name]), dtype="float").rename(name)]

    feed = DataFeed(streams)

    action_scheme = ManagedRiskOrders()
    reward_scheme = SimpleProfit()

    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        window_size=50,
        enable_logger=False,
    )

    done = False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

    assert obs.shape[0] == 50


def test_runs_with_random_start(portfolio):

    df = pd.read_csv("tests/data/input/bitfinex_(BTC,ETH)USD_d.csv").tail(100)
    df = df.rename({"Unnamed: 0": "date"}, axis=1)
    df = df.set_index("date")

    bitfinex_btc = df.loc[:, [name.startswith("BTC") for name in df.columns]]
    bitfinex_eth = df.loc[:, [name.startswith("ETH") for name in df.columns]]

    ta.add_all_ta_features(
        bitfinex_btc,
        colprefix="BTC:",
        **{k: "BTC:" + k for k in ['open', 'high', 'low', 'close', 'volume']}
    )
    ta.add_all_ta_features(
        bitfinex_eth,
        colprefix="ETH:",
        **{k: "ETH:" + k for k in ['open', 'high', 'low', 'close', 'volume']}
    )

    streams = []
    with NameSpace("bitfinex"):
        for name in bitfinex_btc.columns:
            streams += [Stream.source(list(bitfinex_btc[name]), dtype="float").rename(name)]
        for name in bitfinex_eth.columns:
            streams += [Stream.source(list(bitfinex_eth[name]), dtype="float").rename(name)]

    feed = DataFeed(streams)

    action_scheme = ManagedRiskOrders()
    reward_scheme = SimpleProfit()

    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        window_size=50,
        enable_logger=False,
        random_start_pct=0.10,  # Randomly start within the first 10% of the sample
    )

    done = False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

    assert obs.shape[0] == 50


# @pytest.mark.skip()
def test_get_train_test_feed():

    config = {
              "max_episode_length": 27, # smaller or equal is ok, bigger is not,
              "min_episode_length": 10, # bigger or equal is ok, smaller is not
              "make_folds":True,
              "num_folds": 7,
              "symbols": make_symbols(1, 100, False),
              "cv_mode": 'proportional',
              "test_fold_index": 3,
              "reward_window_size": 1,
              "window_size": 2,
              "max_allowed_loss": 100,
              "use_force_sell": True,
              "multy_symbol_env": True
             }

    config = make_folds(config)

    all_lens = []
    for s in config["symbols"]:
        lens = get_episodes_lengths(s["feed"])
        all_lens = all_lens + lens

    assert min(all_lens) == 101

    print('ok >>>> ')
    train, test = get_train_test_feed(config)
    all_lens = get_episodes_lengths(train)
    all_lens += get_episodes_lengths(test)

    print(min(all_lens))
    print(max(all_lens))
    assert config["max_episode_length"] >= max(all_lens)
    assert config["min_episode_length"] <= min(all_lens)


    for s in config["symbols"]:
        last_episode_end=0
        assert len(s["folds"]) > 0
        for f in s["folds"]:
            episodes = s["episodes"][f[0]:f[1]]
            for e in episodes:
                if e[0] > 0:
                    assert last_episode_end == e[0]
                last_episode_end = e[1]

    train, test = get_train_test_feed(config)

    all_lens = []
    all_lens += get_episodes_lengths(train)
    print('lens of train ', all_lens)
    all_lens += get_episodes_lengths(test)
    print('all_lens train + test ', all_lens)

    assert config["max_episode_length"] >= max(all_lens)
    assert config["min_episode_length"] <= min(all_lens)


def test_end_episodes():
    num_symbols=5
    config = {
              "reward_window_size": 7,
              "symbols": make_symbols(num_symbols, 666, True),
              "window_size": 1,
              "max_allowed_loss":100,
              "multy_symbol_env": True,
              "use_force_sell": True,
              "make_folds": False,
              "test": False
             }

    dataset = pd.concat([config["symbols"][i]["feed"] for i in range(len(config["symbols"]))])
    env = create_multy_symbol_env(config)
    action = 0 # do nothing
    obs,_ = env.reset()
    # info = env.env.informer.info(env.env)
    info = get_info(env)
    track=[]
    done = False
    step = 0
    instruments=[]
    volumes=[]
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))
    observations=[np.append(obs[-1], np.append([action, info['net_worth']], volumes))]

    # test feed
    while done == False and step < 242:
        print('step ', step)
        # assert pytest.approx(obs[-1][0], 0.001) == dataset.iloc[step].close
        assert pytest.approx(obs[-1], 0.001) == dataset.iloc[step].close
        # assert pytest.approx(obs[-1][1], 0.001) == dataset.iloc[step].symbol_code
        # assert pytest.approx(obs[-1][2], 0.001) == dataset.iloc[step].end_of_episode
        #
        if step == 38:
            action = 1
        else:
            action = 0

        # if is_end_of_episode(obs) == True:
        if is_end_of_episode(env) == True:

            obs, reward, done, truncated, info = env.step(action)
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            for v in volumes[-(len(volumes)-1):]:
                assert v == 0

            # volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
            # print('volumes: ', volumes)
            # for v in volumes[-(len(volumes)-1):]:
            #     # print(v)
            #     assert v == 0
            step += 1
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        # row = np.append(obs[-1], np.append([action, info['net_worth']], volumes))
        row = np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], volumes))
        observations.append(row)
        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
    track = pd.DataFrame(observations)
    track.columns = np.append(['close',   'symbol_code',  'end_of_episode', 'action', 'net_worth'], instruments)
    print(track.to_markdown())
    return

def test_spread():
    print("_______test_spread_____")
    num_symbols=5
    symbols=[]
    for i in range(num_symbols):
        if i == 2:
            symbols.append(make_flat_symbol("AST"+str(i), i, commission=0, spread=1.13))
        elif i == 4:
            symbols.append(make_flat_symbol("AST"+str(i), i, commission=0, spread=3.66))
        else:
            symbols.append(make_flat_symbol("AST"+str(i), i, commission=0, spread=0.01))

    config = {
              # "symbols": make_symbols(num_symbols, 100, True),
              "symbols": symbols,
              "reward_window_size": 7,
              "window_size": 1,
              "max_allowed_loss":100,
              "multy_symbol_env": True,
              "use_force_sell": True,
              "make_folds": False,
              "test": False
             }
    exchange_options = ExchangeOptions(commission=config["symbols"][-1]["commission"],
                                       # spread=config["symbols"][-1]["spread"])
                                       config=config)

    dataset = pd.concat([config["symbols"][i]["feed"] for i in range(len(config["symbols"]))])
    env = create_multy_symbol_env(config)
    action = 0 # do nothing
    obs,_ = env.reset()
    info = get_info(env)

    track=[]
    done = False
    step = 0
    instruments=[]
    volumes=[]
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))
    observations=[np.append(obs[-1], np.append([action, info['net_worth']], volumes))]

    # test feed
    while done == False and step < 214:
        assert pytest.approx(obs[-1], 0.001) == dataset.iloc[step].close
        # assert pytest.approx(obs[-1][1], 0.001) == dataset.iloc[step].symbol_code
        # assert pytest.approx(obs[-1][2], 0.001) == dataset.iloc[step].end_of_episode

        if step == 0:
            action=0
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            assert volumes[0] == 0
            assert volumes[1] != 0
            assert volumes[1] == 9.999
        elif step == 1:
            action=1
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            assert volumes[0] != 0
            assert volumes[0] == 999.9
            assert volumes[1] == 0
        elif step == 5:
            action=0
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            assert volumes[0] == 0
            assert volumes[1] != 0
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        # row = np.append(obs[-1], np.append([action, info['net_worth']], volumes))
        row = np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], volumes))
        observations.append(row)
        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
    track = pd.DataFrame(observations)
    track.columns = np.append(['close',   'symbol_code',  'end_of_episode', 'action', 'net_worth'], instruments)
    print(track.to_markdown())
    return

@pytest.mark.skip()
def test_comission():
    # !!! comissions are not implemented yet !!!

    num_symbols=5
    symbols=[]
    for i in range(num_symbols):
        symbols.append(make_flat_symbol("Asset"+str(i), i, commission=0.005))

    config = {"symbols": symbols,
              "reward_window_size": 7,
              "window_size": 1,
              "max_allowed_loss":100,
              "multy_symbol_env": True
             }

    dataset = pd.concat([config["symbols"][i]["feed"] for i in range(len(config["symbols"]))])

    env = create_multy_symbol_env(config)
    action = 0 # do nothing
    obs,_ = env.reset()
    info = env.env.informer.info(env.env)

    track=[]
    done = False
    step = 0
    instruments=[]
    volumes=[]
    for w in env.env.action_scheme.portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))
    observations=[np.append(obs[-1], np.append([action, info['net_worth']], volumes))]

    # test feed
    while done == False and step < 2:
        assert pytest.approx(obs[-1][0], 0.001) == dataset.iloc[step].close
        assert pytest.approx(obs[-1][1], 0.001) == dataset.iloc[step].symbol_code
        assert pytest.approx(obs[-1][2], 0.001) == dataset.iloc[step].end_of_episode

        if step == 0:
            action=0
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
            assert volumes[0] == 0
            assert volumes[1] != 0
            assert volumes[1] == 9.95

        elif step == 1:
            action=1
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
            assert volumes[0] != 0
            assert volumes[0] == 990
            assert volumes[1] == 0

        row = np.append(obs[-1], np.append([action, info['net_worth']], volumes))
        observations.append(row)
        volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)

    track = pd.DataFrame(observations)
    track.columns = np.append(['close',   'symbol_code',  'end_of_episode', 'action', 'net_worth'], instruments)
    print(track.to_markdown())
    return

def test_multy_symbols():
    num_symbols=5
    symbols=[]
    for i in range(num_symbols):
        symbols.append(make_sin_symbol("Asset"+str(i), i))
    config = {
              "symbols": make_symbols(num_symbols, 666, True),
              "reward_window_size": 7,
              "window_size": 3,
              "max_allowed_loss":100,
              "multy_symbol_env": True,
              "use_force_sell": True,
              "num_service_cols" : 2,
              "make_folds": False,
              "test": False
             }

    dataset = pd.concat([config["symbols"][i]["feed"] for i in range(len(config["symbols"]))])
    env = create_multy_symbol_env(config)
    action = 0 # 0 - buy asset, 1 - sell asset
    obs,_ = env.reset()
    info = get_info(env)
    # env.render()

    done = False
    step = 0
    track=[]
    instruments=[]
    volumes=[]
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))
    observations=[np.append(obs[-1], np.append([action, info['net_worth']], volumes))]

    # test feed
    while done == False and step < 242:
        print('obs:: ', obs[-1], dataset.iloc[step].close)
        assert pytest.approx(obs[-1], 0.001) == dataset.iloc[step].close
        # assert pytest.approx(obs[-1][1], 0.001) == dataset.iloc[step].symbol_code
        # assert pytest.approx(obs[-1][2], 0.001) == dataset.iloc[step].end_of_episode


        if step == 10 or step == 20:
            action = 1

        if step == 15 or step == 25:
            action = 0

        print(step, ': ', obs, dataset.iloc[step])
        if is_end_of_episode(env) == True:

            obs, reward, done, truncated, info = env.step(action)
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            for v in volumes[-(len(volumes)-1):]:
                assert v == 0

            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            # for v in volumes[-(len(volumes)-1):]:
            #     print(v)
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        assert info["net_worth"] > 0
        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        # check here that volumes doesn't have nan's
        # print('loop by volumes...')
        for v in volumes:
            assert math.isnan(v) == False

        # row = np.append(obs[-1], np.append([action, info['net_worth']], volumes))
        row = np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], volumes))
        observations.append(row)
        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

    track = pd.DataFrame(observations)
    track.columns = np.append(['close',   'symbol_code',  'end_of_episode', 'action', 'net_worth'], instruments)
    # print(track.to_markdown())
    return



def test_multy_symbol_simple_trade_close_manually():
    # * close orders manually (by agent) before end_of_episode    

    num_symbols=5
    config = {
              "symbols": make_symbols(num_symbols, 666, True),
              "reward_window_size": 7,
              "window_size": 1,
              "max_allowed_loss":100,
              "multy_symbol_env": True,
              "use_force_sell": False,
              "num_service_cols" : 2,
              "make_folds": False,
              "test": False
             }

    dataset = pd.concat([config["symbols"][i]["feed"] for i in range(len(config["symbols"]))])

    env = create_multy_symbol_env(config)
    action = 0 # 0 - buy asset, 1 - sell asset
    obs,_ = env.reset()
    info = get_info(env)

    done = False
    step = 0
    track=[]
    instruments=[]
    volumes=[]
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))


    observations=[np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], volumes))]

    # test feed
    while done == False and step < 242:
        # print('obs ', obs)
        assert pytest.approx(obs[-1], 0.001) == dataset.iloc[step].close
        # assert pytest.approx(obs[-1][1], 0.001) == dataset.iloc[step].symbol_code
        # assert pytest.approx(obs[-1][2], 0.001) == dataset.iloc[step].end_of_episode

        if (step > 37 and step < 42) or (step >= 79 and step < 82) or (step >= 119 and step < 123) or (step >= 159 and step < 164):
            action = 1
        else:
            action = 0

        if is_end_of_episode(env) == True:
            obs, reward, done, truncated, info = env.step(action)
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            # for v in volumes[-(len(volumes)-1):]:
            #     assert v == 0
            step += 1
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        # assert net_worth value
        assert info["net_worth"] > 0
        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
        net_worth=volumes[0]
        for v in volumes[-(len(volumes)-1):]:
            net_worth += v*obs[-1]

        # print("info___: ", info)
        # np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), 
        # row = np.append(obs[-1], np.append([action, info['net_worth']], volumes))
        row = np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], volumes))
        observations.append(row)
        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

    track = pd.DataFrame(observations)
    track.columns = np.append(['close',   'symbol_code',  'end_of_episode', 'action', 'net_worth'], instruments)
    print(track.to_markdown())
    for index, row in track.iterrows():
        net_worth_test = sum([row[f"AST{i}"]*row["close"] for i in range(5)]) + row["USDT"]

        print(row["net_worth"], net_worth_test)
        assert pytest.approx(row["net_worth"], 0.001) == net_worth_test
    return

def test_multy_symbol_simple_use_force_sell():
    # * don't close orders manually (by agent) before end_of_episode
    #  use force_sell functionality for that purposes

    num_symbols=5
    config = {
              "symbols": make_symbols(num_symbols, 666, True),
              "reward_window_size": 7,
              "window_size": 1,
              "max_allowed_loss":100,
              "multy_symbol_env": True,
              "use_force_sell": True,
              "num_service_cols" : 2,
              "make_folds": False,
              "test": False
             }

    dataset = pd.concat([config["symbols"][i]["feed"] for i in range(len(config["symbols"]))])

    env = create_multy_symbol_env(config)
    action = 0 # 0 - buy asset, 1 - sell asset
    obs,_ = env.reset()
    info = get_info(env)

    done = False
    step = 0
    track=[]
    instruments=[]
    volumes=[]
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))
    observations=[np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], volumes))]

    # test feed
    while done == False and step < 3420:
        # assert pytest.approx(obs[-1][0], 0.001) == dataset.iloc[step].close
        # print('obs ', obs)
        assert pytest.approx(obs[-1], 0.001) == dataset.iloc[step].close
        # assert pytest.approx(obs[-1][1], 0.001) == dataset.iloc[step].symbol_code
        # print(f"step: {step} close {obs[-1]} {dataset.iloc[step].close} info {info['symbol_code']}  {info['end_of_episode']} dataset: {dataset.iloc[step].symbol_code} {dataset.iloc[step].end_of_episode}")
        assert pytest.approx(info["symbol_code"], 0.001) == dataset.iloc[step].symbol_code
        # assert pytest.approx(obs[-1][2], 0.001) == dataset.iloc[step].end_of_episode
        # print(' end of episode ', info["end_of_episode"], dataset.iloc[step].end_of_episode)
        # assert pytest.approx(info["end_of_episode"], 0.001) == dataset.iloc[step].end_of_episode

        if (step > 57 and step < 63): #  or (step >= 79 and step < 82) or (step >= 119 and step < 123) or (step >= 159 and step < 164):
            # sell 
            action = 1
        else:
            action = 0

        if is_end_of_episode(env) == True:
            obs, reward, done, truncated, info = env.step(action)
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            step += 1
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        # assert net_worth value
        assert info["net_worth"] > 0
        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        # check here that volumes doesn't have nan's
        # print('loop by volumes...')
        for v in volumes:
            assert math.isnan(v) == False

        net_worth=volumes[0]
        for v in volumes[-(len(volumes)-1):]:
            net_worth += v*obs[-1]

        # row = np.append(obs[-1], np.append([action, info['net_worth']], volumes))
        row = np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], volumes))
        observations.append(row)
        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

    track = pd.DataFrame(observations)
    track.columns = np.append(['close',   'symbol_code',  'end_of_episode', 'action', 'net_worth'], instruments)
    print(track.to_markdown())

    # check net_worth calc
    for index, row in track.iterrows():
        net_worth_test = sum([row[f"AST{i}"]*row["close"] for i in range(num_symbols)]) + row["USDT"]
        # print(' row net worth',row["net_worth"], net_worth_test)
        assert pytest.approx(row["net_worth"], 0.001) == net_worth_test

    return

def test_make_synthetic_symbol():
    print("_______test_make_synthetic_symbol_____")
    config = {"name": 'X',
              "spread": 0.001,
              "commission": 0.0001,
              "code": 0,
              "length": 31,
              "max_episode_steps": 11,
              # "max_episode_steps": 152,
              "process": 'flat',
              "price_value": 100}

    config["shatter_on_episode_on_creation"] = True
    s = make_synthetic_symbol(config)
    print(s["feed"])

    last_episode_start=0
    for i, value in enumerate(s["feed"].iterrows(), 0):
        index, row = value
        if row["end_of_episode"] == True:
            print("     > ", i)
            ep_length = i - last_episode_start
            last_episode_start = i+1
            assert ep_length <= config["max_episode_steps"]



@pytest.mark.skip()
# maybe this is useless now
def test_get_episode_lengths():
    result = get_episode_lengths(31, 10)
    # print(result, result.sum())
    assert result.sum() == 31
    result = get_episode_lengths(34, 10)
    # print(result, result.sum())
    assert result.sum() == 34
    result = get_episode_lengths(37, 10)
    # print(result, result.sum())
    assert result.sum() == 37
    result = get_episode_lengths(61, 13)
    # print(result, result.sum())
    assert result.sum() == 61
    result = get_episode_lengths(39, 20)
    # print(result, result.sum())
    assert result.sum() == 39

@pytest.mark.skip()
# it looks like after flattening np.shape(obs) != env.observer.observation_space.shape
def test_observation_shape():
    num_symbols=5
    config = {
              "symbols": make_symbols(num_symbols, 666, False),
              "reward_window_size": 7,
              "window_size": 3,
              "max_allowed_loss":100,
              "multy_symbol_env": True,
              "use_force_sell": True,
              "num_service_cols" : 2,
              "make_folds": False,
              "test": False
             }

    dataset = pd.concat([config["symbols"][i]["feed"] for i in range(len(config["symbols"]))])
    env = create_multy_symbol_env(config)
    action = 0 # 0 - buy asset, 1 - sell asset
    obs,_ = env.reset()

    # print(obs, np.shape(obs), env.env.observer.observation_space)
    print(f'reset obs: {obs} type(obs): {type(obs)} obs.shape: {np.shape(obs)} obs_shape {get_observer(env).observation_space.shape}')
    # assert np.shape(obs) == env.env.observer.observation_space.shape
    assert np.shape(obs) == get_observer(env).observation_space.shape
    obs, reward, done, truncated, info = env.step(0)
    # print(obs, np.shape(obs), env.env.observer.observation_space)
    print(f'step obs: {obs} type(obs): {type(obs)} obs.shape: {np.shape(obs)} ') #, env.env.observer.observation_space)
    assert np.shape(obs) == get_observer(env).observation_space.shape


def test_obs_space_of():
    import gymnasium as gym
    # env = gym.make("LunarLander-v2", render_mode="human")
    env = gym.make('CartPole-v1')
    observation, info = env.reset(seed=42)
    for _ in range(5):
        action = env.action_space.sample()  # this is where you would insert your policy
        obs, reward, terminated, truncated, info = env.step(action)

        print(f'step obs: {obs} type(obs): {type(obs)} obs.shape: {np.shape(obs)} ') #, env.env.observer.observation_space)
        # assert np.shape(obs) == env.env.observer.observation_space.shape
        # obs, reward, done, truncated, info = env.step(0)
        # assert np.shape(obs) == env.env.observer.observation_space.shape

        if terminated or truncated:
            observation, info = env.reset()
            print(f'reset obs: {obs} type(obs): {type(obs)} obs.shape: {np.shape(obs)} ') #, env.env.observer.observation_space)
            # print('reset ', obs, np.shape(obs)) #, env.env.observer.observation_space)

    env.close()

def test_get_dataset():
    num_symbols=5
    config = {
              "max_episode_length": 27, # smaller or equal is ok, bigger is not,
              "min_episode_length": 10, # bigger or equal is ok, smaller is not
              "symbols": make_symbols(num_symbols, 666, False),
              "reward_window_size": 7,
              "window_size": 3,
              "max_allowed_loss":100,
              "multy_symbol_env": True,
              "use_force_sell": True,
              "num_service_cols" : 2,
              "make_folds": False, 
             }


    # no folds
    r = get_dataset(config)
    # episodes_lengths = get_episodes_lengths(r[0])
    episodes_lengths = get_episodes_lengths(r)
    print(f'no folds | episodes_lengths {episodes_lengths} \n min {min(episodes_lengths)} max {max(episodes_lengths)}')
    # assert max(episodes_lengths) <= config["max_episode_length"]
    # assert min(episodes_lengths) >= config["min_episode_length"]
    config["make_folds"] = True
    config["test"] = True
    config["test_fold_index"] = 3
    config["num_folds"] = 7


    # folds test 
    config = make_folds(config)
    r = get_dataset(config)
    # episodes_lengths = get_episodes_lengths(r[1])
    episodes_lengths = get_episodes_lengths(r)
    print(f'folds test | episodes_lengths {episodes_lengths} \n min {min(episodes_lengths)} max {max(episodes_lengths)}')
    assert max(episodes_lengths) <= config["max_episode_length"]
    assert min(episodes_lengths) >= config["min_episode_length"]
    assert max(episodes_lengths) - min(episodes_lengths) <= 1

    # folds train
    config["test"] = False
    r = get_dataset(config)
    # episodes_lengths = get_episodes_lengths(r[0])
    episodes_lengths = get_episodes_lengths(r)
    print(f'episodes {episodes_lengths} \n min {min(episodes_lengths)} max {max(episodes_lengths)}')
    assert max(episodes_lengths) <= config["max_episode_length"]
    assert min(episodes_lengths) >= config["min_episode_length"]
    assert max(episodes_lengths) - min(episodes_lengths) <= 1

    # 2
    num_symbols=2
    config = {
              "max_episode_length": 27, # smaller or equal is ok, bigger is not,
              "min_episode_length": 10, # bigger or equal is ok, smaller is not
              # "symbols": make_symbols(num_symbols, 666, False),
              "symbols": make_symbols(2, 146),
              "reward_window_size": 7,
              "window_size": 3,
              "max_allowed_loss":100,
              "multy_symbol_env": True,
              "use_force_sell": True,
              "num_service_cols" : 2,
              "make_folds": False,
             }


    # no folds
    r = get_dataset(config)
    episodes_lengths = get_episodes_lengths(r)
    print(f'no folds | episodes_lengths {episodes_lengths} \n min {min(episodes_lengths)} max {max(episodes_lengths)}')
    # assert max(episodes_lengths) <= config["max_episode_length"]
    # assert min(episodes_lengths) >= config["min_episode_length"]
    config["make_folds"] = True
    config["test"] = True
    config["test_fold_index"] = 3
    config["num_folds"] = 7


    # folds test 
    config = make_folds(config)
    r = get_dataset(config)
    # episodes_lengths = get_episodes_lengths(r[1])
    episodes_lengths = get_episodes_lengths(r)
    print(f'folds test | episodes_lengths {episodes_lengths} \n min {min(episodes_lengths)} max {max(episodes_lengths)}')
    assert max(episodes_lengths) <= config["max_episode_length"]
    # assert min(episodes_lengths) >= config["min_episode_length"]
    assert max(episodes_lengths) - min(episodes_lengths) <= 1

    # folds train
    config["test"] = False
    r = get_dataset(config)
    # episodes_lengths = get_episodes_lengths(r[0])
    episodes_lengths = get_episodes_lengths(r)
    print(f'episodes {episodes_lengths} \n min {min(episodes_lengths)} max {max(episodes_lengths)}')
    assert max(episodes_lengths) <= config["max_episode_length"]
    # assert min(episodes_lengths) >= config["min_episode_length"]
    assert max(episodes_lengths) - min(episodes_lengths) <= 1



def eval_fold(params):
    score = 2
    framework = 'torch'
    # current_config = set_params(config, params)
    current_config = copy.deepcopy(params)#.copy()
    # print('current_config: feed ', current_config["symbols"][0]["feed"])
    ic(current_config["symbols"][0]["feed"])
    symbols = current_config["symbols"]
    test_fold_index = current_config["test_fold_index"]
    # print('test_fold_index ', test_fold_index)
    ic(test_fold_index)

    eval_config = current_config
    eval_config["test"] = True

    config = (
        DQNConfig()
        # .environment(SimpleCorridor, env_config={"corridor_length": 10})
        # .environment(env="multy_symbol_env", env_config=config)
        .environment(env="multy_symbol_env", env_config=current_config)
        # Training rollouts will be collected using just the learner
        # process, but evaluation will be done in parallel with two
        # workers. Hence, this run will use 3 CPUs total (1 for the
        # learner + 2 more for evaluation workers).
        # .rollouts(num_rollout_workers=0)
        .evaluation(
        #     evaluation_num_workers=2,
        #     # Enable evaluation, once per training iteration.
        #     evaluation_interval=1,
        #     # Run 10 episodes each time evaluation runs (OR "auto" if parallel to
        #     # training).
        #     # evaluation_duration="auto" if args.evaluation_parallel_to_training else 10,
        #     evaluation_duration="auto" if evaluation_parallel_to_training else 10,
        #     # Evaluate parallelly to training.
        #     # evaluation_parallel_to_training=args.evaluation_parallel_to_training,
        #     evaluation_parallel_to_training=evaluation_parallel_to_training,
        #     # evaluation_config=PGConfig.overrides(
            # evaluation_config=DQNConfig.overrides(env_config={"test": True}, explore=True
            evaluation_config=DQNConfig.overrides(env_config=eval_config
        #         env_config={
        #             # Evaluate using LONGER corridor than trained on.
        #             "corridor_length": 5,
        #         },
            ),
            # custom_evaluation_function=eval_fn,
        )
        # .framework(args.framework)
        .framework(framework)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )


    stop = {
        # "training_iteration": args.stop_iters,
        # "timesteps_total": args.stop_timesteps,
        # "episode_reward_mean": args.stop_reward,
        "training_iteration": 5,
        # "timesteps_total": 2000,
        "episode_reward_mean": 15,
    }

    codict = config.to_dict()
    topo = shape_to_topology2(codict["env_config"]["nnt_a"], codict["env_config"]["nnt_btoa"], codict["env_config"]["nnt_lh_ratio"])
    ic(topo)
    codict["model"]["fcnet_hiddens"] = topo
    ic(codict)
    tuner = tune.Tuner(
        # "PG",
        "DQN",
        # param_space=config.to_dict(),
        # param_space=config.to_dict(),
        param_space=codict,
        run_config=air.RunConfig(stop=stop, verbose=1, checkpoint_config=CheckpointConfig(checkpoint_frequency=2)),
        # run_config=RunConfig(stop=TrialPlateauStopper(metric="score"))
        # run_config=RunConfig(stop=TrialPlateauStopper(metric="reward"),
        #                      # checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10),
        #                      checkpoint_config=CheckpointConfig(checkpoint_frequency=3),
        #                      verbose=2)
    )

    # results = tuner.fit()
    results = tuner.fit()#.get_dataframe(filter_metric="score", filter_mode="min")
    best_result = results.get_best_result(metric="loss", mode="min")
    best_checkpoint = best_result.checkpoint
    algo = Algorithm.from_checkpoint(best_checkpoint)
    evaluated = algo.evaluate()

    ic(best_checkpoint)
    ic(results)


    # 2 make validation here
        # get_policy from checkpoint
        #
        #
    test_config = current_config
    test_config["test"] = True
    # test_env = create_multy_symbol_env(test_config)

    ic(best_checkpoint.path)
    # track, trades = simulate(test_config, '/home/happycosmonaut/ray_results/DQN_2023-11-20_17-28-33/DQN_multy_symbol_env_8f027_00000_0_2023-11-20_17-28-34/checkpoint_000002')
    track, trades = simulate(test_config, best_checkpoint.path)
    # get simulation result here
    # compare them to inner simulation/test results
    # should report score here 
    # return np.mean(track["reward"].dropna())
    return {"score": np.mean(track["reward"].dropna())}
# def get_action_scheme(env):
#     return env.env.env.env.action_scheme


def eval(path_to_checkpoint=''):
    restored_algo = Algorithm.from_checkpoint(path_to_checkpoint)
    conf = restored_algo.config["env_config"]
    conf["test"] = False
    model = restored_algo.config.model

    policy1 = restored_algo.get_policy(policy_id="default_policy")
    env = create_multy_symbol_env(conf)
    get_action_scheme(env).portfolio.exchanges[0].options.max_trade_size = 100066600000

    print('max_trade_size ', get_action_scheme(env).portfolio.exchanges[0].options.max_trade_size)
    obs,_ = env.reset()

    

    info = get_info(env)
    action = policy1.compute_single_action(obs)[0]
    print("action: ", action)
    done = False
    step = 0
    volumes=[]
    instruments=[]
    # for w in env.env.action_scheme.portfolio.wallets:
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))

    obss=[]
    reward=float('NaN')
    print('>>>>>>>>>>> START SIMULATION >>>>>>>>>>>>')
    observations=[]
    while done == False:
        wallets = [w.total_balance for w in get_action_scheme(env).portfolio.wallets]
        print(f' step {step}, close {obs[-1]} action {action} info {info}, wallets {wallets}')
        non_zero_wallets=0
        for w in wallets:
            if w != 0:
                non_zero_wallets+=1
        assert non_zero_wallets == 1


        obss.append(obs)
        action = policy1.compute_single_action(obs)[0]
        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
        row = np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], np.append(volumes, [reward])))
        observations.append(row)
        obs, reward, done, truncated, info = env.step(action)
        step += 1



    track = pd.DataFrame(observations)
    track.columns = ['close', 'symbol_code',  'end_of_episode', 'action', 'net_worth'] + instruments + ['reward']
    print(track.to_markdown())

def test_eval_fold():
    config = {
              "max_episode_length": 15, # smaller is ok
              "min_episode_length": 5, # bigger is ok, smaller is not
              "make_folds":True,
              "num_folds": 5,
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

    # action = 0 # 0 - buy asset, 1 - sell asset
    config["nnt_a"] = 256
    config["nnt_btoa"] = 0.3
    config["nnt_lh_ratio"] = 0.005
    config = make_folds(config)
    # print('config ', config)
    ic(config)
    for s in config["symbols"]:
        print(s["feed"].to_markdown())


    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        train_feed, test_feed = get_train_test_feed(config)
        # print('FEED ', len(train_feed), len(test_feed))
        ic('FEED ', len(train_feed), len(test_feed))

    # pprint({key: value for key, value in config.items() if key not in ["symbols"]})
    num_cpus = 2
    local_mode = True
    evaluation_parallel_to_training = True
    ray.init(num_cpus=num_cpus or None, local_mode=local_mode)


    score = eval_fold(config)
    return score



def test_create_ms_env():

    config = {
              # "max_episode_length": 25, # smaller is ok
              "max_episode_length": 15, # smaller is ok
              "min_episode_length": 5, # bigger is ok, smaller is not
              "make_folds":True,
              "num_folds": 3,
              # "symbols": make_symbols(5, 410),
              "symbols": make_symbols(2, 46),
              "cv_mode": 'proportional',
              "test_fold_index": 1,
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

    # print('no folds')
    # config["make_folds"] = False
    # config = make_folds(config)
    # print(get_dataset(config)[0].to_markdown())
    # return

    # some mechanisms of creating feed are stochastic
    # so we need to repeat test many times to make sure that there are no bugs
    # for i in range(20):
    #     config = make_folds(config)

    #     print('train')
    #     env = create_multy_symbol_env(config)

    #     # TODO:
    #         # check price_streams length and values
    #     dataset = get_dataset(config)
    #     print(f'len(dataset) ', len(dataset))
    #     print(get_action_scheme(env).portfolio.exchanges[-1])
    #     print(len(get_action_scheme(env).portfolio.exchanges[-1]._price_streams['USDT/AST0'].iterable))
    #     print(len(get_action_scheme(env).portfolio.exchanges[-1]._price_streams['USDT/AST1'].iterable))
    #     streams = get_action_scheme(env).portfolio.exchanges[-1]._price_streams
    #     for k in streams:
    #         print(k)
    #         print('len_iterable ', len(streams[k].iterable))
    #         assert len(streams[k].iterable) == len(dataset)


    # for i in range(20):
    #     print('test')
    #     config = make_folds(config)
    #     config["test"] = True
    #     env = create_multy_symbol_env(config)
    #     dataset = get_dataset(config)
    #     print(f'len(dataset) ', len(dataset))
    #     streams = get_action_scheme(env).portfolio.exchanges[-1]._price_streams
    #     for k in streams:
    #         print(k)
    #         print('len_iterable ', len(streams[k].iterable))
    #         assert len(streams[k].iterable) == len(dataset)


    for i in range(20):
        config = {
                  # "max_episode_length": 25, # smaller is ok
                  "max_episode_length": 15, # smaller is ok
                  "min_episode_length": 5, # bigger is ok, smaller is not
                  "make_folds":True,
                  "num_folds": 3,
                  # "symbols": make_symbols(5, 410),
                  "symbols": make_symbols(7, 146),
                  "cv_mode": 'proportional',
                  "test_fold_index": 1,
                  "reward_window_size": 1,
                  "window_size": 2,
                  "max_allowed_loss": 0.9,
                  "use_force_sell": True,
                  "multy_symbol_env": True,
                  "test": True
                 }

        # print('test')
        for s in config["symbols"]:
            # print(s["feed"].to_markdown())
            lengths = get_episodes_lengths(s["feed"])
            print(f'before make_folds {lengths=}')
            assert min(lengths) > 3

        config = make_folds(config)

        for s in config["symbols"]:
            # print(s["feed"].to_markdown())
            lengths = get_episodes_lengths(s["feed"])
            print(f'before make_folds {lengths=}')
            assert min(lengths) > 3

        # check s["feed"]
        # for s in config["symbols"]:
        #     print(s["feed"].to_markdown())
        #     lengths = get_episodes_lengths(s["feed"])
        #     print(f'{lengths=}')

        # return
        # config["test"] = True
        # dataset = get_dataset(copyconf)
        dataset = get_dataset(config)
        lengths = get_episodes_lengths(dataset)
        print(f'{lengths=}')
        assert min(lengths) > 3

        config["test"] = True
        copyconf = copy.deepcopy(config)
        env = create_multy_symbol_env(config)
        dataset = get_dataset(copyconf)
        lengths = get_episodes_lengths(dataset)
        print(f'{lengths=}')
        assert min(lengths) > 3
        # return
        print(f'len(dataset) ', len(dataset))
        streams = get_action_scheme(env).portfolio.exchanges[-1]._price_streams
        for k in streams:
            print(k)
            print('len_iterable ', len(streams[k].iterable))
            assert len(streams[k].iterable) == len(dataset)

    # return
    # different length of feed of symbols
    #


def test_env_different_symbol_lengths():
    lengths = [133, 400, 300, 1200, 45]

    for i in range(50):
        symbols=[]
        for i in range(len(lengths)):
            spread = 0.01
            commission=0
            if i == 2:
                commission=0
                spread=1.13
            elif i == 4:
                commission=0
                spread=3.66

            config = {
                    "name": "AST"+str(i),
                    "spread": spread,
                    "commission": commission,
                    "code": i,
                    "length": lengths[i],
                    "max_episode_steps": 11,
                    # "max_episode_steps": 152,
                    # "process": 'flat',
                    "process": 'sin',
                    "price_value": 100,
                    "shatter_on_episode_on_creation": False
                    }

            symbols.append(make_synthetic_symbol(config))


        print('check lengths')
        for i,s in enumerate(symbols):
            print(len(s["feed"]), lengths[i])
            assert len(s["feed"]) == lengths[i]+1

        config = {
                  # "max_episode_length": 25, # smaller is ok
                  "max_episode_length": 15, # smaller is ok
                  "min_episode_length": 5, # bigger is ok, smaller is not
                  "make_folds":True,
                  "num_folds": 3,
                  # "symbols": make_symbols(5, 410),
                  "symbols": symbols,
                  "cv_mode": 'proportional',
                  "test_fold_index": 1,
                  "reward_window_size": 1,
                  "window_size": 2,
                  "max_allowed_loss": 0.9,
                  "use_force_sell": True,
                  "multy_symbol_env": True,
                  "test": False
                 }

        print('test')
        config = make_folds(config)
        config["test"] = True
        env = create_multy_symbol_env(config)
        dataset = get_dataset(config)
        print(get_episodes_lengths(dataset))
        print(f'len(dataset) ', len(dataset))

        streams = get_action_scheme(env).portfolio.exchanges[-1]._price_streams
        for k in streams:
            print(k)
            print('len_iterable ', len(streams[k].iterable))
            assert len(streams[k].iterable) == len(dataset)

def printconf(conf, print_data=False):
    ic('>>> ')
    pdict={}
    for k in conf:
        if k == 'symbols':
            if print_data == True:
                pdict[k] = conf[k]
        else:
            pdict[k] = conf[k]
    ic(pdict)


# def simulate(env, path_to_checkpoint=''):
def simulate(env_config, path_to_checkpoint=''):
    ic.disable()
    restored_algo = Algorithm.from_checkpoint(path_to_checkpoint)
    conf = restored_algo.config["env_config"]
    model = restored_algo.config.model
    policy1 = restored_algo.get_policy(policy_id="default_policy")
    # policy1.export_model("my_nn_model", onnx=11)
    # return

    # ic.enable()
    printconf(conf)
    env_config["test_fold_index"] = 1
    conf.update(env_config)
    # printconf(conf)
    # return
    # printconf(env_config)
    # printconf(conf)
    # return

    env = default.create_multy_symbol_env(conf)
    # env = default.create_multy_symbol_env(env_config)
    get_action_scheme(env).portfolio.exchanges[0].options.max_trade_size = 100000000000

    obs, infu = env.reset()
    info = get_info(env)
    action = policy1.compute_single_action(obs)[0]
    print("action: ", action)
    done = False
    step = 0
    volumes=[]
    instruments=[]
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))

    obss=[]
    reward=float('NaN')

    ic('>>>>>>>>>>> START SIMULATION >>>>>>>>>>>>')
    observations=[]
    while done == False:
        wallets = [w.total_balance for w in get_action_scheme(env).portfolio.wallets]
        ic(f' step {step}, close {obs[-1]} action {action} info {info}, wallets {wallets}')
        non_zero_wallets=0
        for w in wallets:
            if w != 0:
                non_zero_wallets+=1
        assert non_zero_wallets == 1

        obss.append(obs)
        action = policy1.compute_single_action(obs)[0]
        # print(f"step {step}, action {action}, info {info} ")
        volumes = default.get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
        row = np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], np.append(volumes, [reward])))
        # row = np.append(obs[-1], np.append([action, info['net_worth']], np.append(volumes, [reward])))
        observations.append(row)
        # if info["symbol_code"] == 1:
        #     print(':1')
        obs, reward, done, truncated, info = env.step(action)
        step += 1



    track = pd.DataFrame(observations)
    track.columns = ['close', 'symbol_code',  'end_of_episode', 'action', 'net_worth'] + instruments + ['reward']
    print(track.to_markdown())
    return track, get_action_scheme(env).broker.trades


def test_simulate():
    config = {
              "max_episode_length": 15, # smaller is ok
              "min_episode_length": 5, # bigger is ok, smaller is not
              "make_folds":True,
              "num_folds": 5,
              # "symbols": make_symbols(5, 410),
              "symbols": make_symbols(2, 160),
              "cv_mode": 'proportional',
              "test_fold_index": 3,
              "reward_window_size": 1,
              "window_size": 2,
              "max_allowed_loss": 0.9,
              "use_force_sell": True,
              "multy_symbol_env": True,
              "test": True
             }

    config["nn_topology_a"] = 7
    config["nn_topology_b_to_a_ratio"] = 0.3
    config["nn_topology_c_to_b_ratio"] = 0.7
    config["nn_topology_h_to_l_ratio"] = 2

    config = make_folds(config)

    dataset = get_dataset(config) # [ 0 if config["test"] == False else 1]
    print(dataset.to_markdown())
    # for i in range(2):
    #     print(i)
    #     d = dataset.loc[dataset["symbol_code"]==i]
    #     # print(d.to_markdown())
    #     print(f'symbol_code {i}')
    #     print(d["close"])
    # return
    # r = simulate(create_multy_symbol_env(config), '/home/happycosmonaut/ray_results/DQN_2023-11-20_17-28-33/DQN_multy_symbol_env_8f027_00000_0_2023-11-20_17-28-34/checkpoint_000002')
    track, trades = simulate(config, '/home/happycosmonaut/ray_results/DQN_2023-11-20_17-28-33/DQN_multy_symbol_env_8f027_00000_0_2023-11-20_17-28-34/checkpoint_000002')
    plot_history(track, trades, info='test_fold_simulation')
    # ic(track, type(track))

    # return
    test_config = copy.deepcopy(config)
    test_config["test"] = False
    track, trades = simulate(test_config, '/home/happycosmonaut/ray_results/DQN_2023-11-20_17-28-33/DQN_multy_symbol_env_8f027_00000_0_2023-11-20_17-28-34/checkpoint_000002')
    plot_history(track, trades, info='train_folds_simulation')

def test_get_dataset2():

    for i in range(50):
        config = {
                  # "max_episode_length": 25, # smaller is ok
                  "max_episode_length": 15, # smaller is ok
                  "min_episode_length": 5, # bigger is ok, smaller is not
                  "make_folds":True,
                  "num_folds": 3,
                  # "symbols": make_symbols(5, 410),
                  "symbols": make_symbols(7, 146),
                  "cv_mode": 'proportional',
                  "test_fold_index": 1,
                  "reward_window_size": 1,
                  "window_size": 2,
                  "max_allowed_loss": 0.9,
                  "use_force_sell": True,
                  "multy_symbol_env": True,
                  "test": False
                 }

        # print('test')
        for s in config["symbols"]:
            # print(s["feed"].to_markdown())
            lengths = get_episodes_lengths(s["feed"])
            print(f'before make_folds {lengths=}')
            assert min(lengths) > 3

        config = make_folds(config)

        for s in config["symbols"]:
            # print(s["feed"].to_markdown())
            lengths = get_episodes_lengths(s["feed"])
            print(f'before make_folds {lengths=}')
            assert min(lengths) > 3

def get_cv_scores(config):
    num_folds = config["num_folds"]
    config = make_folds(config)
    config["test_fold_index"] = tune.grid_search(np.arange(0,num_folds))

    num_iterations=config.get('num_train_iterations', 2)
    # stop = {
    #     # "training_iteration": args.stop_iters,
    #     # "timesteps_total": args.stop_timesteps,
    #     # "episode_reward_mean": args.stop_reward,
    #     "training_iteration": num_iterations,
    #     # "timesteps_total": 2000,
    #     # "episode_reward_mean": 15,
    #     "score": 57
    # }
    tuner = tune.Tuner(
        eval_fold,
        param_space=config,

        # run_config=air.RunConfig(stop=stop, verbose=1, checkpoint_config=CheckpointConfig(checkpoint_frequency=2)),
        # run_config=air.RunConfig(stop=stop, verbose=1),
        run_config=air.RunConfig(verbose=1),
        # run_config=RunConfig(stop=TrialPlateauStopper(metric="score"))
        # run_config=RunConfig(stop=TrialPlateauStopper(metric="reward"),
        #                      # checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10),
        #                      checkpoint_config=CheckpointConfig(checkpoint_frequency=3),
        #                      verbose=2)
    )
    # results = tuner.fit().get_best_result(metric="score", mode="min")
    # results = tuner.fit().get_dataframe(filter_metric="score", filter_mode="min")
    results = tuner.fit()

    df = results.get_dataframe(filter_metric="score", filter_mode="min")
    # print('this is results.. ')
    # print(type(results))
    # print(results)

    # ic(df)
    # cv_score = df["score"]
    # print(f'cv_score {cv_score}')
    # return cv_score
    return df


def test_get_cv_score():
    ray.shutdown()
    num_cpus = 2
    local_mode = True
    framework = 'torch'
    evaluation_parallel_to_training = True
    # ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)
    ray.init(num_cpus=num_cpus or None, local_mode=local_mode)

    config = {
              "max_episode_length": 50, # bigger is ok, smaller is not
              "min_episode_length": 5, # bigger is ok, smaller is not
              "make_folds":True,
              "num_folds": 3,
              "symbols": make_symbols(5, 41),
              # "symbols": make_symbols(2, 40),

              "cv_mode": 'proportional',
              "test_fold_index": 1,
              "reward_window_size": 1,
              "window_size": 2,
              "max_allowed_loss": 100,
              "use_force_sell": True,
              "multy_symbol_env": True,
              "num_train_iterations": 15,
              "test": False,
              "nnt_a": 34,
              "nnt_btoa": 0.3,
              "nnt_lh_ratio": 0.005
             }

    scores = get_cv_scores(config)
    # ic(f'we got scores {scores}')
    print(scores)

def hpo(config):

    def evaluate(config):
        scores = get_cv_scores(config)
        return {"score":scores["score"].mean(), "df": scores}

    num_folds = config["num_folds"]
    config = make_folds(config)

    # num_iterations=config.get('num_train_iterations', 2)
    # stop = {
    #     # "training_iteration": args.stop_iters,
    #     # "timesteps_total": args.stop_timesteps,
    #     # "episode_reward_mean": args.stop_reward,
    #     "training_iteration": num_iterations,
    #     # "timesteps_total": 2000,
    #     # "episode_reward_mean": 15,
    #     "score": 57
    # }
    tuner = tune.Tuner(
        evaluate,
        # param_space=config.to_dict(),
        # param_space=config,
        # param_space={"test_fold_index": tune.grid_search(np.arange(0,num_folds)),  "num_folds": 7},
        param_space=config,

        tune_config=tune.TuneConfig(
            # metric="episode_reward_mean",
            metric="score",
            mode="max",
            # search_alg=algo,
            num_samples=2
        ),
        # tune_config=... ,
        # run_config=air.RunConfig(stop=stop, verbose=1, checkpoint_config=CheckpointConfig(checkpoint_frequency=2)),
        # run_config=air.RunConfig(stop=stop, verbose=1),
        # run_config=RunConfig(stop=TrialPlateauStopper(metric="score"))
        # run_config=RunConfig(stop=TrialPlateauStopper(metric="reward"),
        #                      # checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10),
        #                      checkpoint_config=CheckpointConfig(checkpoint_frequency=3),
        #                      verbose=2)
    )
    # results = tuner.fit().get_best_result(metric="score", mode="min")
    # results = tuner.fit().get_dataframe(filter_metric="score", filter_mode="min")
    results = tuner.fit()
    df = results.get_dataframe(filter_metric="score", filter_mode="min")
    print(df)
    return df


def test_hpo():

    ray.shutdown()
    num_cpus = 2
    local_mode = True
    framework = 'torch'
    evaluation_parallel_to_training = True
    # ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)
    ray.init(num_cpus=num_cpus or None, local_mode=local_mode)

    config = {"max_episode_length": 50, # bigger is ok, smaller is not
              'min_episode_length': 10,
              "make_folds":True,
              "num_folds": 3,
              "symbols": make_symbols(2, 100),
              "cv_mode": 'proportional',
              "test_fold_index": 2,
              "reward_window_size": 1,
              "window_size": 2,
              "max_allowed_loss": 100,
              "use_force_sell": True,
              "multy_symbol_env": True,
              "test": False,
              "num_train_iterations": 7,
              "nnt_a": tune.grid_search(np.arange(32,256)),
              "nnt_btoa": tune.grid_search(np.arange(0.2, 1.3)),
              "nnt_lh_ratio": tune.grid_search(np.arange(0.002, 0.01))
             }

    # config["test_fold_index"] = tune.grid_search(np.arange(0,num_folds))
    # dataset = pd.concat([config["symbols"][i]["feed"] for i in range(len(config["symbols"]))])
    # env = create_multy_symbol_env(config)
    # action = 0 # 0 - buy asset, 1 - sell asset
    # config["nn_topology_a"] = 7
    # config["nn_topology_b_to_a_ratio"] = 0.3
    # config["nn_topology_c_to_b_ratio"] = 0.7
    # config["nn_topology_h_to_l_ratio"] = 2
    # env = default.create_multy_symbol_env(config)

    hpo(config)

def test_mlflow():
    import mlflow
    import tempfile
    from ray import train, tune
    from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow

    ic('mlflow imported')
    smoke_test = True
    if smoke_test:
        mlflow_tracking_uri = os.path.join(tempfile.gettempdir(), "mlruns")
    else:
        mlflow_tracking_uri = "<MLFLOW_TRACKING_URI>"

    config = {"max_episode_length": 50, # bigger is ok, smaller is not
              'min_episode_length': 10,
              "make_folds":True,
              "num_folds": 5,
              "symbols": make_symbols(2, 100),
              "cv_mode": 'proportional',
              "test_fold_index": 3,
              "reward_window_size": 1,
              "window_size": 2,
              "max_allowed_loss": 100,
              "use_force_sell": True,
              "multy_symbol_env": True,
              "test": False,
              "nn_topology_a": tune.grid_search(np.arange(2,5)),
              "nn_topology_b_to_a_ratio" : tune.grid_search(np.arange(0.1,0.5)),
              "nn_topology_c_to_b_ratio" : tune.grid_search(np.arange(0.2,1)),
              "nn_topology_h_to_l_ratio" : tune.grid_search(np.arange(1,3))
             }

    def evaluate(config):
        scores = get_cv_scores(config)
        # train.report({"iterations": step, "mean_loss": intermediate_score})
        # scores = [1,2,3,4,4,5,6,77]
        # time.sleep(0.1)
        # return {"score":np.mean(scores), "scores": scores}
        return {"score": scores.mean(), "scores": scores}

    num_folds = config["num_folds"]
    config = make_folds(config)


    num_iterations=config.get('num_train_iterations', 2)
    stop = {
        # "training_iteration": args.stop_iters,
        # "timesteps_total": args.stop_timesteps,
        # "episode_reward_mean": args.stop_reward,
        "training_iteration": num_iterations,
        # "timesteps_total": 2000,
        # "episode_reward_mean": 15,
        "score": 57
    }

    tuner = tune.Tuner(
        evaluate,
        param_space=config,

        # run_config=air.RunConfig(stop=stop, verbose=1, checkpoint_config=CheckpointConfig(checkpoint_frequency=2)),
        # run_config=air.RunConfig(stop=stop, verbose=1),
        run_config=train.RunConfig(
                name="mlflow",
                callbacks=[
                    MLflowLoggerCallback(
                        tracking_uri=mlflow_tracking_uri,
                        experiment_name="mlflow_callback_example",
                        save_artifact=True,
                    )
                ],
                stop=stop,
                verbose=1
            ),
        # run_config=RunConfig(stop=TrialPlateauStopper(metric="score"))
        # run_config=RunConfig(stop=TrialPlateauStopper(metric="reward"),
        #                      # checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10),
        #                      checkpoint_config=CheckpointConfig(checkpoint_frequency=3),
        #                      verbose=2)
    )

    results = tuner.fit()
    df = results.get_dataframe(filter_metric="score", filter_mode="min")
    return df

def shape_to_topology2(a, b_to_a, lh_ratio):
    b = math.ceil(a*b_to_a)
    depth = math.ceil(max(a,b) * lh_ratio)
    topo = [a]
    ic(a,b,depth)
    if depth > 2:
        # layer_delta = math.ceil((b-a) / (depth-2))
        layer_delta = math.ceil((b-a) / (depth-1))
        ic(layer_delta)
        for i in range(depth-2):
            # print(i)
            cur_layer = topo[-1] + layer_delta
            # ic(cur_layer)
            topo.append(cur_layer)

    topo.append(b)
    # print(depth)
    return(topo)

def test_shape_to_topology():
    r1 = shape_to_topology2(7, 0.3, 0.7)
    ic(r1)
    r2 = shape_to_topology2(256, 0.3, 0.05)
    ic(r2)
    r3 = shape_to_topology2(256, 0.3, 0.01)
    ic(r3)
    r4 = shape_to_topology2(256, 0.3, 0.005)
    ic(r4)
    r5 = shape_to_topology2(512, 0.6, 0.005)
    ic(r5)
    r6 = shape_to_topology2(512, 0.6, 0.009)
    ic(r6)


def test_idle_embedded_tuners_hpo():

    ray.shutdown()
    num_cpus = 2
    local_mode = True
    framework = 'torch'
    evaluation_parallel_to_training = True
    # ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)
    ray.init(num_cpus=num_cpus or None, local_mode=local_mode)

    config = {
                "learn_nothing":    True,
                "num_folds":        3,
                "rand_params":      1,
                "num_train_iterations": 10,
                "h1": tune.uniform(0, 3),
                "h2": tune.uniform(2, 7)
             }

    hyperparam_mutations = {
        "a": lambda: tune.uniform(0, 3),
        "b": lambda: tune.choice([1, 2, 3, 98])
    }

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations=hyperparam_mutations,
        # custom_explore_fn=explore,
    )

    from ray.tune.search.bayesopt import BayesOptSearch
    # algo = BayesOptSearch(random_search_steps=4)
    algo = BayesOptSearch(random_search_steps=7)
    ic(config)
    # return

    def cv_scores(config):

        def inner_eval(config):
            # here I can learn nothing
            if config["learn_nothing"]:
                time.delay(5)
                return {"score":random.random()}
            else:
                # make algo.. standart control problem
                # and train it with tuner 5 iterations
                return {"score":-1}


        num_folds = config["num_folds"]
        # config = make_folds(config)
        config["test_fold_index"] = tune.grid_search(np.arange(0,num_folds))

        num_iterations=config.get('num_train_iterations', 2)
        stop = {
            "training_iteration": num_iterations,
        }
        tuner = tune.Tuner(
            inner_eval,
            param_space=config,
            run_config=air.RunConfig(stop=stop, verbose=1),
            # run_config=RunConfig(stop=TrialPlateauStopper(metric="score"))
            # run_config=RunConfig(stop=TrialPlateauStopper(metric="reward"),
            #                      # checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10),
            #                      checkpoint_config=CheckpointConfig(checkpoint_frequency=3),
            #                      verbose=2)
        )
        results = tuner.fit()

        df = results.get_dataframe(filter_metric="score", filter_mode="min")
        ic(df)
        cv_score = df["score"]
        return df

    def objective(config):
        # score = random.random()
        scores = cv_scores(config)
        # ic(scores)
        time.sleep(5)
        return {"score":scores["score"].mean()}
        # return {"score": random.random()}


    num_iterations=config.get('num_train_iterations', 2)
    ic(num_iterations)
    stop = {
        "training_iteration": num_iterations,
        "score": 57
    }

    search_space = {  # ②
        "a": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
        "b": tune.choice([1, 2, 3]),
    }

    # pbt = PopulationBasedTraining(
    #     time_attr="time_total_s",
    #     perturbation_interval=120,
    #     resample_probability=0.25,
    #     # Specifies the mutations of these hyperparams
    #     hyperparam_mutations=hyperparam_mutations,
    #     # custom_explore_fn=explore,
    # )

    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            # metric="episode_reward_mean",
            metric="score",
            mode="max",
            search_alg=algo,
            # scheduler=pbt,
            num_samples=7 # if args.smoke_test else 2,
        ),
        # param_space=config,
        # param_space=search_space,
        param_space=config,
        # run_config=air.RunConfig(stop=stop, verbose=1, checkpoint_config=CheckpointConfig(checkpoint_frequency=2)),
        run_config=air.RunConfig(stop=stop, verbose=1),
        # run_config=RunConfig(stop=TrialPlateauStopper(metric="score"))
        # run_config=RunConfig(stop=TrialPlateauStopper(metric="reward"),
        #                      # checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10),
        #                      checkpoint_config=CheckpointConfig(checkpoint_frequency=3),
        #                      verbose=2)
    )

    results = tuner.fit()
    df = results.get_dataframe(filter_metric="score", filter_mode="min")
    return df


def test_ray_example():
    def objective(config):  # ①
        score = config["a"] ** 2 + config["b"]
        return {"score": score}


    search_space = {  # ②
        "a": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
        "b": tune.choice([1, 2, 3]),
    }

    tuner = tune.Tuner(objective, param_space=search_space)  # ③

    results = tuner.fit()
    print(results.get_best_result(metric="score", mode="min").config)

if __name__ == "__main__":
    # ic.disable()
    ic.enable()
    ic.configureOutput(includeContext=True)

    # test_ray_example()
    # test_idle_embedded_tuners_hpo()


    # test_env_different_symbol_lengths()
    # test_get_dataset2()
    # test_get_dataset()
    # test_make_folds()
    # test_shape_to_topology()
    # test_eval_fold()
    test_get_cv_score()
    # test_hpo()
    # test_mlflow()
    # test_simulate()
    # test_create_ms_env()
    # eval('/home/happycosmonaut/ray_results/DQN_2023-11-16_21-27-38/DQN_multy_symbol_env_4bf15_00000_0_2023-11-16_21-27-40/checkpoint_000002')
    # test_get_train_test_feed()
    # test_observation_shape()
    # test_obs_space_of()
    # test_multy_symbols()
    # test_multy_symbol_simple_trade_close_manually()
    # test_multy_symbol_simple_use_force_sell()
    # test_end_episodes()
    # test_comission()
    # test_spread()
    # test_make_synthetic_symbol()
    # test_get_episode_lengths()
