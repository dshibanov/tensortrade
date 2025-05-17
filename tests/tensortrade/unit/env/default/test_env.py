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
# import ray
# from ray import tune
# from ray.tune.registry import register_env
from tensortrade.env.default import *
import copy
# from ray.rllib.algorithms.dqn.dqn import DQNConfig
# from ray import air, tune
# from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
# from ray.rllib.utils import check_env
# from ray.rllib.algorithms.algorithm import Algorithm
from icecream import ic
# from plot import plot_history, set_gruvbox
# from tensortrade.utils.plot import plot_history, set_gruvbox

# from ray.tune.schedulers import PopulationBasedTraining
# from evaluators import *

# register_env("multy_symbol_env", create_multy_symbol_env)

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
              "num_service_cols" : 1,
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

        row = np.append(np.append(obs[-1], np.append(get_observer(env).symbol_code, info["end_of_episode"])), np.append([action, info['net_worth']], volumes))
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

    symbols = make_symbols(process=FLAT)

    config = {
              # "symbols": make_symbols(num_symbols, 100, True),
              "symbols": symbols,
              "reward_window_size": 7,
              "window_size": 1,
              "max_allowed_loss":100,
              "multy_symbol_env": True,
              "use_force_sell": True,
              "num_service_cols" : 1,
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

        row = np.append(np.append(obs[-1], np.append(get_observer(env).symbol_code, info["end_of_episode"])), np.append([action, info['net_worth']], volumes))
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

    config = {
              # "symbols": make_symbols(num_symbols, 666, True),
              "symbols": make_symbols(num_symbols, 12, True),
              "reward_window_size": 7,
              "window_size": 3,
              "max_allowed_loss":100,
              "multy_symbol_env": True,
              "use_force_sell": True,
              "num_service_cols" : 1,
              "make_folds": False,
              "num_folds":5,
              "max_episode_length": 5,
              "test": False
             }

    print(config)
    # return
    dataset = pd.concat([config["symbols"][i]["feed"] for i in range(len(config["symbols"]))])
    env = create_multy_symbol_env(config)
    # return
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
        observed_close = obs[-1]
        # print('obs:: ', obs[-2], dataset.iloc[step].close)
        assert pytest.approx(observed_close , 0.001) == dataset.iloc[step].close
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
        assert 'symbol_code' in info
        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        # check here that volumes doesn't have nan's
        # print('loop by volumes...')
        for v in volumes:
            assert math.isnan(v) == False

        # row = np.append(obs[-1], np.append([action, info['net_worth']], volumes))
        # row = np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], volumes))
        row = np.append(np.append(obs[-1], np.append(get_observer(env).symbol_code, info["end_of_episode"])),
                        np.append([action, info['net_worth']], volumes))
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
              "num_service_cols" : 1,
              "make_folds": False,
              "test": False,
              "base_symbol": 'USDT'
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


    # observations=[np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], volumes))]
    observations=[np.append(np.append(obs[-1], np.append(get_observer(env).symbol_code, info["end_of_episode"])), np.append([action, info['net_worth']], volumes))]


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

        row = np.append(np.append(obs[-1], np.append(get_observer(env).symbol_code, info["end_of_episode"])),
                        np.append([action, info['net_worth']], volumes))
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
              "symbols": make_symbols(num_symbols, 15, True),
              "reward_window_size": 7,
              "window_size": 1,
              "max_allowed_loss":100,
              "multy_symbol_env": True,
              "use_force_sell": True,
              "num_service_cols" : 1,
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
    observations=[np.append(np.append(obs[-1], np.append(get_observer(env).symbol_code, info["end_of_episode"])), np.append([action, info['net_worth']], volumes))]

    # test feed
    while done == False and step < 3420:
        assert pytest.approx(obs[-1], 0.001) == dataset.iloc[step].close
        assert pytest.approx(get_observer(env).symbol_code, 0.001) == dataset.iloc[step].symbol_code

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

        assert info["net_worth"] > 0
        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        # check here that volumes doesn't have nan's
        for v in volumes:
            assert math.isnan(v) == False

        net_worth=volumes[0]
        for v in volumes[-(len(volumes)-1):]:
            net_worth += v*obs[-1]

        row = np.append(np.append(obs[-1], np.append(get_observer(env).symbol_code, info["end_of_episode"])), np.append([action, info['net_worth']], volumes))
        observations.append(row)
        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

    track = pd.DataFrame(observations)
    track.columns = np.append(['close',   'symbol_code',  'end_of_episode', 'action', 'net_worth'], instruments)
    print(track.to_markdown())

    # check net_worth calc
    for index, row in track.iterrows():
        net_worth_test = sum([row[f"AST{i}"]*row["close"] for i in range(num_symbols)]) + row["USDT"]
        assert pytest.approx(row["net_worth"], 0.001) == net_worth_test

    return

def test_make_synthetic_symbol():
    print("_______test_make_synthetic_symbol_____")
    config = {"name": 'X',
              "spread": 0.001,
              "commission": 0.0001,
              "code": 0,
              "num_of_samples": 31,
              "max_episode_steps": 41,
              # "max_episode_steps": 152,
              # "process": FLAT,
              "process": SIN,
              # "process": 'KUSKS', # this one for checking exception when
                  # processes is not correct
              "price_value": 100,
              "start_date": '1/1/2011'}

    config["shatter_on_episode_on_creation"] = True
    s = make_synthetic_symbol(config)
    print(s["feed"])
    # print(s)

    assert 'symbol' not in s["feed"].columns
    assert 'close' in s["feed"].columns

    last_episode_start=0
    for i, value in enumerate(s["feed"].iterrows(), 0):
        index, row = value
        if row["end_of_episode"] == True:
            print("     > ", i)
            ep_length = i - last_episode_start
            last_episode_start = i+1

            print(ep_length, config["max_episode_steps"])
            assert ep_length <= config["max_episode_steps"]

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
    print(np.shape(obs), get_observer(env).observation_space.shape)
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

if __name__ == "__main__":
    # ic.disable()
    ic.enable()
    ic.configureOutput(includeContext=True)


    # TODO: revise all of these tests and remove useless


    # test_ray_example() # OK but we should remove it
    # test_idle_embedded_tuners_hpo() # looks OK but should be moved


    # test_env_different_symbol_lengths() # FIXME: NOT OK
    # test_get_dataset2() # OK
    # test_get_dataset() # OK

    # test_make_folds() # OK
    # test_shape_to_topology() # OK


    # FIXME
    # !! some of these tests testing functionality which is not related to
    # tensortrade, so they should be removed or moved to searcher project
    #
    # !! check before that tests are working in searcher 


    # test_hpo() # FIXME: not ok .. but this is related to searcher functionality so
                 # we should remove it from here   
    # test_mlflow() # move it to somewhere
    # test_simulate() # FIXME: NOT OK
                    # checkpoint not found error


    # test_create_ms_env() # NOT OK
    # eval('/home/happycosmonaut/ray_results/DQN_2023-11-16_21-27-38/DQN_multy_symbol_env_4bf15_00000_0_2023-11-16_21-27-40/checkpoint_000002')
    # test_get_train_test_feed() # OK
    # test_observation_shape() # FIXME: some problems with this test
    # # test_obs_space_of() # OK
    test_multy_symbols() # OK
    # test_multy_symbol_simple_trade_close_manually() # OK
    # test_multy_symbol_simple_use_force_sell() # OK
    # test_end_episodes() # OK
    # test_comission() # NOT OK
    # test_spread() # OK
    # test_make_synthetic_symbol() # OK
    # test_eval_fold() # OK but should be removed
    # test_get_cv_score() # OK but should be removed
