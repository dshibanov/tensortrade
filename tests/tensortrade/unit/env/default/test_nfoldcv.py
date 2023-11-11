import numpy as np
import pytest
import sys, os, time
sys.path.append(os.getcwd())
import pandas as pd
import pytest
import ray
from ray import tune
from ray.tune.registry import register_env
from tensortrade.env.default import *
# import math

import copy
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray import air, tune
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.rllib.utils import check_env

# TODO: please rename this file to test_env.py
# or better move these tests to existed one

register_env("multy_symbol_env", create_multy_symbol_env)
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
    while done == False and step < 242:
        assert pytest.approx(obs[-1][0], 0.001) == dataset.iloc[step].close
        # assert pytest.approx(obs[-1][1], 0.001) == dataset.iloc[step].symbol_code
        # assert pytest.approx(obs[-1][2], 0.001) == dataset.iloc[step].end_of_episode
        #
        if step == 38:
            action = 1
        else:
            action = 0

        if is_end_of_episode(obs) == True:
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
            for v in volumes[-(len(volumes)-1):]:
                assert v == 0

            obs, reward, done, truncated, info = env.step(action)
            # volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
            # print('volumes: ', volumes)
            # for v in volumes[-(len(volumes)-1):]:
            #     # print(v)
            #     assert v == 0
            step += 1
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)

        # row = np.append(obs[-1], np.append([action, info['net_worth']], volumes))
        row = np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], volumes))
        observations.append(row)
        volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
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
    while done == False and step < 214:
        assert pytest.approx(obs[-1][0], 0.001) == dataset.iloc[step].close
        # assert pytest.approx(obs[-1][1], 0.001) == dataset.iloc[step].symbol_code
        # assert pytest.approx(obs[-1][2], 0.001) == dataset.iloc[step].end_of_episode

        if step == 0:
            action=0
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
            assert volumes[0] == 0
            assert volumes[1] != 0
            assert volumes[1] == 9.999
        elif step == 1:
            action=1
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
            assert volumes[0] != 0
            assert volumes[0] == 999.9
            assert volumes[1] == 0
        elif step == 5:
            action=0
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
            assert volumes[0] == 0
            assert volumes[1] != 0
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)

        # row = np.append(obs[-1], np.append([action, info['net_worth']], volumes))
        row = np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], volumes))
        observations.append(row)
        volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
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
    info = env.env.informer.info(env.env)
    # env.render()

    done = False
    step = 0
    track=[]
    instruments=[]
    volumes=[]
    for w in env.env.action_scheme.portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))
    observations=[np.append(obs[-1], np.append([action, info['net_worth']], volumes))]

    # test feed
    while done == False and step < 242:
        print('obs:: ', obs[-1][0], dataset.iloc[step].close)
        assert pytest.approx(obs[-1][0], 0.001) == dataset.iloc[step].close
        # assert pytest.approx(obs[-1][1], 0.001) == dataset.iloc[step].symbol_code
        # assert pytest.approx(obs[-1][2], 0.001) == dataset.iloc[step].end_of_episode


        if step == 10 or step == 20:
            action = 1

        if step == 15 or step == 25:
            action = 0

        print(step, ': ', obs, dataset.iloc[step])
        if is_end_of_episode(obs) == True:
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
            for v in volumes[-(len(volumes)-1):]:
                assert v == 0

            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
            # for v in volumes[-(len(volumes)-1):]:
            #     print(v)
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)

        assert info["net_worth"] > 0
        volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)

        # check here that volumes doesn't have nan's
        # print('loop by volumes...')
        for v in volumes:
            assert math.isnan(v) == False

        # row = np.append(obs[-1], np.append([action, info['net_worth']], volumes))
        row = np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], volumes))
        observations.append(row)
        volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)

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
    info = env.env.informer.info(env.env)

    done = False
    step = 0
    track=[]
    instruments=[]
    volumes=[]
    for w in env.env.action_scheme.portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))


    observations=[np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], volumes))]

    # test feed
    while done == False and step < 242:
        # print('obs ', obs)
        assert pytest.approx(obs[-1][0], 0.001) == dataset.iloc[step].close
        # assert pytest.approx(obs[-1][1], 0.001) == dataset.iloc[step].symbol_code
        # assert pytest.approx(obs[-1][2], 0.001) == dataset.iloc[step].end_of_episode

        if (step > 37 and step < 42) or (step >= 79 and step < 82) or (step >= 119 and step < 123) or (step >= 159 and step < 164):
            action = 1
        else:
            action = 0

        if is_end_of_episode(obs) == True:
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
            for v in volumes[-(len(volumes)-1):]:
                assert v == 0

            obs, reward, done, truncated, info = env.step(action)
            step += 1
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)

        # assert net_worth value
        assert info["net_worth"] > 0
        volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
        net_worth=volumes[0]
        for v in volumes[-(len(volumes)-1):]:
            net_worth += v*obs[-1][0]

        # print("info___: ", info)
        # np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), 
        # row = np.append(obs[-1], np.append([action, info['net_worth']], volumes))
        row = np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], volumes))
        observations.append(row)
        volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)

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
    info = env.env.informer.info(env.env)

    done = False
    step = 0
    track=[]
    instruments=[]
    volumes=[]
    for w in env.env.action_scheme.portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))
    observations=[np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], volumes))]

    # test feed
    while done == False and step < 242:
        # assert pytest.approx(obs[-1][0], 0.001) == dataset.iloc[step].close
        print('obs ', obs)
        assert pytest.approx(obs[-1][0], 0.001) == dataset.iloc[step].close
        # assert pytest.approx(obs[-1][1], 0.001) == dataset.iloc[step].symbol_code
        print(f"step: {step} close {obs[-1]} {dataset.iloc[step].close} info {info['symbol_code']}  {info['end_of_episode']} dataset: {dataset.iloc[step].symbol_code} {dataset.iloc[step].end_of_episode}")
        assert pytest.approx(info["symbol_code"], 0.001) == dataset.iloc[step].symbol_code
        # assert pytest.approx(obs[-1][2], 0.001) == dataset.iloc[step].end_of_episode
        # print(' end of episode ', info["end_of_episode"], dataset.iloc[step].end_of_episode)
        # assert pytest.approx(info["end_of_episode"], 0.001) == dataset.iloc[step].end_of_episode

        if (step > 57 and step < 63): #  or (step >= 79 and step < 82) or (step >= 119 and step < 123) or (step >= 159 and step < 164):
            # sell 
            action = 1
        else:
            action = 0

        if is_end_of_episode(obs) == True:
            obs, reward, done, truncated, info = env.step(action)
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
            step += 1
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)

        # assert net_worth value
        assert info["net_worth"] > 0
        volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)

        # check here that volumes doesn't have nan's
        print('loop by volumes...')
        for v in volumes:
            assert math.isnan(v) == False

        net_worth=volumes[0]
        for v in volumes[-(len(volumes)-1):]:
            net_worth += v*obs[-1][0]

        # row = np.append(obs[-1], np.append([action, info['net_worth']], volumes))
        row = np.append(np.append(obs[-1], np.append(info["symbol_code"], info["end_of_episode"])), np.append([action, info['net_worth']], volumes))
        observations.append(row)
        volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)

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

    print(obs, np.shape(obs), env.env.observer.observation_space)
    assert np.shape(obs) == env.env.observer.observation_space.shape
    obs, reward, done, truncated, info = env.step(0)
    print(obs, np.shape(obs), env.env.observer.observation_space)
    assert np.shape(obs) == env.env.observer.observation_space.shape


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



    r = get_dataset(config)
    episodes_lengths = get_episodes_lengths(r[0])
    print(f'episodes {episodes_lengths} \n min {min(episodes_lengths)} max {max(episodes_lengths)}')
    # assert max(episodes_lengths) <= config["max_episode_length"]
    # assert min(episodes_lengths) >= config["min_episode_length"]
    config["make_folds"] = True
    config["test"] = True
    config["test_fold_index"] = 3
    config["num_folds"] = 7
    config = make_folds(config)
    r = get_dataset(config)
    episodes_lengths = get_episodes_lengths(r[1])
    print(f'episodes {episodes_lengths} \n min {min(episodes_lengths)} max {max(episodes_lengths)}')
    assert max(episodes_lengths) <= config["max_episode_length"]
    assert min(episodes_lengths) >= config["min_episode_length"]
    config["test"] = False
    r = get_dataset(config)
    episodes_lengths = get_episodes_lengths(r[0])
    print(f'episodes {episodes_lengths} \n min {min(episodes_lengths)} max {max(episodes_lengths)}')
    assert max(episodes_lengths) <= config["max_episode_length"]
    assert min(episodes_lengths) >= config["min_episode_length"]

def eval_fold(params):
    score = 2
    def set_params(config, params):
        # print(params)
        for p in params:
            # print(p)
            config[p] = params[p]
        return config

    framework = 'torch'
    # current_config = set_params(config, params)
    current_config = copy.deepcopy(params)#.copy()
    print('current_config: feed ', current_config["symbols"][0]["feed"])
    symbols = current_config["symbols"]
    test_fold_index = current_config["test_fold_index"]
    print('test_fold_index ', test_fold_index)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     train_feed, test_feed = get_train_test_feed(current_config)
    #     print('FEED ', len(train_feed), len(test_feed))


    eval_config = current_config
    eval_config["test"] = True
    env = create_multy_symbol_env(current_config)
    print('check_env ')
    check_env(env)


    a = env.reset()
    pprint(a)
    # return

    config = (
        # PGConfig()
        DQNConfig()
        # .environment(SimpleCorridor, env_config={"corridor_length": 10})
        # .environment(env="multy_symbol_env", env_config=config)
        .environment(env="multy_symbol_env", env_config=current_config)
        # Training rollouts will be collected using just the learner
        # process, but evaluation will be done in parallel with two
        # workers. Hence, this run will use 3 CPUs total (1 for the
        # learner + 2 more for evaluation workers).
        # .rollouts(num_rollout_workers=0)
        # .evaluation(
        # #     evaluation_num_workers=2,
        # #     # Enable evaluation, once per training iteration.
        # #     evaluation_interval=1,
        # #     # Run 10 episodes each time evaluation runs (OR "auto" if parallel to
        # #     # training).
        # #     # evaluation_duration="auto" if args.evaluation_parallel_to_training else 10,
        # #     evaluation_duration="auto" if evaluation_parallel_to_training else 10,
        # #     # Evaluate parallelly to training.
        # #     # evaluation_parallel_to_training=args.evaluation_parallel_to_training,
        # #     evaluation_parallel_to_training=evaluation_parallel_to_training,
        # #     # evaluation_config=PGConfig.overrides(
        #     # evaluation_config=DQNConfig.overrides(env_config={"test": True}, explore=True
        #     evaluation_config=DQNConfig.overrides(env_config=eval_config
        # #         env_config={
        # #             # Evaluate using LONGER corridor than trained on.
        # #             "corridor_length": 5,
        # #         },
        #     )
        #     custom_evaluation_function=eval_fn,
        # )
        # .framework(args.framework)
        .framework(framework)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    stop = {
        # "training_iteration": args.stop_iters,
        # "timesteps_total": args.stop_timesteps,
        # "episode_reward_mean": args.stop_reward,
        "training_iteration": 3,
        # "timesteps_total": 2000,
        "episode_reward_mean": 15,
    }

    # pprint({key: value for key, value in config.to_dict().items() if key not in ["symbols"]})
    # return
    tuner = tune.Tuner(
        # "PG",
        "DQN",
        # param_space=config.to_dict(),
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=1, checkpoint_config=CheckpointConfig(checkpoint_frequency=2)),
        # run_config=RunConfig(stop=TrialPlateauStopper(metric="score"))
        # run_config=RunConfig(stop=TrialPlateauStopper(metric="reward"),
        #                      # checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10),
        #                      checkpoint_config=CheckpointConfig(checkpoint_frequency=3),
        #                      verbose=2)
    )
    # results = tuner.fit()
    results = tuner.fit().get_dataframe(filter_metric="score", filter_mode="min")
    print(results)


    # 2 make validation here
    #
    # 3 return validation_score

    # return {"score": score}



def test_eval_fold():
    config = {"max_episode_length": 50, # bigger is ok, smaller is not
              "make_folds":True,
              "num_folds": 7,
              "symbols": make_symbols(5, 410),
              "cv_mode": 'proportional',
              "test_fold_index": 3,
              "reward_window_size": 1,
              "window_size": 2,
              "max_allowed_loss": 100,
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
    # print('config ', config)
    # print('AFTER ')
    pprint({key: value for key, value in config.items() if key not in ["symbols"]})

    # evaluation_config=DQNConfig.overrides(explore=False)
    # pprint(evaluation_config)
    num_cpus = 2
    local_mode = True
    # framework = 'torch'
    evaluation_parallel_to_training = True
    # # ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)
    ray.init(num_cpus=num_cpus or None, local_mode=local_mode)
    eval_fold(config)
    return

if __name__ == "__main__":
    # test_get_dataset()
    test_eval_fold()
    # test_get_train_test_feed()
    # test_observation_shape()
    # test_multy_symbols()
    # test_multy_symbol_simple_trade_close_manually()
    # test_multy_symbol_simple_use_force_sell()
    # test_end_episodes()
    # # test_comission()
    # test_spread()
    # test_make_synthetic_symbol()
    # test_get_episode_lengths()
