import numpy as np
import pytest
import sys, os, time
sys.path.append(os.getcwd())
import pandas as pd
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
import tensortrade.env.default as default
import ray.rllib.utils as rlutils

# TICKER = 'AAPL'  # TODO: replace this with your own ticker
# TRAIN_START_DATE = '2022-02-09'  # TODO: replace this with your own start date
# TRAIN_END_DATE = '2022-03-09'  # TODO: replace this with your own end date
# EVAL_START_DATE = '2022-10-01'  # TODO: replace this with your own end date
# EVAL_END_DATE = '2022-11-12'  # TODO: replace this with your own end date

from gymnasium.wrappers import EnvCompatibility
import ray
from ray import tune
from ray.tune.registry import register_env
from tensortrade.env.generic.multy_symbol_env import *

def test_end_episodes():
    num_symbols=5
    symbols=[]
    for i in range(num_symbols):
        symbols.append(make_sin_symbol("Asset"+str(i), i))
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
    while done == False and step < 242:
        assert pytest.approx(obs[-1][0], 0.001) == dataset.iloc[step].close
        assert pytest.approx(obs[-1][1], 0.001) == dataset.iloc[step].symbol_code
        assert pytest.approx(obs[-1][2], 0.001) == dataset.iloc[step].end_of_episode

        # if obs[-1][2] == True:
        if is_end_of_episode(obs) == True:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
            for v in volumes[-(len(volumes)-1):]:
                print(v)
                assert v == 0
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
            assert volumes[0] == 0

        row = np.append(obs[-1], np.append([action, info['net_worth']], volumes))
        observations.append(row)
        volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
    track = pd.DataFrame(observations)
    track.columns = np.append(['close',   'symbol_code',  'end_of_episode', 'action', 'net_worth'], instruments)
    print(track.to_markdown())
    return

def test_spread():
    num_symbols=5
    symbols=[]
    for i in range(num_symbols):
        if i == 2:
            symbols.append(make_flat_symbol("AST"+str(i), i, commission=0, spread=1.13))
        elif i == 4:
            symbols.append(make_flat_symbol("AST"+str(i), i, commission=0, spread=3.66))
        else:
            symbols.append(make_flat_symbol("AST"+str(i), i, commission=0, spread=0.01))

    config = {"symbols": symbols,
              "reward_window_size": 7,
              "window_size": 1,
              "max_allowed_loss":100,
              "multy_symbol_env": True
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
        assert pytest.approx(obs[-1][1], 0.001) == dataset.iloc[step].symbol_code
        assert pytest.approx(obs[-1][2], 0.001) == dataset.iloc[step].end_of_episode

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

        row = np.append(obs[-1], np.append([action, info['net_worth']], volumes))
        observations.append(row)
        volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
    track = pd.DataFrame(observations)
    track.columns = np.append(['close',   'symbol_code',  'end_of_episode', 'action', 'net_worth'], instruments)
    print(track.to_markdown())
    return

def test_comission():
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
    config = {"symbols": symbols,
              "reward_window_size": 7,
              "window_size": 1,
              "max_allowed_loss":100,
              "multy_symbol_env": True
             }

    dataset = pd.concat([config["symbols"][i]["feed"] for i in range(len(config["symbols"]))])
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(dataset)
    # print(dataset.to_markdown())

    env = create_multy_symbol_env(config)
    action = 0 # do nothing
    obs,_ = env.reset()
    info = env.env.informer.info(env.env)
    # env.render()

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
        assert pytest.approx(obs[-1][1], 0.001) == dataset.iloc[step].symbol_code
        assert pytest.approx(obs[-1][2], 0.001) == dataset.iloc[step].end_of_episode
        if obs[-1][2] == True:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            # check wallets here
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
            for v in volumes[-(len(volumes)-1):]:
                print(v)
                assert v == 0
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
            assert volumes[0] == 0

        row = np.append(obs[-1], np.append([action, info['net_worth']], volumes))
        # print(row)
        observations.append(row)

        print(step, ': ', obs, dataset.iloc[step])
        print(obs[-1][0], dataset.iloc[step].close)

        if step == 39:
            print('39')

        volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
        print(volumes)


    track = pd.DataFrame(observations)

    # print(track.to_markdown())
    # return
    # track.columns = ['close',   'symbol_code',  'end_of_episode', 'action', 'net_worth']
    # columns = 
    track.columns = np.append(['close',   'symbol_code',  'end_of_episode', 'action', 'net_worth'], instruments)
    print(track.to_markdown())

    return
    print(step)
    volumes=[]
    for w in env.env.action_scheme.portfolio.wallets:
        balance = w.total_balance
        # print(balance.instrument, balance.size)
        volumes.append(float(balance.size))
    print(volumes)
    return

    print('---====================== START =======================---')
    while done == False and step < 44:
        print(f'step: {step} action: {action}')
        if step == 3:
            action = 1 # buy

        if step == 7:
            action = 0 # sell

        if step == 9:
            action = 1

        if step > 11:
            action = 0

        if step == 39:
            print(obs)

        obs, reward, done, truncated, info = env.step(action)
        print(step,':',obs)
        # net_worth.append(info['net_worth'])
        # actions.append(action)
        # row = obs[-1]
        volumes=[]
        for w in env.env.action_scheme.portfolio.wallets:
            balance = w.total_balance
            # print(balance.instrument, balance.size)
            volumes.append(float(balance.size))
        # wallets_volumes.append(volumes)
        row = np.append(obs[-1], np.append([action, info['net_worth']], volumes))
        # print(row)
        observations.append(row)
        # get assets here
        # loop via assets and cash


        # env.render()
        print('step: ', step, 'obs: ', obs, ' reward: ', reward)
        print('info ', info)
        step +=1

        # if obs[-1][-1] == True:
        #     print('end of episode.. please close all orders')
        #     # TODO: close orders by.. from exchange or broker
        #     # without any action..
        #     # env closing opened orders, not agent

        #     # and here open order again with action
        #     obs, reward, done, truncated, info = env.step(action)
        #     # thats it


    # obs, reward, done, truncated, info = env.step(1)
    # print('step: ', step, 'obs: ', obs, ' reward: ', reward)

    # history = pd.DataFrame(env.env.observer.renderer_history)
    # trades=env.env.action_scheme.broker.trades
    # print('---trades')
    # for t in trades:
    #     print(type(trades[t]))
    #     print(len(trades[t]))
    #     print(trades[t][0])


    # TODO:
        #  - make df from lists [DONE]
        #  - add assets volumes
        #
        #  - check for orders closing in end_episode point
        #       assert 
        #
        #  - 

    # print(observations)
    # print(np.shape(observations))

    # print(actions)
    # print(net_worth)

    track = pd.DataFrame(observations)

    # print(track.to_markdown())
    # return
    # track.columns = ['close',   'symbol_code',  'end_of_episode', 'action', 'net_worth']
    # columns = 
    track.columns = np.append(['close',   'symbol_code',  'end_of_episode', 'action', 'net_worth'], instruments)
    print(track.to_markdown())

    # TODO:
        #   ask where to get trades history
        #   orders history

def test_nfoldcv():
    pass

if __name__ == "__main__":

    # from gymnasium.spaces import Space, Discrete
    # space = Discrete(2)
    # print(space)
    # test_multy_symbols()

    # test_end_episodes()
    # test_comission()
    test_spread()
    # test_nfoldcv()
    # make_flat_feed()
    # print(make_flat_symbol('TEST'))
