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

def test_end_episodes():
    num_symbols=5
    symbols=[]
    for i in range(num_symbols):
        symbols.append(make_sin_symbol("Asset"+str(i), i))
    config = {"symbols": symbols,
              "reward_window_size": 7,
              "window_size": 1,
              "max_allowed_loss":100,
              "multy_symbol_env": True,
              "use_force_sell": True
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

        if is_end_of_episode(obs) == True:
            obs, reward, done, truncated, info = env.step(action)
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
            print('volumes: ', volumes)
            for v in volumes[-(len(volumes)-1):]:
                # print(v)
                assert v == 0
            step += 1
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
              "multy_symbol_env": True,
              "use_force_sell": True
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
    config = {"symbols": symbols,
              "reward_window_size": 7,
              "window_size": 1,
              "max_allowed_loss":100,
              "multy_symbol_env": True
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
        assert pytest.approx(obs[-1][0], 0.001) == dataset.iloc[step].close
        assert pytest.approx(obs[-1][1], 0.001) == dataset.iloc[step].symbol_code
        assert pytest.approx(obs[-1][2], 0.001) == dataset.iloc[step].end_of_episode


        if step == 10 or step == 20:
            action = 1

        if step == 15 or step == 25:
            action = 0

        print(step, ': ', obs, dataset.iloc[step])
        if is_end_of_episode(obs) == True:
            print('end_of_episode == True')
            volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)
            for v in volumes[-(len(volumes)-1):]:
                print(v)
                # assert v == 0

            obs, reward, done, truncated, info = env.step(action)
            step += 1
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


def test_nfoldcv():
    pass

def test_multy_simbol_simple_trade_close_manually():
    # * close orders manually (by agent) before end_of_episode    

    num_symbols=5
    symbols=[]
    for i in range(num_symbols):
        symbols.append(make_sin_symbol("Asset"+str(i), i))
    config = {"symbols": symbols,
              "reward_window_size": 7,
              "window_size": 1,
              "max_allowed_loss":100,
              "multy_symbol_env": True,
              "use_force_sell": False
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
    observations=[np.append(obs[-1], np.append([action, info['net_worth']], volumes))]

    # test feed
    while done == False and step < 242:
        assert pytest.approx(obs[-1][0], 0.001) == dataset.iloc[step].close
        assert pytest.approx(obs[-1][1], 0.001) == dataset.iloc[step].symbol_code
        assert pytest.approx(obs[-1][2], 0.001) == dataset.iloc[step].end_of_episode

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

        row = np.append(obs[-1], np.append([action, info['net_worth']], volumes))
        observations.append(row)
        volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)

    track = pd.DataFrame(observations)
    track.columns = np.append(['close',   'symbol_code',  'end_of_episode', 'action', 'net_worth'], instruments)
    print(track.to_markdown())


    # check net_worth calc
    for index, row in track.iterrows():
        # row["net_worth"]
        net_worth_test = sum([row[f"AST{i}"]*row["close"] for i in range(5)]) + row["USDT"]
        assert pytest.approx(row["net_worth"], 0.001) == net_worth_test
    return

def test_multy_simbol_simple_use_force_sell():
    # * don't close orders manually (by agent) before end_of_episode
    #  use force_sell functionality for that purposes

    num_symbols=5
    symbols=[]
    for i in range(num_symbols):
        symbols.append(make_sin_symbol("Asset"+str(i), i))
    config = {"symbols": symbols,
              "reward_window_size": 7,
              "window_size": 1,
              "max_allowed_loss":100,
              "multy_symbol_env": True,
              "use_force_sell": True
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
    observations=[np.append(obs[-1], np.append([action, info['net_worth']], volumes))]

    # test feed
    while done == False and step < 242:
        assert pytest.approx(obs[-1][0], 0.001) == dataset.iloc[step].close
        assert pytest.approx(obs[-1][1], 0.001) == dataset.iloc[step].symbol_code
        assert pytest.approx(obs[-1][2], 0.001) == dataset.iloc[step].end_of_episode

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
        net_worth=volumes[0]
        for v in volumes[-(len(volumes)-1):]:
            net_worth += v*obs[-1][0]

        row = np.append(obs[-1], np.append([action, info['net_worth']], volumes))
        observations.append(row)
        volumes = get_wallets_volumes(env.env.action_scheme.portfolio.wallets)

    track = pd.DataFrame(observations)
    track.columns = np.append(['close',   'symbol_code',  'end_of_episode', 'action', 'net_worth'], instruments)
    print(track.to_markdown())

    # check net_worth calc
    for index, row in track.iterrows():
        row["net_worth"]
        net_worth_test = sum([row[f"AST{i}"]*row["close"] for i in range(num_symbols)]) + row["USDT"]
        assert pytest.approx(row["net_worth"], 0.001) == net_worth_test

    return

if __name__ == "__main__":
    test_multy_symbols()
    test_multy_simbol_simple_trade_close_manually()
    test_multy_simbol_simple_use_force_sell()
    test_end_episodes()
    # test_comission()
    # test_spread()
    # test_nfoldcv()