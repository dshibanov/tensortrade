import os
from quantutils.tracks import load_tracks
from pprint import pprint
import json
import pandas as pd
import numpy as np
import tensortrade.env.config as econf
from tensortrade.env.config import prepare, get_agent
from tensortrade.env.default import get_obs_header, get_action_scheme, get_info, get_observer, get_param, get_wallets_volumes, get_feed

import quantstats as qs
from finplot.plotmetrics import plot
from tensortrade.utils.synthetics import make_synthetic_symbol
import pytest
import copy
import matplotlib.pyplot as plt

# @pytest.fixture
def config():


    with open("../quanttools/datamart/tracks/options.json", "r") as file:
        options = json.load(file)

    timeframes = ['0']
    config = {'env':{
                    'data': {
                              'timeframes': timeframes,
                              # 'from': s['quotes'][timeframes[0]].index[0],
                              # 'to': s['quotes'][timeframes[0]].index[-1],
                              'symbols': [], # s
                              #            ],
                              # 'num_folds': 3,
                              'max_episode_length': 40,
                              'min_episode_length': 15,
                              'add_index': True
                            },

                    'action_scheme': {'name': 'tensortrade.env.default.actions.MSBSCMT',
                                      'params': []},
                    'reward_scheme': {'name': 'tensortrade.env.default.rewards.SimpleProfit',
                                      'params': [{'name': 'window_size', 'value': 2}]},


                    'exchange_config': options,
                    # in this section general params
                    'params':[{'name': "feed_calc_mode", 'value': econf.FEED_MODE_NORMAL},
                            {'name': "make_folds", 'value': False},
                            {'name': "multy_symbol_env", 'value': True},
                            {'name': "use_force_sell", 'value': True},
                            {'name': "add_features_to_row", 'value': True},
                            {'name': "max_allowed_loss", 'value': 100},
                            {'name': "test", 'value': False},
                            {'name': "reward_window_size", 'value': 7},
                            {'name': "window_size", 'value': econf.TICKS_PER_BAR+1},
                            # {'name': "window_size", 'value': 2},
                            {'name': "num_service_cols", 'value': 2},
                            # {'name': "load_feed_from", 'value': 'feed.csv'},
                              {'name': "load_feed_from", 'value': ''},
                              {'name': "leverage", 'value': 5},

                            ## save_feed, save calculated feed 
                            ## WARNING: this works if num of your agent is 1,
                            ## Otherwise it will work not correctly
                            {'name': "save_feed", 'value': False}],


                    "amount": 10,     # trade amount
                    "trade_proportion": 0.1 # proportion of cash to trade
                },

              'agents': [
                    # {'name': 'tensortrade.agents.dummy_agent.DummyAgent',
                    #      'params': [
                    #          # {'name': 'trades', 'value': (tracks[0]['df']['open'], )}
                    #          {'name': 'open', 'value': tracks[0]['df']['open']},
                    #          {'name': 'close', 'value': tracks[0]['df']['close']},
                    #          {'name': 'timeframes', 'value': ['0']},


                             # {'name': 'n_actions', 'value': 2},
                             #        {'name': 'observation_space_shape', 'value': (10,1)},
                             #        {'name': 'fast_ma', 'value': 3, 'optimize': True, 'lower': 2,
                             #         'upper': 5},
                             #        {'name': 'slow_ma', 'value': 5, 'optimize': True, 'lower': 5,
                             #         'upper': 11},
                             #        {'name': 'timeframes', 'value': timeframes}
                             #        ]},
                         # {'name': 'agents.sma_cross_rl.SMACross_TwoScreens',
                         # {'name': 'tensortrade.agents.sma_cross.SMACross',
                         #  'params': [{'name': 'n_actions', 'value': 2},
                         #             {'name': 'observation_space_shape', 'value': (10, 1)},
                         #             {'name': 'fast_ma', 'value': 2},
                         #             {'name': 'slow_ma', 'value': 11},
                         #             {'name': 'timeframes', 'value': timeframes}]}
                         ],
               # 'datamart': dm.DataMart(),
               'params': [{'name': 'add_features_to_row', 'value': True},
                            {'name':'check_track', 'value':True}],
              # "evaluate": simulate,
              "algo": {},
              "max_episode_length": 15, # smaller is ok
              "min_episode_length": 5, # bigger is ok, smaller is not
              "make_folds":True,
              "num_folds": 5,
              "cv_mode": 'proportional',
              "test_fold_index": 3,
              "reward_window_size": 1,
              "window_size": econf.TICKS_PER_BAR,
              "max_allowed_loss": 0.9,
              "use_force_sell": True,
              "multy_symbol_env": True,
              "test": False,
    }

    prepare(config)
    return config


def action_test_config():
    _config = config()
    timeframes = ['15m']
    synth = make_synthetic_symbol({
        'name': 'AST0',
        'from': '2020-1-1',
        'to': '2020-1-5',
        'synthetic': True,
        'ohlcv': True,
        'code': 0,
        'timeframes': timeframes
    })


    tracks = load_tracks("../quanttools/datamart/tracks")
    plot_tracks = False
    tnum=1
    tracks[tnum]['df']['order_book']['datetime'] = pd.to_datetime(tracks[tnum]['df']['order_book']['timestamp'], unit='ms')
    s = track_to_symbol(tracks[tnum])
    _config['env']['data']['symbols'] = [s]
    _config['env']['data']['symbols'][0]['feed'] = None
    _config['env']['data']['symbols'][0]['quotes'] = copy.deepcopy(synth['quotes'])
    _config['env']['data']['symbols'][0]['timeframes'] = timeframes
    _config['env']['data']['timeframes'] = timeframes
    _config['env']['data']['from'] = _config['env']['data']['symbols'][0]['quotes'][timeframes[0]].index[0]
    _config['env']['data']['to'] = _config['env']['data']['symbols'][0]['quotes'][timeframes[0]].index[-1]
    _config['env']['action_scheme'] = {'name': 'tensortrade.env.default.actions.TestActionScheme',
                                      'params': []}

    return _config



def track_to_symbol(track):
    ticks = track['df']['order_book']
    symbol = {'name': track['symbol'], 'feed': ticks.copy()}
    feed  = ticks.copy()

    # TODO: !! We use 0 here as a timeframe,
    # but we need to add it to constants

    timeframe = '0'
    symbol['quotes'] = {}
    symbol['timeframes'] = ['0']
    symbol['quotes'][timeframe] = feed
    symbol['quotes'][timeframe]['timestamp'] = pd.to_datetime(symbol['quotes'][timeframe]['timestamp'], unit="ms")
    symbol['quotes'][timeframe].set_index("timestamp", inplace=True)
    symbol['quotes'][timeframe].rename(columns={"bid": "close"}, inplace=True)
    symbol['quotes'][timeframe].drop(columns=["ask", "ask_normalized", "bid_normalized"], inplace=True)
    end_of_episode = pd.Series(np.full(len(symbol['quotes'][timeframe]), False))
    symbol['quotes'][timeframe] = symbol['quotes'][timeframe].assign(end_of_episode=end_of_episode.values).assign(code=pd.Series(np.full(len(symbol['quotes'][timeframe]), 0)).values)
    symbol['quotes'][timeframe].loc[symbol['quotes'][timeframe].index[-1], 'end_of_episode'] = True
    for name in ["open", "high", "low"]: symbol['quotes'][timeframe][name] = symbol['quotes'][timeframe]["close"]
    symbol['quotes'][timeframe]['volume'] = pd.Series(np.full(len(symbol['quotes'][timeframe]), 0)).values
    symbol['quotes'][timeframe] = symbol['quotes'][timeframe][['open', 'high', 'low', 'close', 'volume', 'end_of_episode', 'code']]
    symbol['code'] = 0
    return symbol


def test_track_to_symbol():
    tracks = load_tracks("../quanttools/datamart/tracks")
    s = track_to_symbol(tracks[0])
    print(s['quotes'][s['timeframes'][0]].columns.tolist())
    assert s['quotes'][s['timeframes'][0]].columns.tolist() == ['open', 'high', 'low', 'close', 'volume', 'end_of_episode', 'code']




def get_feed_from_track(track):
    pprint(track)


def test_get_feed_from_track():
    tracks = load_tracks("../quanttools/datamart/tracks")
    print(len(tracks))
    f = get_feed_from_track(tracks[0])
    print(f)


def test_buy_limit_entry_market_close(config):
    env_conf = econf.EnvConfig(config['env'])
    env = env_conf.build()
    obs_header = get_obs_header(env)
    ac = get_action_scheme(env)
    action = 8 # 0 - buy asset, 1 - sell asset, 2 - close_all (if u can) or do nothing
    obs,_ = env.reset()
    info = get_info(env)
    done = False
    observer = get_observer(env)
    track=[]
    instruments=[]
    volumes=[]
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))

    row_header = ['action', 'net_worth'] + instruments + ['end_of_episode', 'symbol_code', 'reward', 'leverage']
    track.append(np.concatenate(([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, 0, ac.leverage])))

    if get_param(config['params'], 'add_features_to_row')['value']:
        track[-1] = np.concatenate((track[-1], obs))

    entry_step = 0
    step = 0
    trade_num = 0
    while done == False and step < 3420:

        close_idx = obs_header.index(f"0_{config['env']['data']['timeframes'][0]}_close")
        cc = obs[close_idx]


        # print(f's: {step}')
        if abs(cc - 0.005) < 0.0005 and trade_num == 0:
            # BUY SL/TP
            action = 9
            trade_num += 1

        # elif step > 500:
        #     action = 9
        #     trade_num += 1

        # elif abs(cc - 0.01) < 0.0005 and trade_num == 1:
        #     # CLOSE ALL
        #     print(f'____action: 8')
        #     action = 8
        #     trade_num += 1

        elif step > 700:
            action = 8
            trade_num += 1

        # elif abs(cc - 0.011) < 0.0005 and trade_num == 2:
        #     # SELL_LIMIT
        #     print(f'_____action: 6')
        #     action = 6
        #     trade_num += 1


        # elif abs(cc - 0.005) < 0.0005 and trade_num == 3:
        #     # CLOSE ALL
        #     action = 8
        #     trade_num += 1

        if type(action) != int:
            raise Exception("Wrong type of action")

        if observer.end_of_episode == True:
            obs, reward, done, truncated, info = env.step(action)
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            step += 1
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        track.append(np.concatenate(
            ([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, reward, ac.leverage])))

        if get_param(config['params'], 'add_features_to_row')['value']:
            track[-1] = np.concatenate((track[-1], obs))

    track = pd.DataFrame(track, index=get_feed(env).index)
    if get_param(config['params'], 'add_features_to_row')['value']:
        track.columns = row_header + obs_header
    else:
        track.columns = row_header

    score = qs.stats.sharpe(track['net_worth'])
    plot_params = {'plot_on_main': []}
    lower_tmf = config['env']['data']['timeframes'][0]

    for c in track.columns:
        if f'0_{lower_tmf}_SMA_real' in c:
            plot_params['plot_on_main'].append({'name': c})


    assert len(get_action_scheme(env).broker.trades) > 0

    metrics =  {
                # "score": score,
                'track': track,
                'trades': get_action_scheme(env).broker.trades,
                # 'config': config,
                'plot_params': plot_params
               }

    plot(metrics)
    return

def test_buy_limit_entry_stop_loss(config):
    pass

def test_buy_limit_entry_take_profit(config):
    pass

def test_sell_limit_entry_market_close(config):
    pass

def test_sell_limit_entry_stop_loss(config):
    pass

def test_sell_limit_entry_take_profit(config):
    pass

def test_buy_market_entry_market_close(config):
    pass

def test_buy_market_entry_stop_loss(config):
    pass

def test_buy_market_entry_take_profit(config):
    pass

def test_sell_market_entry_market_close(config):
    pass

def test_sell_market_entry_stop_loss(config):
    pass

def test_sell_market_entry_take_profit(config):
    pass

def test_buy_stop_entry_market_close(config):
    pass

def test_buy_stop_entry_stop_loss(config):
    pass

def test_buy_stop_entry_take_profit(config):
    pass

def test_sell_stop_entry_market_close(config):
    pass

def test_sell_stop_entry_stop_loss(config):
    pass

def test_sell_stop_entry_take_profit(config):
    pass



def test_margin_call(config):
    from tensortrade.env.default.actions import StopLossPercent, TakeProfitPercent

    # _max, _min = synth['quotes'][synth['timeframes'][0]]['close'].max(), synth['quotes'][synth['timeframes'][0]]['close'].min()
    config['env']['action_scheme']['stop_loss_policy'] = StopLossPercent(percent=75)
    config['env']['action_scheme']['take_profit_policy'] = TakeProfitPercent(percent=95)
    env_conf = econf.EnvConfig(config['env'])
    env = env_conf.build()
    obs_header = get_obs_header(env)
    ac = get_action_scheme(env)
    # action = 0 # 0 - buy asset, 1 - sell asset
    action = 10 # 0 - buy asset, 1 - sell asset, 2 - close_all (if u can) or do nothing
    obs,_ = env.reset()
    info = get_info(env)
    done = False
    observer = get_observer(env)
    track=[]
    instruments=[]
    volumes=[]
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))

    row_header = ['action', 'net_worth'] + instruments + ['end_of_episode', 'symbol_code', 'reward', 'leverage']
    track.append(np.concatenate(([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, 0, ac.leverage])))

    if get_param(config['params'], 'add_features_to_row')['value']:
        track[-1] = np.concatenate((track[-1], obs))

    entry_step = 0
    step = 0
    trade_num = 0
    while done == False and step < 3420:

        close_idx = obs_header.index(f"0_{config['env']['data']['timeframes'][0]}_close")
        cc = obs[close_idx]

        print(f's: {step}')
        if step == 100:
            # BUY SL/TP
            action = 9


        if step == 269:
            print('lol')



        if type(action) != int:
            raise Exception("Wrong type of action")

        if observer.end_of_episode == True:
            obs, reward, done, truncated, info = env.step(action)
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            step += 1
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        track.append(np.concatenate(
            ([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, reward, ac.leverage])))

        if get_param(config['params'], 'add_features_to_row')['value']:
            track[-1] = np.concatenate((track[-1], obs))

    track = pd.DataFrame(track, index=get_feed(env).index[:len(track)])
    if get_param(config['params'], 'add_features_to_row')['value']:
        track.columns = row_header + obs_header
    else:
        track.columns = row_header

    score = qs.stats.sharpe(track['net_worth'])
    plot_params = {'plot_on_main': []}
    lower_tmf = config['env']['data']['timeframes'][0]

    for c in track.columns:
        if f'0_{lower_tmf}_SMA_real' in c:
            plot_params['plot_on_main'].append({'name': c})


    assert len(get_action_scheme(env).broker.trades) > 0

    metrics =  {
                # "score": score,
                'track': track,
                'trades': get_action_scheme(env).broker.trades,
                # 'config': config,
                'plot_params': plot_params
               }

    plot(metrics)
    return


def test_commissions():
    pass

def test_funding_rates():
    pass






def test_dynamic_leverage(config):
    # pprint(config)

    timeframes = ['15m']
    synth = make_synthetic_symbol({
        'name': 'AST0',
        'from': '2020-1-1',
        'to': '2020-1-5',
        'synthetic': True,
        'ohlcv': True,
        'code': 0,
        'timeframes': timeframes,
        'y_scale': 10000
    })

    tracks = load_tracks("../quanttools/datamart/tracks")
    plot_tracks = False
    tnum=1
    tracks[tnum]['df']['order_book']['datetime'] = pd.to_datetime(tracks[tnum]['df']['order_book']['timestamp'], unit='ms')
    s = track_to_symbol(tracks[tnum])

    config['env']['data']['symbols'] = [s]


    config['env']['data']['symbols'][0]['feed'] = None
    config['env']['data']['symbols'][0]['quotes'] = copy.deepcopy(synth['quotes'])
    config['env']['data']['symbols'][0]['timeframes'] = timeframes
    config['env']['data']['timeframes'] = timeframes
    config['env']['data']['from'] = config['env']['data']['symbols'][0]['quotes'][timeframes[0]].index[0]
    config['env']['data']['to'] = config['env']['data']['symbols'][0]['quotes'][timeframes[0]].index[-1]

    # print(config['env']['data']['symbols'][0])

    env_conf = econf.EnvConfig(config['env'])
    env = env_conf.build()
    obs_header = get_obs_header(env)
    ac = get_action_scheme(env)
    action = 2 # 0 - buy asset, 1 - sell asset, 2 - close_all (if u can) or do nothing
    obs,_ = env.reset()
    info = get_info(env)
    done = False
    observer = get_observer(env)
    track=[]
    instruments=[]
    volumes=[]
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))

    row_header = ['action', 'net_worth'] + instruments + ['end_of_episode', 'symbol_code', 'reward', 'leverage']
    track.append(np.concatenate(([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, 0, ac.leverage])))

    # if gp(config, 'add_features_to_row')['value']:
    if get_param(config['params'], 'add_features_to_row')['value']:
        track[-1] = np.concatenate((track[-1], obs))

    # get steps where to trade
    # entry = tracks[0]['df']['order_book']['timestamp'].get_loc(tracks[0]['df']['open']['timestamp'])
    ob = tracks[tnum]['df']['order_book']
    # idx = ob[ob['timestamp'] == tracks[0]['df']['open']['timestamp']].index

    trade_open = tracks[tnum]['df']['open']
    trade_close = tracks[tnum]['df']['close']


    if plot_tracks:
        import matplotlib.pyplot as plt
        plt.plot(tracks[tnum]['df']['order_book']['timestamp'], tracks[tnum]['df']['order_book']['ask'], color='blue')
        plt.plot(tracks[tnum]['df']['order_book']['timestamp'], tracks[tnum]['df']['order_book']['bid'], color='red')
        plt.plot(trade_open['timestamp'], trade_open['price'], marker='o', color='blue', markersize=5)
        plt.plot(trade_close['timestamp'], trade_close['price'], marker='o', color='red', markersize=5)
        plt.show()

    print(tracks[tnum]['direction'])
    entry_step = 0
    exit_step = len(ob)-1
    print(entry_step, exit_step)

    step = 0
    trade_num = 0
    while done == False and step < 3420:

        close_idx = obs_header.index(f'0_{timeframes[0]}_close')
        cc = obs[close_idx]

        if step == 506:
            print('step 506')

        # ac = get_action_scheme(env)
        if abs(cc - 0.005) < 0.0005 and trade_num == 0:
            # BUY
            action = 0
            trade_num += 1

        if abs(cc - 0.01) < 0.0005 and trade_num == 1:
            ac.leverage = 10
            # SELL
            action = 1
            trade_num += 1

        if abs(cc - 0.005) < 0.0005 and trade_num == 2:
            ac.leverage = 20
            # BUY
            action = 0
            trade_num += 1

        if abs(cc - 0.01) < 0.0005 and trade_num == 3:
            ac.leverage = 50
            # SELL
            action = 1
            trade_num += 1

        if abs(cc - 0.005) < 0.0005 and trade_num == 4:
            # ac.leverage = 50
            # BUY
            action = 2
            trade_num += 1


        if type(action) != int:
            raise Exception("Wrong type of action")

        if observer.end_of_episode == True:
            obs, reward, done, truncated, info = env.step(action)
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            step += 1
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        track.append(np.concatenate(
            ([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, reward, ac.leverage])))

        if get_param(config['params'], 'add_features_to_row')['value']:
            track[-1] = np.concatenate((track[-1], obs))

    track = pd.DataFrame(track, index=get_feed(env).index)
    if get_param(config['params'], 'add_features_to_row')['value']:
        track.columns = row_header + obs_header
    else:
        track.columns = row_header

    score = qs.stats.sharpe(track['net_worth'])
    plot_params = {'plot_on_main': []}
    lower_tmf = config['env']['data']['timeframes'][0]

    for c in track.columns:
        if f'0_{lower_tmf}_SMA_real' in c:
            plot_params['plot_on_main'].append({'name': c})

    metrics =  {
                # "score": score,
                'track': track,
                'trades': get_action_scheme(env).broker.trades,
                # 'config': config,
                'plot_params': plot_params
               }



    plot(metrics)
    return

def test_reverse(config):
    # pprint(config)

    timeframes = ['15m']
    synth = make_synthetic_symbol({
        'name': 'AST0',
        'from': '2020-1-1',
        'to': '2020-1-5',
        'synthetic': True,
        'ohlcv': True,
        'code': 0,
        'timeframes': timeframes,
        'y_scale': 10000
    })

    tracks = load_tracks("../quanttools/datamart/tracks")
    plot_tracks = False
    tnum = 1
    tracks[tnum]['df']['order_book']['datetime'] = pd.to_datetime(tracks[tnum]['df']['order_book']['timestamp'],
                                                                  unit='ms')
    s = track_to_symbol(tracks[tnum])

    config['env']['data']['symbols'] = [s]

    config['env']['data']['symbols'][0]['feed'] = None
    config['env']['data']['symbols'][0]['quotes'] = copy.deepcopy(synth['quotes'])
    config['env']['data']['symbols'][0]['timeframes'] = timeframes
    config['env']['data']['timeframes'] = timeframes
    config['env']['data']['from'] = config['env']['data']['symbols'][0]['quotes'][timeframes[0]].index[0]
    config['env']['data']['to'] = config['env']['data']['symbols'][0]['quotes'][timeframes[0]].index[-1]

    # print(config['env']['data']['symbols'][0])

    env_conf = econf.EnvConfig(config['env'])
    env = env_conf.build()
    obs_header = get_obs_header(env)
    ac = get_action_scheme(env)
    # action = 0 # 0 - buy asset, 1 - sell asset
    action = 2  # 0 - buy asset, 1 - sell asset, 2 - close_all (if u can) or do nothing
    obs, _ = env.reset()
    info = get_info(env)
    done = False
    observer = get_observer(env)
    track = []
    instruments = []
    volumes = []
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))

    row_header = ['action', 'net_worth'] + instruments + ['end_of_episode', 'symbol_code', 'reward', 'leverage']
    track.append(np.concatenate(
        ([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, 0, ac.leverage])))

    # if gp(config, 'add_features_to_row')['value']:
    if get_param(config['params'], 'add_features_to_row')['value']:
        track[-1] = np.concatenate((track[-1], obs))

    # get steps where to trade
    # entry = tracks[0]['df']['order_book']['timestamp'].get_loc(tracks[0]['df']['open']['timestamp'])
    ob = tracks[tnum]['df']['order_book']
    # idx = ob[ob['timestamp'] == tracks[0]['df']['open']['timestamp']].index

    trade_open = tracks[tnum]['df']['open']
    trade_close = tracks[tnum]['df']['close']

    if plot_tracks:
        import matplotlib.pyplot as plt
        plt.plot(tracks[tnum]['df']['order_book']['timestamp'], tracks[tnum]['df']['order_book']['ask'], color='blue')
        plt.plot(tracks[tnum]['df']['order_book']['timestamp'], tracks[tnum]['df']['order_book']['bid'], color='red')
        plt.plot(trade_open['timestamp'], trade_open['price'], marker='o', color='blue', markersize=5)
        plt.plot(trade_close['timestamp'], trade_close['price'], marker='o', color='red', markersize=5)
        plt.show()

    print(tracks[tnum]['direction'])
    entry_step = 0
    exit_step = len(ob) - 1
    print(entry_step, exit_step)

    step = 0
    trade_num = 0
    while done == False and step < 3420:

        close_idx = obs_header.index(f'0_{timeframes[0]}_close')
        cc = obs[close_idx]

        if step == 506:
            print('step 506')

        # ac = get_action_scheme(env)
        if abs(cc - 0.005) < 0.0005 and trade_num == 0:
            # BUY
            action = 0
            trade_num += 1

        if abs(cc - 0.01) < 0.0005 and trade_num == 1:
            # ac.leverage = 10
            # SELL
            action = 1
            trade_num += 1

        if abs(cc - 0.005) < 0.0005 and trade_num == 2:
            # ac.leverage = 20
            # BUY
            action = 0
            trade_num += 1

        if abs(cc - 0.01) < 0.0005 and trade_num == 3:
            # ac.leverage = 50
            # SELL
            action = 1
            trade_num += 1

        if abs(cc - 0.005) < 0.0005 and trade_num == 4:
            # ac.leverage = 50
            # BUY
            action = 2
            trade_num += 1

        if type(action) != int:
            raise Exception("Wrong type of action")

        if observer.end_of_episode == True:
            obs, reward, done, truncated, info = env.step(action)
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            step += 1
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
        track.append(np.concatenate(
            ([action, info['net_worth']], volumes,
             [observer.end_of_episode, observer.symbol_code, reward, ac.leverage])))

        if get_param(config['params'], 'add_features_to_row')['value']:
            track[-1] = np.concatenate((track[-1], obs))

    track = pd.DataFrame(track, index=get_feed(env).index)
    if get_param(config['params'], 'add_features_to_row')['value']:
        track.columns = row_header + obs_header
    else:
        track.columns = row_header

    score = qs.stats.sharpe(track['net_worth'])
    plot_params = {'plot_on_main': []}
    lower_tmf = config['env']['data']['timeframes'][0]

    for c in track.columns:
        if f'0_{lower_tmf}_SMA_real' in c:
            plot_params['plot_on_main'].append({'name': c})

    metrics = {
        # "score": score,
        'track': track,
        'trades': get_action_scheme(env).broker.trades,
        # 'config': config,
        'plot_params': plot_params
    }

    plot(metrics)
    return

def test_averaging(config):
    # pprint(config)

    timeframes = ['15m']
    synth = make_synthetic_symbol({
        'name': 'AST0',
        'from': '2020-1-1',
        'to': '2020-1-5',
        'synthetic': True,
        'ohlcv': True,
        'code': 0,
        'timeframes': timeframes
    })


    tracks = load_tracks("../quanttools/datamart/tracks")
    plot_tracks = False
    tnum=1
    tracks[tnum]['df']['order_book']['datetime'] = pd.to_datetime(tracks[tnum]['df']['order_book']['timestamp'], unit='ms')
    s = track_to_symbol(tracks[tnum])

    config['env']['data']['symbols'] = [s]


    config['env']['data']['symbols'][0]['feed'] = None
    config['env']['data']['symbols'][0]['quotes'] = copy.deepcopy(synth['quotes'])
    config['env']['data']['symbols'][0]['timeframes'] = timeframes
    config['env']['data']['timeframes'] = timeframes
    config['env']['data']['from'] = config['env']['data']['symbols'][0]['quotes'][timeframes[0]].index[0]
    config['env']['data']['to'] = config['env']['data']['symbols'][0]['quotes'][timeframes[0]].index[-1]

    # print(config['env']['data']['symbols'][0])

    env_conf = econf.EnvConfig(config['env'])
    env = env_conf.build()
    obs_header = get_obs_header(env)
    ac = get_action_scheme(env)
    # action = 0 # 0 - buy asset, 1 - sell asset
    action = 2 # 0 - buy asset, 1 - sell asset, 2 - close_all (if u can) or do nothing
    obs,_ = env.reset()
    info = get_info(env)
    done = False
    observer = get_observer(env)
    track=[]
    instruments=[]
    volumes=[]
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))

    row_header = ['action', 'net_worth'] + instruments + ['end_of_episode', 'symbol_code', 'reward', 'leverage']
    track.append(np.concatenate(([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, 0, ac.leverage])))

    # if gp(config, 'add_features_to_row')['value']:
    if get_param(config['params'], 'add_features_to_row')['value']:
        track[-1] = np.concatenate((track[-1], obs))

    # get steps where to trade
    # entry = tracks[0]['df']['order_book']['timestamp'].get_loc(tracks[0]['df']['open']['timestamp'])
    ob = tracks[tnum]['df']['order_book']
    # idx = ob[ob['timestamp'] == tracks[0]['df']['open']['timestamp']].index

    trade_open = tracks[tnum]['df']['open']
    trade_close = tracks[tnum]['df']['close']


    if plot_tracks:
        plt.plot(tracks[tnum]['df']['order_book']['timestamp'], tracks[tnum]['df']['order_book']['ask'], color='blue')
        plt.plot(tracks[tnum]['df']['order_book']['timestamp'], tracks[tnum]['df']['order_book']['bid'], color='red')
        plt.plot(trade_open['timestamp'], trade_open['price'], marker='o', color='blue', markersize=5)
        plt.plot(trade_close['timestamp'], trade_close['price'], marker='o', color='red', markersize=5)
        plt.show()

    print(tracks[tnum]['direction'])
    entry_step = 0
    exit_step = len(ob)-1
    print(entry_step, exit_step)

    step = 0
    trade_num = 0
    while done == False and step < 3420:

        close_idx = obs_header.index(f'0_{timeframes[0]}_close')
        cc = obs[close_idx]


        # ac = get_action_scheme(env)
        if abs(cc - 50) < 5 and trade_num == 0:
            # BUY
            action = 0
            trade_num += 1

        if abs(cc - 100) < 5 and trade_num == 1:
            # ac.leverage = 10
            # BUY
            action = 0
            trade_num += 1

        if abs(cc - 50) < 5 and trade_num == 2:
            # ac.leverage = 20
            # BUY
            action = 0
            trade_num += 1

        # if abs(cc - 100) < 5 and trade_num == 3:
        #     ac.leverage = 50
        #     # SELL
        #     action = 1
        #     trade_num += 1

        # if abs(cc - 50) < 5 and trade_num == 4:
        #     # ac.leverage = 50
        #     # SELL
        #     action = 2
        #     trade_num += 1


        if type(action) != int:
            raise Exception("Wrong type of action")

        if observer.end_of_episode == True:
            obs, reward, done, truncated, info = env.step(action)
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            step += 1
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
        # net_worth=volumes[0]
        # close_idx = obs_header.index(f'0_{timeframes[0]}_close')
        # cc = obs[close_idx]
        # for v in volumes[-(len(volumes)-1):]:
        #     # cc = close(obs)
        #     net_worth += v*cc


        # track.append(np.concatenate(([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, reward])))
        track.append(np.concatenate(
            ([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, reward, ac.leverage])))

        if get_param(config['params'], 'add_features_to_row')['value']:
            track[-1] = np.concatenate((track[-1], obs))

    track = pd.DataFrame(track, index=get_feed(env).index)
    if get_param(config['params'], 'add_features_to_row')['value']:
        track.columns = row_header + obs_header
    else:
        track.columns = row_header

    score = qs.stats.sharpe(track['net_worth'])
    plot_params = {'plot_on_main': []}
    lower_tmf = config['env']['data']['timeframes'][0]

    for c in track.columns:
        if f'0_{lower_tmf}_SMA_real' in c:
            plot_params['plot_on_main'].append({'name': c})

    # plot_params['plot_on_main'].append({'name': 'leverage'})

    metrics =  {
                # "score": score,
                'track': track,
                'trades': get_action_scheme(env).broker.trades,
                # 'config': config,
                'plot_params': plot_params
               }



    # print('----- plot ----- ')
    #
    # import matplotlib.pyplot as plt
    # # plt.plot(metrics['track'][['0_price', '0_0_SMA_real_3', '0_0_SMA_real_5']])
    # plt.plot(metrics['track'][['0_price']]) # '0_0_SMA_real_3', '0_0_SMA_real_5']])
    # plt.show()

    plot(metrics)
    return



def test_reducing(config):
    # pprint(config)

    timeframes = ['15m']
    synth = make_synthetic_symbol({
        'name': 'AST0',
        'from': '2020-1-1',
        'to': '2020-1-5',
        'synthetic': True,
        'ohlcv': True,
        'code': 0,
        'timeframes': timeframes
    })


    tracks = load_tracks("../quanttools/datamart/tracks")
    plot_tracks = False
    tnum=1
    tracks[tnum]['df']['order_book']['datetime'] = pd.to_datetime(tracks[tnum]['df']['order_book']['timestamp'], unit='ms')
    s = track_to_symbol(tracks[tnum])

    config['env']['data']['symbols'] = [s]
    config['env']['data']['symbols'][0]['feed'] = None
    config['env']['data']['symbols'][0]['quotes'] = copy.deepcopy(synth['quotes'])
    config['env']['data']['symbols'][0]['timeframes'] = timeframes
    config['env']['data']['timeframes'] = timeframes
    config['env']['data']['from'] = config['env']['data']['symbols'][0]['quotes'][timeframes[0]].index[0]
    config['env']['data']['to'] = config['env']['data']['symbols'][0]['quotes'][timeframes[0]].index[-1]
    get_param(config['env']['params'], 'leverage')['value'] = 10


    config['env']['action_scheme'] = {'name': 'tensortrade.env.default.actions.TestActionScheme',
                                      'params': []}

    env_conf = econf.EnvConfig(config['env'])
    env = env_conf.build()
    obs_header = get_obs_header(env)
    ac = get_action_scheme(env)
    action = 8
    obs,_ = env.reset()
    info = get_info(env)
    done = False
    observer = get_observer(env)
    track=[]
    instruments=[]
    volumes=[]
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))

    row_header = ['action', 'net_worth'] + instruments + ['end_of_episode', 'symbol_code', 'reward', 'leverage']
    track.append(np.concatenate(([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, 0, ac.leverage])))

    # if gp(config, 'add_features_to_row')['value']:
    if get_param(config['params'], 'add_features_to_row')['value']:
        track[-1] = np.concatenate((track[-1], obs))

    # get steps where to trade
    # entry = tracks[0]['df']['order_book']['timestamp'].get_loc(tracks[0]['df']['open']['timestamp'])
    ob = tracks[tnum]['df']['order_book']
    # idx = ob[ob['timestamp'] == tracks[0]['df']['open']['timestamp']].index

    trade_open = tracks[tnum]['df']['open']
    trade_close = tracks[tnum]['df']['close']

    print(tracks[tnum]['direction'])
    entry_step = 0
    exit_step = len(ob)-1
    print(entry_step, exit_step)

    step = 0
    trade_num = 0
    while done == False and step < 3420:
        close_idx = obs_header.index(f'0_{timeframes[0]}_close')
        cc = obs[close_idx]

        if abs(cc - 0.005) < 0.0005 and trade_num == 0:
            # BUY2
            action = 2
            trade_num += 1

        elif abs(cc - 0.01) < 0.0005 and trade_num == 1:
            # SELL1
            action = 1
            trade_num += 1

        elif step > 900 and trade_num == 2:
            action = 8

        # elif abs(cc - 0.005) < 0.0005 and trade_num == 2:
        #     # SELL2
        #     action = 3
        #     trade_num += 1
        #
        # elif abs(cc - 0.01) < 0.0005 and trade_num == 3:
        #     # BUY1
        #     action = 0
        #     trade_num += 1
        #
        # elif step > 500 and trade_num == 4:
        #     action = 8



        if observer.end_of_episode == True:
            obs, reward, done, truncated, info = env.step(action)
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            step += 1
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
        track.append(np.concatenate(
            ([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, reward, ac.leverage])))

        if get_param(config['params'], 'add_features_to_row')['value']:
            track[-1] = np.concatenate((track[-1], obs))

    track = pd.DataFrame(track, index=get_feed(env).index)
    if get_param(config['params'], 'add_features_to_row')['value']:
        track.columns = row_header + obs_header
    else:
        track.columns = row_header

    score = qs.stats.sharpe(track['net_worth'])
    plot_params = {'plot_on_main': []}
    lower_tmf = config['env']['data']['timeframes'][0]

    for c in track.columns:
        if f'0_{lower_tmf}_SMA_real' in c:
            plot_params['plot_on_main'].append({'name': c})

    # plot_params['plot_on_main'].append({'name': 'leverage'})

    metrics =  {
                # "score": score,
                'track': track,
                'trades': get_action_scheme(env).broker.trades,
                # 'config': config,
                'plot_params': plot_params
               }

    plot(metrics)
    return



def test_stop_loss(config):
    # pprint(config)

    timeframes = ['15m']
    synth = make_synthetic_symbol({
        'name': 'AST0',
        'from': '2020-1-1',
        'to': '2020-1-5',
        'synthetic': True,
        'ohlcv': True,
        'code': 0,
        'timeframes': timeframes
    })


    tracks = load_tracks("../quanttools/datamart/tracks")
    plot_tracks = False
    tnum=1
    tracks[tnum]['df']['order_book']['datetime'] = pd.to_datetime(tracks[tnum]['df']['order_book']['timestamp'], unit='ms')
    s = track_to_symbol(tracks[tnum])
    config['env']['data']['symbols'] = [s]
    config['env']['data']['symbols'][0]['feed'] = None
    config['env']['data']['symbols'][0]['quotes'] = copy.deepcopy(synth['quotes'])
    config['env']['data']['symbols'][0]['timeframes'] = timeframes
    config['env']['data']['timeframes'] = timeframes
    config['env']['data']['from'] = config['env']['data']['symbols'][0]['quotes'][timeframes[0]].index[0]
    config['env']['data']['to'] = config['env']['data']['symbols'][0]['quotes'][timeframes[0]].index[-1]

    # print(config['env']['data']['symbols'][0])

    env_conf = econf.EnvConfig(config['env'])
    env = env_conf.build()
    obs_header = get_obs_header(env)
    ac = get_action_scheme(env)
    # action = 0 # 0 - buy asset, 1 - sell asset
    action = 2 # 0 - buy asset, 1 - sell asset, 2 - close_all (if u can) or do nothing
    obs,_ = env.reset()
    info = get_info(env)
    done = False
    observer = get_observer(env)
    track=[]
    instruments=[]
    volumes=[]
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))

    row_header = ['action', 'net_worth'] + instruments + ['end_of_episode', 'symbol_code', 'reward', 'leverage']
    track.append(np.concatenate(([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, 0, ac.leverage])))

    # if gp(config, 'add_features_to_row')['value']:
    if get_param(config['params'], 'add_features_to_row')['value']:
        track[-1] = np.concatenate((track[-1], obs))

    # get steps where to trade
    # entry = tracks[0]['df']['order_book']['timestamp'].get_loc(tracks[0]['df']['open']['timestamp'])
    ob = tracks[tnum]['df']['order_book']
    # idx = ob[ob['timestamp'] == tracks[0]['df']['open']['timestamp']].index

    trade_open = tracks[tnum]['df']['open']
    trade_close = tracks[tnum]['df']['close']

    print(tracks[tnum]['direction'])
    entry_step = 0
    exit_step = len(ob)-1
    print(entry_step, exit_step)

    step = 0
    trade_num = 0
    while done == False and step < 3420:

        close_idx = obs_header.index(f'0_{timeframes[0]}_close')
        cc = obs[close_idx]


        # ac = get_action_scheme(env)
        if abs(cc - 50) < 5 and trade_num == 0:
            # BUY
            action = 0
            trade_num += 1

        if abs(cc - 100) < 5 and trade_num == 1:
            # ac.leverage = 10
            # BUY
            action = 0
            trade_num += 1

        if abs(cc - 50) < 5 and trade_num == 2:
            # ac.leverage = 20
            # BUY
            action = 0
            trade_num += 1

        # if abs(cc - 100) < 5 and trade_num == 3:
        #     ac.leverage = 50
        #     # SELL
        #     action = 1
        #     trade_num += 1

        # if abs(cc - 50) < 5 and trade_num == 4:
        #     # ac.leverage = 50
        #     # SELL
        #     action = 2
        #     trade_num += 1


        if type(action) != int:
            raise Exception("Wrong type of action")

        if observer.end_of_episode == True:
            obs, reward, done, truncated, info = env.step(action)
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            step += 1
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
        # net_worth=volumes[0]
        # close_idx = obs_header.index(f'0_{timeframes[0]}_close')
        # cc = obs[close_idx]
        # for v in volumes[-(len(volumes)-1):]:
        #     # cc = close(obs)
        #     net_worth += v*cc


        # track.append(np.concatenate(([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, reward])))
        track.append(np.concatenate(
            ([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, reward, ac.leverage])))

        if get_param(config['params'], 'add_features_to_row')['value']:
            track[-1] = np.concatenate((track[-1], obs))

    track = pd.DataFrame(track, index=get_feed(env).index)
    if get_param(config['params'], 'add_features_to_row')['value']:
        track.columns = row_header + obs_header
    else:
        track.columns = row_header

    score = qs.stats.sharpe(track['net_worth'])
    plot_params = {'plot_on_main': []}
    lower_tmf = config['env']['data']['timeframes'][0]

    for c in track.columns:
        if f'0_{lower_tmf}_SMA_real' in c:
            plot_params['plot_on_main'].append({'name': c})

    # plot_params['plot_on_main'].append({'name': 'leverage'})

    metrics =  {
                # "score": score,
                'track': track,
                'trades': get_action_scheme(env).broker.trades,
                # 'config': config,
                'plot_params': plot_params
               }

    plot(metrics)
    return


def test_take_profit(config):
    # pprint(config)

    timeframes = ['15m']
    synth = make_synthetic_symbol({
        'name': 'AST0',
        'from': '2020-1-1',
        'to': '2020-1-5',
        'synthetic': True,
        'ohlcv': True,
        'code': 0,
        'timeframes': timeframes
    })


    tracks = load_tracks("../quanttools/datamart/tracks")
    plot_tracks = False
    tnum=1
    tracks[tnum]['df']['order_book']['datetime'] = pd.to_datetime(tracks[tnum]['df']['order_book']['timestamp'], unit='ms')
    s = track_to_symbol(tracks[tnum])

    config['env']['data']['symbols'] = [s]
    config['env']['data']['symbols'][0]['feed'] = None
    config['env']['data']['symbols'][0]['quotes'] = copy.deepcopy(synth['quotes'])
    config['env']['data']['symbols'][0]['timeframes'] = timeframes
    config['env']['data']['timeframes'] = timeframes
    config['env']['data']['from'] = config['env']['data']['symbols'][0]['quotes'][timeframes[0]].index[0]
    config['env']['data']['to'] = config['env']['data']['symbols'][0]['quotes'][timeframes[0]].index[-1]
    config['env']['action_scheme'] = {'name': 'tensortrade.env.default.actions.TestActionScheme',
                                      'params': []}
    # print(config['env']['data']['symbols'][0])

    _max, _min = synth['quotes'][synth['timeframes'][0]]['close'].max(), synth['quotes'][synth['timeframes'][0]]['close'].min()
    env_conf = econf.EnvConfig(config['env'])
    env = env_conf.build()
    obs_header = get_obs_header(env)
    ac = get_action_scheme(env)
    # action = 0 # 0 - buy asset, 1 - sell asset
    action = 8 # 0 - buy asset, 1 - sell asset, 2 - close_all (if u can) or do nothing
    obs,_ = env.reset()
    info = get_info(env)
    done = False
    observer = get_observer(env)
    track=[]
    instruments=[]
    volumes=[]
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))

    row_header = ['action', 'net_worth'] + instruments + ['end_of_episode', 'symbol_code', 'reward', 'leverage']
    track.append(np.concatenate(([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, 0, ac.leverage])))

    # if gp(config, 'add_features_to_row')['value']:
    if get_param(config['params'], 'add_features_to_row')['value']:
        track[-1] = np.concatenate((track[-1], obs))

    # get steps where to trade
    # entry = tracks[0]['df']['order_book']['timestamp'].get_loc(tracks[0]['df']['open']['timestamp'])
    ob = tracks[tnum]['df']['order_book']
    # idx = ob[ob['timestamp'] == tracks[0]['df']['open']['timestamp']].index

    trade_open = tracks[tnum]['df']['open']
    trade_close = tracks[tnum]['df']['close']

    print(tracks[tnum]['direction'])
    entry_step = 0
    exit_step = len(ob)-1
    print(entry_step, exit_step)

    step = 0
    trade_num = 0
    while done == False and step < 3420:

        close_idx = obs_header.index(f'0_{timeframes[0]}_close')
        cc = obs[close_idx]


        print(f's: {step}')
        # ac = get_action_scheme(env)
        # if abs(cc - 0.005) < 0.0005 and trade_num == 0 and step > 377:
        if abs(cc - 0.005) < 0.0005 and trade_num == 0:
            # BUY SL/TP
            action = 9
            trade_num += 1

        # elif step > 500:
        #     action = 9
        #     trade_num += 1

        # elif abs(cc - 0.01) < 0.0005 and trade_num == 1:
        #     # CLOSE ALL
        #     print(f'____action: 8')
        #     action = 8
        #     trade_num += 1

        elif step > 700:
            action = 8
            trade_num += 1

        # elif abs(cc - 0.011) < 0.0005 and trade_num == 2:
        #     # SELL_LIMIT
        #     print(f'_____action: 6')
        #     action = 6
        #     trade_num += 1


        # elif abs(cc - 0.005) < 0.0005 and trade_num == 3:
        #     # CLOSE ALL
        #     action = 8
        #     trade_num += 1

        if type(action) != int:
            raise Exception("Wrong type of action")

        if observer.end_of_episode == True:
            obs, reward, done, truncated, info = env.step(action)
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            step += 1
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        track.append(np.concatenate(
            ([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, reward, ac.leverage])))

        if get_param(config['params'], 'add_features_to_row')['value']:
            track[-1] = np.concatenate((track[-1], obs))

    track = pd.DataFrame(track, index=get_feed(env).index)
    if get_param(config['params'], 'add_features_to_row')['value']:
        track.columns = row_header + obs_header
    else:
        track.columns = row_header

    score = qs.stats.sharpe(track['net_worth'])
    plot_params = {'plot_on_main': []}
    lower_tmf = config['env']['data']['timeframes'][0]

    for c in track.columns:
        if f'0_{lower_tmf}_SMA_real' in c:
            plot_params['plot_on_main'].append({'name': c})

    # plot_params['plot_on_main'].append({'name': 'leverage'})


    assert len(get_action_scheme(env).broker.trades) > 0

    metrics =  {
                # "score": score,
                'track': track,
                'trades': get_action_scheme(env).broker.trades,
                # 'config': config,
                'plot_params': plot_params
               }

    plot(metrics)
    return


def test_linked(config):

    from tensortrade.env.default.actions import StopLossPercent, TakeProfitPercent
    config['env']['action_scheme']['stop_loss_policy'] = StopLossPercent(percent=25)
    config['env']['action_scheme']['take_profit_policy'] = TakeProfitPercent(percent=35)
    env_conf = econf.EnvConfig(config['env'])
    env = env_conf.build()
    obs_header = get_obs_header(env)
    ac = get_action_scheme(env)
    # action = 0 # 0 - buy asset, 1 - sell asset
    action = 10 # 0 - buy asset, 1 - sell asset, 2 - close_all (if u can) or do nothing
    obs,_ = env.reset()
    info = get_info(env)
    done = False
    observer = get_observer(env)
    track=[]
    instruments=[]
    volumes=[]
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))

    row_header = ['action', 'net_worth'] + instruments + ['end_of_episode', 'symbol_code', 'reward', 'leverage']
    track.append(np.concatenate(([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, 0, ac.leverage])))

    if get_param(config['params'], 'add_features_to_row')['value']:
        track[-1] = np.concatenate((track[-1], obs))

    entry_step = 0
    step = 0
    trade_num = 0
    while done == False and step < 3420:

        close_idx = obs_header.index(f"0_{config['env']['data']['timeframes'][0]}_close")
        cc = obs[close_idx]

        if abs(cc - 0.005) < 0.0005 and trade_num == 0:
            # BUY SL/TP
            action = 9
            trade_num += 1

        # elif step > 500:
        #     action = 9
        #     trade_num += 1

        # elif abs(cc - 0.01) < 0.0005 and trade_num == 1:
        #     # CLOSE ALL
        #     print(f'____action: 8')
        #     action = 8
        #     trade_num += 1

        elif step > 700:
            action = 10
            trade_num += 1

        # elif abs(cc - 0.011) < 0.0005 and trade_num == 2:
        #     # SELL_LIMIT
        #     print(f'_____action: 6')
        #     action = 6
        #     trade_num += 1


        # elif abs(cc - 0.005) < 0.0005 and trade_num == 3:
        #     # CLOSE ALL
        #     action = 8
        #     trade_num += 1

        if type(action) != int:
            raise Exception("Wrong type of action")

        if observer.end_of_episode == True:
            obs, reward, done, truncated, info = env.step(action)
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            step += 1
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        track.append(np.concatenate(
            ([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, reward, ac.leverage])))

        if get_param(config['params'], 'add_features_to_row')['value']:
            track[-1] = np.concatenate((track[-1], obs))

    track = pd.DataFrame(track, index=get_feed(env).index)
    if get_param(config['params'], 'add_features_to_row')['value']:
        track.columns = row_header + obs_header
    else:
        track.columns = row_header

    score = qs.stats.sharpe(track['net_worth'])
    plot_params = {'plot_on_main': []}
    lower_tmf = config['env']['data']['timeframes'][0]

    for c in track.columns:
        if f'0_{lower_tmf}_SMA_real' in c:
            plot_params['plot_on_main'].append({'name': c})


    assert len(get_action_scheme(env).broker.trades) > 0

    metrics =  {
                # "score": score,
                'track': track,
                'trades': get_action_scheme(env).broker.trades,
                # 'config': config,
                'plot_params': plot_params
               }

    plot(metrics)
    return


def test_limit_entry(config):
    # pprint(config)

    timeframes = ['15m']
    synth = make_synthetic_symbol({
        'name': 'AST0',
        'from': '2020-1-1',
        'to': '2020-1-5',
        'synthetic': True,
        'ohlcv': True,
        'code': 0,
        'timeframes': timeframes
    })


    tracks = load_tracks("../quanttools/datamart/tracks")
    plot_tracks = False
    tnum=1
    tracks[tnum]['df']['order_book']['datetime'] = pd.to_datetime(tracks[tnum]['df']['order_book']['timestamp'], unit='ms')
    s = track_to_symbol(tracks[tnum])

    config['env']['data']['symbols'] = [s]


    config['env']['data']['symbols'][0]['feed'] = None
    config['env']['data']['symbols'][0]['quotes'] = copy.deepcopy(synth['quotes'])
    config['env']['data']['symbols'][0]['timeframes'] = timeframes
    config['env']['data']['timeframes'] = timeframes
    config['env']['data']['from'] = config['env']['data']['symbols'][0]['quotes'][timeframes[0]].index[0]
    config['env']['data']['to'] = config['env']['data']['symbols'][0]['quotes'][timeframes[0]].index[-1]

    config['env']['action_scheme'] = {'name': 'tensortrade.env.default.actions.TestActionScheme',
                                      'params': []}
    # print(config['env']['data']['symbols'][0])

    _max, _min = synth['quotes'][synth['timeframes'][0]]['close'].max(), synth['quotes'][synth['timeframes'][0]]['close'].min()
    env_conf = econf.EnvConfig(config['env'])
    env = env_conf.build()
    obs_header = get_obs_header(env)
    ac = get_action_scheme(env)
    # action = 0 # 0 - buy asset, 1 - sell asset
    action = 4 # 0 - buy asset, 1 - sell asset, 2 - close_all (if u can) or do nothing
    obs,_ = env.reset()
    info = get_info(env)
    done = False
    observer = get_observer(env)
    track=[]
    instruments=[]
    volumes=[]
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))

    row_header = ['action', 'net_worth'] + instruments + ['end_of_episode', 'symbol_code', 'reward', 'leverage']
    track.append(np.concatenate(([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, 0, ac.leverage])))

    # if gp(config, 'add_features_to_row')['value']:
    if get_param(config['params'], 'add_features_to_row')['value']:
        track[-1] = np.concatenate((track[-1], obs))

    # get steps where to trade
    # entry = tracks[0]['df']['order_book']['timestamp'].get_loc(tracks[0]['df']['open']['timestamp'])
    ob = tracks[tnum]['df']['order_book']
    # idx = ob[ob['timestamp'] == tracks[0]['df']['open']['timestamp']].index

    trade_open = tracks[tnum]['df']['open']
    trade_close = tracks[tnum]['df']['close']

    print(tracks[tnum]['direction'])
    entry_step = 0
    exit_step = len(ob)-1
    print(entry_step, exit_step)

    step = 0
    trade_num = 0
    while done == False and step < 3420:

        close_idx = obs_header.index(f'0_{timeframes[0]}_close')
        cc = obs[close_idx]


        print(f's: {step}')
        # ac = get_action_scheme(env)
        # if abs(cc - 0.005) < 0.0005 and trade_num == 0 and step > 377:
        if abs(cc - 0.005) < 0.0005 and trade_num == 0:
            # BUY_LIMIT
            action = 4
            trade_num += 1

        elif abs(cc - 0.01) < 0.0005 and trade_num == 1:
            # CLOSE ALL
            print(f'____action: 8')
            action = 8
            trade_num += 1

        elif abs(cc - 0.011) < 0.0005 and trade_num == 2:
            # SELL_LIMIT
            print(f'_____action: 6')
            action = 6
            trade_num += 1


        elif abs(cc - 0.005) < 0.0005 and trade_num == 3:
            # CLOSE ALL
            action = 8
            trade_num += 1

        if type(action) != int:
            raise Exception("Wrong type of action")

        if observer.end_of_episode == True:
            obs, reward, done, truncated, info = env.step(action)
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            step += 1
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        track.append(np.concatenate(
            ([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, reward, ac.leverage])))

        if get_param(config['params'], 'add_features_to_row')['value']:
            track[-1] = np.concatenate((track[-1], obs))

    track = pd.DataFrame(track, index=get_feed(env).index)
    if get_param(config['params'], 'add_features_to_row')['value']:
        track.columns = row_header + obs_header
    else:
        track.columns = row_header

    score = qs.stats.sharpe(track['net_worth'])
    plot_params = {'plot_on_main': []}
    lower_tmf = config['env']['data']['timeframes'][0]

    for c in track.columns:
        if f'0_{lower_tmf}_SMA_real' in c:
            plot_params['plot_on_main'].append({'name': c})

    # plot_params['plot_on_main'].append({'name': 'leverage'})


    assert len(get_action_scheme(env).broker.trades) > 0

    metrics =  {
                # "score": score,
                'track': track,
                'trades': get_action_scheme(env).broker.trades,
                # 'config': config,
                'plot_params': plot_params
               }

    plot(metrics)
    return

def test_stop_entry(config):
    # pprint(config)

    timeframes = ['15m']
    synth = make_synthetic_symbol({
        'name': 'AST0',
        'from': '2020-1-1',
        'to': '2020-1-5',
        'synthetic': True,
        'ohlcv': True,
        'code': 0,
        'timeframes': timeframes
    })


    tracks = load_tracks("../quanttools/datamart/tracks")
    plot_tracks = False
    tnum=1
    tracks[tnum]['df']['order_book']['datetime'] = pd.to_datetime(tracks[tnum]['df']['order_book']['timestamp'], unit='ms')
    s = track_to_symbol(tracks[tnum])
    config['env']['data']['symbols'] = [s]
    config['env']['data']['symbols'][0]['feed'] = None
    config['env']['data']['symbols'][0]['quotes'] = copy.deepcopy(synth['quotes'])
    config['env']['data']['symbols'][0]['timeframes'] = timeframes
    config['env']['data']['timeframes'] = timeframes
    config['env']['data']['from'] = config['env']['data']['symbols'][0]['quotes'][timeframes[0]].index[0]
    config['env']['data']['to'] = config['env']['data']['symbols'][0]['quotes'][timeframes[0]].index[-1]

    # print(config['env']['data']['symbols'][0])

    env_conf = econf.EnvConfig(config['env'])
    env = env_conf.build()
    obs_header = get_obs_header(env)
    ac = get_action_scheme(env)
    # action = 0 # 0 - buy asset, 1 - sell asset
    action = 2 # 0 - buy asset, 1 - sell asset, 2 - close_all (if u can) or do nothing
    obs,_ = env.reset()
    info = get_info(env)
    done = False
    observer = get_observer(env)
    track=[]
    instruments=[]
    volumes=[]
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))

    row_header = ['action', 'net_worth'] + instruments + ['end_of_episode', 'symbol_code', 'reward', 'leverage']
    track.append(np.concatenate(([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, 0, ac.leverage])))

    # if gp(config, 'add_features_to_row')['value']:
    if get_param(config['params'], 'add_features_to_row')['value']:
        track[-1] = np.concatenate((track[-1], obs))

    # get steps where to trade
    # entry = tracks[0]['df']['order_book']['timestamp'].get_loc(tracks[0]['df']['open']['timestamp'])
    ob = tracks[tnum]['df']['order_book']
    # idx = ob[ob['timestamp'] == tracks[0]['df']['open']['timestamp']].index

    trade_open = tracks[tnum]['df']['open']
    trade_close = tracks[tnum]['df']['close']

    print(tracks[tnum]['direction'])
    entry_step = 0
    exit_step = len(ob)-1
    print(entry_step, exit_step)

    step = 0
    trade_num = 0
    while done == False and step < 3420:

        close_idx = obs_header.index(f'0_{timeframes[0]}_close')
        cc = obs[close_idx]


        # ac = get_action_scheme(env)
        if abs(cc - 50) < 5 and trade_num == 0:
            # BUY
            action = 0
            trade_num += 1

        if abs(cc - 100) < 5 and trade_num == 1:
            # ac.leverage = 10
            # BUY
            action = 0
            trade_num += 1

        if abs(cc - 50) < 5 and trade_num == 2:
            # ac.leverage = 20
            # BUY
            action = 0
            trade_num += 1

        # if abs(cc - 100) < 5 and trade_num == 3:
        #     ac.leverage = 50
        #     # SELL
        #     action = 1
        #     trade_num += 1

        # if abs(cc - 50) < 5 and trade_num == 4:
        #     # ac.leverage = 50
        #     # SELL
        #     action = 2
        #     trade_num += 1


        if type(action) != int:
            raise Exception("Wrong type of action")

        if observer.end_of_episode == True:
            obs, reward, done, truncated, info = env.step(action)
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            step += 1
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
        # net_worth=volumes[0]
        # close_idx = obs_header.index(f'0_{timeframes[0]}_close')
        # cc = obs[close_idx]
        # for v in volumes[-(len(volumes)-1):]:
        #     # cc = close(obs)
        #     net_worth += v*cc


        # track.append(np.concatenate(([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, reward])))
        track.append(np.concatenate(
            ([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, reward, ac.leverage])))

        if get_param(config['params'], 'add_features_to_row')['value']:
            track[-1] = np.concatenate((track[-1], obs))

    track = pd.DataFrame(track, index=get_feed(env).index)
    if get_param(config['params'], 'add_features_to_row')['value']:
        track.columns = row_header + obs_header
    else:
        track.columns = row_header

    score = qs.stats.sharpe(track['net_worth'])
    plot_params = {'plot_on_main': []}
    lower_tmf = config['env']['data']['timeframes'][0]

    for c in track.columns:
        if f'0_{lower_tmf}_SMA_real' in c:
            plot_params['plot_on_main'].append({'name': c})

    # plot_params['plot_on_main'].append({'name': 'leverage'})

    metrics =  {
                # "score": score,
                'track': track,
                'trades': get_action_scheme(env).broker.trades,
                # 'config': config,
                'plot_params': plot_params
               }

    plot(metrics)
    return




def test_tracks(config):
    # agent = get_agent(config['agents'][config.get('agent_num', 0)])
    # config['env']['data']['features'] = agent.get_features()
    #
    #

    tracks = load_tracks("../quanttools/datamart/tracks")
    plot_tracks = False
    tnum=1
    tracks[tnum]['df']['order_book']['datetime'] = pd.to_datetime(tracks[tnum]['df']['order_book']['timestamp'], unit='ms')
    s = track_to_symbol(tracks[tnum])

    config['env']['data']['symbols'] = [s]

    pprint(options['DOGEUSDT']['funding_rates'][-1])
    env_conf = econf.EnvConfig(config['env'])
    env = env_conf.build()
    obs_header = get_obs_header(env)
    action = 0 # 0 - buy asset, 1 - sell asset
    obs,_ = env.reset()
    info = get_info(env)
    done = False
    observer = get_observer(env)
    track=[]
    instruments=[]
    volumes=[]
    for w in get_action_scheme(env).portfolio.wallets:
        balance = w.total_balance
        instruments.append(str(balance.instrument))
        volumes.append(float(balance.size))

    row_header = ['action', 'net_worth'] + instruments + ['end_of_episode', 'symbol_code', 'reward']
    track.append(np.concatenate(([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, 0])))

    # if gp(config, 'add_features_to_row')['value']:
    if get_param(config['params'], 'add_features_to_row')['value']:
        track[-1] = np.concatenate((track[-1], obs))

    # get steps where to trade
    # entry = tracks[0]['df']['order_book']['timestamp'].get_loc(tracks[0]['df']['open']['timestamp'])
    ob = tracks[tnum]['df']['order_book']
    # idx = ob[ob['timestamp'] == tracks[0]['df']['open']['timestamp']].index

    trade_open = tracks[tnum]['df']['open']
    trade_close = tracks[tnum]['df']['close']


    if plot_tracks:
        import matplotlib.pyplot as plt
        plt.plot(tracks[tnum]['df']['order_book']['timestamp'], tracks[tnum]['df']['order_book']['ask'], color='blue')
        plt.plot(tracks[tnum]['df']['order_book']['timestamp'], tracks[tnum]['df']['order_book']['bid'], color='red')
        plt.plot(trade_open['timestamp'], trade_open['price'], marker='o', color='blue', markersize=5)
        plt.plot(trade_close['timestamp'], trade_close['price'], marker='o', color='red', markersize=5)
        plt.show()
    # return



    print(tracks[tnum]['direction'])
    entry_step = 0
    exit_step = len(ob)-1
    print(entry_step, exit_step)
    # return
    # for index, row in ob.iterrows():
    #     delta = row['timestamp'] - trade_open['timestamp']
    #     if delta >= 0 and row['ask'] == trade_open['price'] and entry_step == -1:
    #         # get entry step
    #         entry_step = ob.index.get_loc(index)
    #         print('got entry_step: ', entry_step, ' with delta ', delta)



    #     delta = row['timestamp'] - trade_close['timestamp']
    #     print('exit delta: ', delta)
    #     if delta >= 0 and row['bid'] == trade_close['price'] and exit_step == -1:
    #         # get entry step
    #         exit_step = ob.index.get_loc(index)
    #         print('got exit_step: ', exit_step, ' with delta ', delta)


    # exit = tracks[tnum]['df']['open']['timestamp']

    step = 0
    while done == False and step < 3420:
        # action = agent.get_action(obs, header=obs_header)
        #
        if step == entry_step:
            action = 0 # actually here we need kind of get action

        if step == exit_step:
            action = 1
        if type(action) != int:
            raise Exception("Wrong type of action")

        if observer.end_of_episode == True:
            obs, reward, done, truncated, info = env.step(action)
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
            step += 1
        else:
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)

        volumes = get_wallets_volumes(get_action_scheme(env).portfolio.wallets)
        net_worth=volumes[0]
        close_idx = obs_header.index('0_0_close')
        cc = obs[close_idx]
        for v in volumes[-(len(volumes)-1):]:
            # cc = close(obs)
            net_worth += v*cc

        track.append(np.concatenate(([action, info['net_worth']], volumes, [observer.end_of_episode, observer.symbol_code, reward])))
        if get_param(config['params'], 'add_features_to_row')['value']:
            track[-1] = np.concatenate((track[-1], obs))

    track = pd.DataFrame(track, index=get_feed(env).index)
    if get_param(config['params'], 'add_features_to_row')['value']:
        track.columns = row_header + obs_header
    else:
        track.columns = row_header

    score = qs.stats.sharpe(track['net_worth'])
    plot_params = {'plot_on_main': []}
    lower_tmf = config['env']['data']['timeframes'][0]

    for c in track.columns:
        if f'0_{lower_tmf}_SMA_real' in c:
            plot_params['plot_on_main'].append({'name': c})

    metrics =  {"score": score,
                'track': track,
                'trades': get_action_scheme(env).broker.trades,
                'config': config,
                'plot_params': plot_params
               }



    print('----- plot ----- ')

    import matplotlib.pyplot as plt
    plt.plot(metrics['track'][['0_price', '0_0_SMA_real_3', '0_0_SMA_real_5']])
    plt.show()

    # return
    plot(metrics)

    print('2232')
    return
    # 5. simulate

    # 6. compare tracks

if __name__ == "__main__":
    # test_track_to_symbol()
    # test_get_feed_from_track()
    # test_derivatives()
    # test_reverse(config())
    # test_dynamic_leverage(config())

    # TODO: 
    # test_reducing(config())
    # test_averaging(config())

    # test_stop_entry(config())
    # test_limit_entry(config())
    # test_stop_loss(config())
    # test_take_profit(config())

    test_linked(action_test_config())
    # test_margin_call(action_test_config())
    # test_commissions()
    # test_funding_rates()






