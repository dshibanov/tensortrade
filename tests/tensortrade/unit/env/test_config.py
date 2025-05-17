import sys
sys.path.append('./')
from tensortrade.env.config import *


def test_get_agent():

    timeframes = ['15m', '1h']
    conf = {'name': 'tensortrade.agents.sma_cross.SMACross',
             'params': [{'name': 'n_actions', 'value': 2},
                        {'name': 'observation_space_shape', 'value': (10,1)},
                        {'name': 'fast_ma', 'value': 3, 'optimize': True, 'lower': 2,
                         'upper': 5},
                        {'name': 'slow_ma', 'value': 5, 'optimize': True, 'lower': 5,
                         'upper': 11},
                        {'name': 'timeframes', 'value': timeframes}]}
    a = get_agent(conf)
