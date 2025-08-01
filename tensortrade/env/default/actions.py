import logging
from abc import abstractmethod
from itertools import product
from typing import Union, List, Any

# from gym.spaces import Space, Discrete
from gymnasium.spaces import Space, Discrete

from tensortrade.core import Clock
from tensortrade.env.generic import ActionScheme, TradingEnv
from tensortrade.oms.instruments import ExchangePair
from tensortrade.oms.orders import (
    Broker,
    Order,
    OrderListener,
    OrderSpec,
    proportion_order,
    risk_managed_order,
    market_order,
    derivative_order,
    TradeSide,
    TradeType
)
from tensortrade.oms.wallets import Portfolio
from quantutils.parameters import get_param

from abc import ABC, abstractmethod
from decimal import Decimal

class PlacementPolicy(ABC):
    def __init__(self):
        pass



class StopLossPercent(PlacementPolicy):
    def __init__(self,**kwargs):
        self.percent = kwargs.get('percent', 5)

    def __call__(self, price = None, side = None):
        k = self.percent * Decimal('0.01')
        value = (price * (1 + k)) if side == TradeSide.SELL else (price * (1 - k))
        return value
        # return (price*(1 + self.percent*Decimal('0.01'))) if side == TradeSide.SELL else (price*(1 - self.percent*Decimal('0.01')))

class TakeProfitPercent(PlacementPolicy):
    def __init__(self,**kwargs):
        self.percent = kwargs.get('percent', 5)

    def __call__(self, price = None, side = None):
        k = self.percent*Decimal('0.01')
        value = (price*(1+k)) if side == TradeSide.BUY else (price*(1 - k))
        return value


class LimitEntrySimple(PlacementPolicy):
    def __init__(self,**kwargs):
        self.percent = kwargs.get('percent', 5)

    def __call__(self):
        a = 5
        return a


class StopEntrySimple(PlacementPolicy):
    def __init__(self,**kwargs):
        self.percent = kwargs.get('percent', 5)

    def __call__(self):
        a = 5
        return a

class StopLossWickBelowPriorPeak(PlacementPolicy):
    def __init__(self,**kwargs):
        pass

    def __call__(self):
        a = 5
        return a


class TensorTradeActionScheme(ActionScheme):
    """An abstract base class for any `ActionScheme` that wants to be
    compatible with the built in OMS.

    The structure of the action scheme is built to make sure that action space
    can be used with the system, provided that the user defines the methods to
    interpret that action.

    Attributes
    ----------
    portfolio : 'Portfolio'
        The portfolio object to be used in defining actions.
    broker : 'Broker'
        The broker object to be used for placing orders in the OMS.

    Methods
    -------
    perform(env,portfolio)
        Performs the action on the given environment.
    get_orders(action,portfolio)
        Gets the list of orders to be submitted for the given action.
    """

    def __init__(self) -> None:
        super().__init__()
        self.portfolio: 'Portfolio' = None
        self.broker: 'Broker' = Broker()
        self.broker.action_scheme = self

    @property
    def active_limits(self):
        if self.broker.unexecuted:
            return [u for u in self.broker.unexecuted if u.is_limit_order and u.is_active]
        else:
            return []


    @property
    def clock(self) -> 'Clock':
        """The reference clock from the environment. (`Clock`)

        When the clock is set for the we also set the clock for the portfolio
        as well as the exchanges defined in the portfolio.

        Returns
        -------
        `Clock`
            The environment clock.
        """
        return self._clock

    @clock.setter
    def clock(self, clock: 'Clock') -> None:
        self._clock = clock

        components = [self.portfolio] + self.portfolio.exchanges
        for c in components:
            c.clock = clock
        self.broker.clock = clock

    def perform(self, env: 'TradingEnv', action: Any) -> None:
        """Performs the action on the given environment.

        Under the TT action scheme, the subclassed action scheme is expected
        to provide a method for getting a list of orders to be submitted to
        the broker for execution in the OMS.

        Parameters
        ----------
        env : 'TradingEnv'
            The environment to perform the action on.
        action : Any
            The specific action selected from the action space.
        """
        orders = self.get_orders(action, self.portfolio)

        for order in orders:
            if order:
                logging.info('Step {}: {} {}'.format(order.step, order.side, order.quantity))
                self.broker.submit(order)

        self.broker.update()

    @abstractmethod
    def get_orders(self, action: Any, portfolio: 'Portfolio') -> 'List[Order]':
        """Gets the list of orders to be submitted for the given action.

        Parameters
        ----------
        action : Any
            The action to be interpreted.
        portfolio : 'Portfolio'
            The portfolio defined for the environment.

        Returns
        -------
        List[Order]
            A list of orders to be submitted to the broker.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Resets the action scheme."""
        self.portfolio.reset()
        self.broker.reset()


class BSH(TensorTradeActionScheme):
    """A simple discrete action scheme where the only options are to buy, sell,
    or hold.

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base intrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    """

    registered_name = "bsh"

    def __init__(self, cash: 'Wallet', asset: 'Wallet'):
        super().__init__()
        self.cash = cash
        self.asset = asset

        self.listeners = []
        self.action = 0

    @property
    def action_space(self):
        return Discrete(2)

    def attach(self, listener):
        self.listeners += [listener]
        return self

    def get_orders(self, action: int, portfolio: 'Portfolio') -> 'Order':
        order = None

        if abs(action - self.action) > 0:
            src = self.cash if self.action == 0 else self.asset
            tgt = self.asset if self.action == 0 else self.cash

            if src.balance == 0:  # We need to check, regardless of the proposed order, if we have balance in 'src'
                return []  # Otherwise just return an empty order list

            order = proportion_order(portfolio, src, tgt, 1.0)
            self.action = action

        for listener in self.listeners:
            listener.on_action(action)

        return [order]

    def reset(self):
        super().reset()
        self.action = 0

class MultySymbolBSH(TensorTradeActionScheme):
    """BSH for multiple symbols environments.

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base intrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    """

    registered_name = "multy_symbol_bsh"

    # def __init__(self, cash: 'Wallet', asset: 'Wallet'):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.listeners = []
        self.action = 1
        self.started = False


    @property
    def action_space(self):
        return Discrete(2)

    def attach(self, listener):
        self.listeners += [listener]
        return self

    def get_orders(self, action: int, portfolio: 'Portfolio') -> 'Order':
        order = None
        current_symbol_code = self.config["current_symbol_code"]
        cash = portfolio.wallets[0]
        asset = portfolio.wallets[current_symbol_code+1]
        if ((self.started == False) or (abs(action - self.action) > 0)):
            self.started = True
            src = cash if action == 0 else asset
            tgt = asset if action == 0 else cash
            if src.balance == 0:  # We need to check, regardless of the proposed order, if we have balance in 'src'
                return []  # Otherwise just return an empty order list
            order = proportion_order(portfolio, src, tgt, 1.0)
            self.action = action

        for listener in self.listeners:
            listener.on_action(action)

        return [order]

    def force_sell(self):
        # *forced by episode ending
        action = 1
        orders = self.get_orders(action, self.portfolio)

        for order in orders:
            if order:
                logging.info('Step {}: {} {}'.format(order.step, order.side, order.quantity))
                self.broker.submit(order)

        self.broker.update()

    def reset(self):
        super().reset()
        self.action = 1
        self.started = False


class TestActionScheme(MultySymbolBSH):
    """ For testing purposes
        Action scheme for multiple symbols environments with margin trade support

        0: BUY1
            buy with self.amount and self.leverage

        1: SELL1
            sell with self.amount and self.leverage

        2: BUY2
            buy with self.amount*2 and self.leverage

        3: SELL2
            sell with self.amount*2 and self.leverage

        4: BUY_LIMIT
            limit entry buy with self.amount and self.leverage

        5: BUY_STOP
            stop entry buy with self.amount and self.leverage

        6: SELL_LIMIT
            limit entry sell with self.amount and self.leverage

        7: SELL_STOP
            stop entry sell with self.amount and self.leverage

        8: CLOSE
            close all

        * scheme allows to hold not more then one contract per symbol!!

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base intrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    """

    registered_name = "test_action_scheme"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._leverage = config.get('leverage', {'value': 5})['value']
        self.listeners = []
        self.action = 2
        self.started = False
        self.amount = config.get('amount', 10)

        self.stop_loss_policy = config['action_scheme'].get('stop_loss_policy', StopLossPercent(percent=5))
        self.take_profit_policy = config['action_scheme'].get('take_profit_policy', TakeProfitPercent(percent=5))
        self.limit_entry_policy = config['action_scheme'].get('limit_entry_policy', LimitEntrySimple(percent=5))
        self.stop_entry_policy = config['action_scheme'].get('stop_entry_policy', StopEntrySimple(percent=5))
        self.last_action = None

    @property
    def action_space(self):
        return Discrete(3)

    def attach(self, listener):
        self.listeners += [listener]
        return self

    @property
    def leverage(self):
        """Getter for price"""
        return self._leverage

    @leverage.setter
    def leverage(self, value: int):
        """Setter for price"""
        if value < 0:
            raise ValueError("Leverage cannot be negative")
        self._leverage = value

    def get_orders(self, action: int, portfolio: 'Portfolio') -> 'Order':

        if self.last_action and self.last_action == action and action not in [4,5,6,7,9]:
            return []
        else:
            self.last_action = action

        print(f'Ab: {action}')

        order = None
        current_symbol_code = self.config["current_symbol_code"]
        cash = portfolio.wallets[0]
        asset = portfolio.wallets[current_symbol_code+1]
        contracts_num = len(portfolio.contracts)

        if contracts_num > 0:
            contract = portfolio.contracts[0]
        elif contracts_num > 1:
            print('So much contracts')
            raise Exception(f'contracts_num > 1: {contracts_num}')

        self.action = action

        order = None
        current_symbol_code = self.config["current_symbol_code"]
        cash = portfolio.wallets[0]
        asset = portfolio.wallets[current_symbol_code+1]
        pair = portfolio.exchange_pairs[current_symbol_code]

        print(f'A: {action}')
        match action:
            case 0:
            # BUY1
            #
                quantity = get_asset_quantity(self.amount, self._leverage, pair, TradeSide.BUY)
                # return [derivative_order(TradeSide('buy'), pair, pair.price(TradeSide.BUY), quantity, portfolio, leverage=self._leverage)]
                return [derivative_order(TradeSide.BUY, pair, pair.price(TradeSide.BUY), quantity, portfolio, leverage=self._leverage)]

            case 1:
            # SELL1
                quantity = get_asset_quantity(self.amount, self._leverage, pair, TradeSide.BUY)
                return [derivative_order(TradeSide.SELL, pair, pair.price(TradeSide.SELL), quantity, portfolio, leverage=self._leverage)]

            case 2:
                # BUY2: buy amount*2 

                quantity = get_asset_quantity(self.amount*2, self._leverage, pair, TradeSide.BUY)
                return [derivative_order(TradeSide.BUY, pair, pair.price(TradeSide.BUY), quantity, portfolio, leverage=self._leverage)]

            case 3:
                # SELL2: sell amount*2
                quantity = get_asset_quantity(self.amount*2, self._leverage, pair, TradeSide.BUY)
                return [derivative_order(TradeSide.SELL, pair, pair.price(TradeSide.SELL), quantity, portfolio, leverage=self._leverage)]



                # 5 sell limit
                # 6 buy market
                # 7 sell market
                # 8 buy stop
                # 9 sell stop

            case 4:
                # BUY_LIMIT
                # check that we doesn't have unexecuted buy limit order, if so return []
                b = self.broker
                a_limits = self.active_limits

                limit_price = Decimal('0.005')


                # print('a_limits ', a_limits)
                if contracts_num > 0:
                    print(f'contracts_num {contracts_num}')
                    if contract.side == TradeSide.BUY:
                        # print('do nothing we already stay LONG')
                        return []
                elif a_limits:
                    print(f'len(a_limits) {len(a_limits)}')
                    if len(a_limits) > 1:
                        raise Exception(f'len(self.active_limits) > 1: {len(a_limits)}')
                    if a_limits[0].is_buy:
                        return []
                else:
                    print('FREE')
                    cp = pair.price(TradeSide.BUY)
                    cpsell = pair.price(TradeSide.SELL)
                    # here we also have to check do we have limit order here?
                    if limit_price > cp:
                        # move to execute_derivative_order
                        print(f'invalid price.. lim: {limit_price} cp: {cp}')
                        return []
                    else:
                        print('ok.. derivative order')
                        quantity = get_asset_quantity(self.amount, self._leverage, pair, TradeSide.BUY)
                        return [derivative_order(TradeSide.BUY, pair, limit_price, quantity, portfolio, leverage=self._leverage, trade_type=TradeType.LIMIT)]



            case 5:
            # BUY_STOP
            #
                raise NotImplementedError('sorry..')
                return []

            case 6:
            # SELL_LIMIT
            #
                b = self.broker
                a_limits = self.active_limits
                limit_price = Decimal('0.014')
                print(' case 6')


                # print('a_limits ', a_limits)
                if contracts_num > 0:
                    print(f'contracts_num {contracts_num} ! {contract.side}')
                    if contract.side == TradeSide.SELL:
                        print(f'contract.PnL: {contract.value} cp: {contract.current_price}')
                        return []
                elif a_limits:
                    print(f'len(a_limits) {len(a_limits)}')
                    if len(a_limits) > 1:
                        raise Exception(f'len(self.active_limits) > 1: {len(a_limits)}')
                    if a_limits[0].is_sell:
                        return []
                else:
                    print('FREE')
                    cp = pair.price(TradeSide.BUY)
                    cpsell = pair.price(TradeSide.SELL)
                    # here we also have to check do we have limit order here?
                    if limit_price < cp:
                        # move to execute_derivative_order
                        print(f'invalid price.. lim: {limit_price} cp: {cp}')
                        return []
                    else:
                        print('ok.. derivative order')
                        quantity = get_asset_quantity(self.amount, self._leverage, pair, TradeSide.SELL)
                        return [derivative_order(TradeSide.SELL, pair, limit_price, quantity, portfolio, leverage=self._leverage, trade_type=TradeType.LIMIT)]

            case 7:
            # SELL_STOP
                raise NotImplementedError('sorry..')
                return []

            case 8:
                raise NotImplementedError('sorry..')
                return []

            case 9:
                # BUY MARKET w TP & SL
                # return order with takeprofit
                if contracts_num > 0:
                    print(f'contracts_num {contracts_num}')
                    if contract.side == TradeSide.BUY:
                        # print('do nothing we already stay LONG')
                        return []
                else:
                    quantity = get_asset_quantity(self.amount, self._leverage, pair, TradeSide.BUY)
                    price = pair.price(TradeSide.BUY)
                    # return [derivative_order(TradeSide.BUY, pair, price, quantity, portfolio, leverage=self._leverage, trade_type=TradeType.MARKET, stop_loss=StopLossPercent(percent=15)(price, TradeSide.BUY), take_profit=TakeProfitPercent(percent=55)(price, TradeSide.BUY))]
                    return [derivative_order(TradeSide.BUY, pair, price, quantity, portfolio, leverage=self._leverage,
                                         trade_type = TradeType.MARKET,
                                         stop_loss = self.stop_loss_policy(price, TradeSide.BUY),
                                         take_profit = self.take_profit_policy(price, TradeSide.BUY))]

            case 10:
                # CLOSE
                if contracts_num > 0:
                    if portfolio.contracts[0].side == TradeSide.BUY:
                        return [derivative_order(TradeSide.SELL, pair, pair.price(TradeSide.SELL), portfolio.contracts[0].quantity, portfolio, leverage=self._leverage)]

                    if portfolio.contracts[0].side == TradeSide.SELL:
                        return [derivative_order(TradeSide.BUY, pair, pair.price(TradeSide.SELL), portfolio.contracts[0].quantity, portfolio, leverage=self._leverage)]
                else:
                    return []

            case _:
                raise Exception(f'Unknown action: {action}')

        for listener in self.listeners:
            listener.on_action(action)

        return [order]

    def force_sell(self):
        # *forced by episode ending
        action = 1
        orders = self.get_orders(action, self.portfolio)

        for order in orders:
            if order:
                logging.info('Step {}: {} {}'.format(order.step, order.side, order.quantity))
                self.broker.submit(order)

        self.broker.update()

    def reset(self):
        super().reset()
        self.action = 1
        self.started = False

class SimpleOrders(TensorTradeActionScheme):
    """A discrete action scheme that determines actions based on a list of
    trading pairs, order criteria, and trade sizes.

    Parameters
    ----------
    criteria : List[OrderCriteria]
        A list of order criteria to select from when submitting an order.
        (e.g. MarketOrder, LimitOrder w/ price, StopLoss, etc.)
    trade_sizes : List[float]
        A list of trade sizes to select from when submitting an order.
        (e.g. '[1, 1/3]' = 100% or 33% of balance is tradable.
        '4' = 25%, 50%, 75%, or 100% of balance is tradable.)
    durations : List[int]
        A list of durations to select from when submitting an order.
    trade_type : TradeType
        A type of trade to make.
    order_listener : OrderListener
        A callback class to use for listening to steps of the order process.
    min_order_pct : float
        The minimum value when placing an order, calculated in percent over net_worth.
    min_order_abs : float
        The minimum value when placing an order, calculated in absolute order value.
    """

    def __init__(self,
                 criteria: 'Union[List[OrderCriteria], OrderCriteria]' = None,
                 trade_sizes: 'Union[List[float], int]' = 10,
                 durations: 'Union[List[int], int]' = None,
                 trade_type: 'TradeType' = TradeType.MARKET,
                 order_listener: 'OrderListener' = None,
                 min_order_pct: float = 0.02,
                 min_order_abs: float = 0.00) -> None:
        super().__init__()
        self.min_order_pct = min_order_pct
        self.min_order_abs = min_order_abs
        criteria = self.default('criteria', criteria)
        self.criteria = criteria if isinstance(criteria, list) else [criteria]

        trade_sizes = self.default('trade_sizes', trade_sizes)
        if isinstance(trade_sizes, list):
            self.trade_sizes = trade_sizes
        else:
            self.trade_sizes = [(x + 1) / trade_sizes for x in range(trade_sizes)]

        durations = self.default('durations', durations)
        self.durations = durations if isinstance(durations, list) else [durations]

        self._trade_type = self.default('trade_type', trade_type)
        self._order_listener = self.default('order_listener', order_listener)

        self._action_space = None
        self.actions = None

    @property
    def action_space(self) -> Space:
        if not self._action_space:
            self.actions = product(
                self.criteria,
                self.trade_sizes,
                self.durations,
                [TradeSide.BUY, TradeSide.SELL]
            )
            self.actions = list(self.actions)
            self.actions = list(product(self.portfolio.exchange_pairs, self.actions))
            self.actions = [None] + self.actions

            self._action_space = Discrete(len(self.actions))
        return self._action_space

    def get_orders(self,
                   action: int,
                   portfolio: 'Portfolio') -> 'List[Order]':

        if action == 0:
            return []

        (ep, (criteria, proportion, duration, side)) = self.actions[action]

        instrument = side.instrument(ep.pair)
        wallet = portfolio.get_wallet(ep.exchange.id, instrument=instrument)

        balance = wallet.balance.as_float()
        size = (balance * proportion)
        size = min(balance, size)

        quantity = (size * instrument).quantize()

        if size < 10 ** -instrument.precision \
                or size < self.min_order_pct * portfolio.net_worth \
                or size < self.min_order_abs:
            return []

        order = Order(
            step=self.clock.step,
            side=side,
            trade_type=self._trade_type,
            exchange_pair=ep,
            price=ep.price,
            quantity=quantity,
            criteria=criteria,
            end=self.clock.step + duration if duration else None,
            portfolio=portfolio
        )

        if self._order_listener is not None:
            order.attach(self._order_listener)

        return [order]


class ManagedRiskOrders(TensorTradeActionScheme):
    """A discrete action scheme that determines actions based on managing risk,
       through setting a follow-up stop loss and take profit on every order.

    Parameters
    ----------
    stop : List[float]
        A list of possible stop loss percentages for each order.
    take : List[float]
        A list of possible take profit percentages for each order.
    trade_sizes : List[float]
        A list of trade sizes to select from when submitting an order.
        (e.g. '[1, 1/3]' = 100% or 33% of balance is tradable.
        '4' = 25%, 50%, 75%, or 100% of balance is tradable.)
    durations : List[int]
        A list of durations to select from when submitting an order.
    trade_type : `TradeType`
        A type of trade to make.
    order_listener : OrderListener
        A callback class to use for listening to steps of the order process.
    min_order_pct : float
        The minimum value when placing an order, calculated in percent over net_worth.
    min_order_abs : float
        The minimum value when placing an order, calculated in absolute order value.
    """

    def __init__(self,
                 stop: 'List[float]' = [0.02, 0.04, 0.06],
                 take: 'List[float]' = [0.01, 0.02, 0.03],
                 trade_sizes: 'Union[List[float], int]' = 10,
                 durations: 'Union[List[int], int]' = None,
                 trade_type: 'TradeType' = TradeType.MARKET,
                 order_listener: 'OrderListener' = None,
                 min_order_pct: float = 0.02,
                 min_order_abs: float = 0.00) -> None:
        super().__init__()
        self.min_order_pct = min_order_pct
        self.min_order_abs = min_order_abs
        self.stop = self.default('stop', stop)
        self.take = self.default('take', take)

        trade_sizes = self.default('trade_sizes', trade_sizes)
        if isinstance(trade_sizes, list):
            self.trade_sizes = trade_sizes
        else:
            self.trade_sizes = [(x + 1) / trade_sizes for x in range(trade_sizes)]

        durations = self.default('durations', durations)
        self.durations = durations if isinstance(durations, list) else [durations]

        self._trade_type = self.default('trade_type', trade_type)
        self._order_listener = self.default('order_listener', order_listener)

        self._action_space = None
        self.actions = None

    @property
    def action_space(self) -> 'Space':
        if not self._action_space:
            self.actions = product(
                self.stop,
                self.take,
                self.trade_sizes,
                self.durations,
                [TradeSide.BUY, TradeSide.SELL]
            )
            self.actions = list(self.actions)
            self.actions = list(product(self.portfolio.exchange_pairs, self.actions))
            self.actions = [None] + self.actions

            self._action_space = Discrete(len(self.actions))
        return self._action_space

    def get_orders(self, action: int, portfolio: 'Portfolio') -> 'List[Order]':

        if action == 0:
            return []

        (exchange_pair, (stop, take, proportion, duration, side)) = self.actions[action]
        print(exchange_pair)
        print(type(exchange_pair))
        print('ep, (stop, take, proportion, duration, side) :: ', exchange_pair, stop, take, proportion, duration, side)
        # ep, (stop, take, proportion, duration, side) ::  my_exchange:USDT/ASSET 0.02 0.01 0.1 None sell
        side = TradeSide(side)

        instrument = side.instrument(exchange_pair.pair)
        wallet = portfolio.get_wallet(exchange_pair.exchange.id, instrument=instrument)

        balance = wallet.balance.as_float()
        size = (balance * proportion)
        size = min(balance, size)
        quantity = (size * instrument).quantize()

        if size < 10 ** -instrument.precision \
                or size < self.min_order_pct * portfolio.net_worth \
                or size < self.min_order_abs:
            return []

        params = {
            'side': side,
            'exchange_pair': exchange_pair,
            'price': exchange_pair.price,
            'quantity': quantity,
            'down_percent': stop,
            'up_percent': take,
            'portfolio': portfolio,
            'trade_type': self._trade_type,
            'end': self.clock.step + duration if duration else None
        }

        order = risk_managed_order(**params)

        if self._order_listener is not None:
            order.attach(self._order_listener)

        return [order]



class MSBSCMT(MultySymbolBSH):
    """Multy-Symbol-Buy-Sell-Close-Margin-Trade
        action scheme for multiple symbols environments with margin trade support

        3 actions:

            buy: long position (0)
                close position if u have short
                and then open long

            sell: short position (1)
                close position if u have long
                and then open short

            close: close position (2)
                close position if u have any
                and stay away from market

        * scheme allows to hold not more then one contract!!

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base intrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    """

    registered_name = "msbscmt"

    # def __init__(self, cash: 'Wallet', asset: 'Wallet'):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._leverage = get_param(config['params'],'leverage')['value']
        self.listeners = []
        self.action = 2
        self.started = False
        self.amount = config['amount']

    @property
    def action_space(self):
        return Discrete(3)

    def attach(self, listener):
        self.listeners += [listener]
        return self

    @property
    def leverage(self):
        """Getter for price"""
        return self._leverage

    @leverage.setter
    def leverage(self, value: int):
        """Setter for price"""
        if value < 0:
            raise ValueError("Leverage cannot be negative")
        self._leverage = value

    def get_orders(self, action: int, portfolio: 'Portfolio') -> 'Order':
        order = None
        current_symbol_code = self.config["current_symbol_code"]
        cash = portfolio.wallets[0]
        asset = portfolio.wallets[current_symbol_code+1]
        contracts_num = len(portfolio.contracts)
        self.action = action

        order = None
        current_symbol_code = self.config["current_symbol_code"]
        cash = portfolio.wallets[0]
        asset = portfolio.wallets[current_symbol_code+1]
        pair = portfolio.exchange_pairs[current_symbol_code]

        if 5 > 3:
            if contracts_num > 1:
                print('So much contracts')
                raise Exception(f'contracts_num > 1: {contracts_num}')

            if contracts_num > 0:
                contract = portfolio.contracts[0]

            if action == 0:
                if contracts_num > 0:
                    if contract.side == TradeSide.BUY:
                        # print('do nothing we already stay LONG')
                        return []

                    if contract.side == TradeSide.SELL:
                        print('close LONG open SHORT')
                        quantity = get_asset_quantity(self.amount*2, self._leverage, pair, TradeSide.BUY)
                        # return [derivative_order(TradeSide('buy'), pair, pair.price(TradeSide.BUY), self.amount*2, portfolio, leverage=self._leverage)]
                        return [derivative_order(TradeSide('buy'), pair, pair.price(TradeSide.BUY), quantity, portfolio, leverage=self._leverage)]
                else:
                    quantity = get_asset_quantity(self.amount, self._leverage, pair, TradeSide.BUY)
                    # return [derivative_order(TradeSide('buy'), pair, pair.price(TradeSide.BUY), self.amount, portfolio, leverage=self._leverage)]
                    return [derivative_order(TradeSide('buy'), pair, pair.price(TradeSide.BUY), quantity, portfolio, leverage=self._leverage)]

            if action == 1:
                if contracts_num > 0:
                    if portfolio.contracts[0].side == TradeSide.SELL:
                        # print('do nothing we already stay SHORT')
                        return []

                    if portfolio.contracts[0].side == TradeSide.BUY:
                        print('close SHORT open LONG')
                        # then go to broker where all of this is processin
                        quantity = get_asset_quantity(self.amount*2, self._leverage, pair, TradeSide.SELL)
                        return [derivative_order(TradeSide.SELL, pair, pair.price(TradeSide.SELL), quantity, portfolio, leverage=self._leverage)]
                else:
                    quantity = get_asset_quantity(self.amount, self._leverage, pair, TradeSide.SELL)
                    return [derivative_order(TradeSide.SELL, pair, pair.price(TradeSide.SELL), quantity, portfolio, leverage=self._leverage)]

            if action == 2:
                if contracts_num > 0:
                    if portfolio.contracts[0].side == TradeSide.BUY:
                        # return [derivative_order(TradeSide.SELL, pair, pair.price(TradeSide.SELL), portfolio.contracts[0].order.quantity.size, portfolio, leverage=self._leverage)]
                        return [derivative_order(TradeSide.SELL, pair, pair.price(TradeSide.SELL), contract.quantity, portfolio, leverage=self._leverage)]

                    if portfolio.contracts[0].side == TradeSide.SELL:
                        # return [derivative_order(TradeSide.BUY, pair, pair.price(TradeSide.SELL), portfolio.contracts[0].order.quantity.size, portfolio, leverage=self._leverage)]
                        return [derivative_order(TradeSide.BUY, pair, pair.price(TradeSide.SELL), contract.quantity, portfolio, leverage=self._leverage)]

                else:
                    return []

        for listener in self.listeners:
            listener.on_action(action)

        return [order]

    def force_sell(self):
        # *forced by episode ending
        action = 1
        orders = self.get_orders(action, self.portfolio)

        for order in orders:
            if order:
                logging.info('Step {}: {} {}'.format(order.step, order.side, order.quantity))
                self.broker.submit(order)

        self.broker.update()

    def reset(self):
        super().reset()
        self.action = 1
        self.started = False


def get_asset_quantity(margin, leverage, pair, side):
    # quantity = margin * leverage * pair.pair.quote
    raw_qty = margin * leverage / pair.price(side)
    step = Decimal(str(pair.step_size))
    qty = (Decimal(str(raw_qty)) // step) * step * pair.pair.quote
    return qty.quantize()


_registry = {
    'bsh': BSH,
    'simple': SimpleOrders,
    'managed-risk': ManagedRiskOrders,
}

def get(identifier: str) -> 'ActionScheme':
    """Gets the `ActionScheme` that matches with the identifier.

    Parameters
    ----------
    identifier : str
        The identifier for the `ActionScheme`.

    Returns
    -------
    'ActionScheme'
        The action scheme associated with the `identifier`.

    Raises
    ------
    KeyError:
        Raised if the `identifier` is not associated with any `ActionScheme`.
    """
    if identifier not in _registry.keys():
        raise KeyError(f"Identifier {identifier} is not associated with any `ActionScheme`.")
    return _registry[identifier]()
