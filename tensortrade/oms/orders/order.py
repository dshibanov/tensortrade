# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


import uuid

from enum import Enum
from typing import Callable
from decimal import Decimal

from tensortrade.core import TimedIdentifiable, Observable
from tensortrade.core.exceptions import InvalidOrderQuantity
from tensortrade.oms.instruments import Quantity, ExchangePair
from tensortrade.oms.orders import Trade, TradeSide, TradeType


from tensortrade.core.exceptions import (
    InsufficientFunds,
    DoubleLockedQuantity,
    DoubleUnlockedQuantity,
    QuantityNotLocked
)

import copy






class OrderStatus(Enum):
    """An enumeration for the status of an order."""

    PENDING = "pending"
    OPEN = "open"
    CANCELLED = "cancelled"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"

    def __str__(self):
        return self.value


def average_entry_price(orders):
    total_cost = sum(Decimal(str(price)) * Decimal(str(margin)) for price, margin in orders)
    total_qty = sum(Decimal(str(margin)) for _, margin in orders)
    return total_cost / total_qty

class Order(TimedIdentifiable, Observable):
    """A class to represent ordering an amount of a financial instrument.

    Responsibilities of the Order:
        1. Confirming its own validity.
        2. Tracking its trades and reporting it back to the broker.
        3. Managing movement of quantities from order to order.
        4. Generating the next order in its path given that there is a
           'OrderSpec' for how to make the next order.
        5. Managing its own state changes when it can.

    Parameters
    ----------
     side : `TradeSide`
        The side of the order.
    exchange_pair : `ExchangePair`
        The exchange pair to perform the order for.
    price : float
        The price of the order.
    trade_type : `TradeType`
        The type of trade being made.
    exchange_pair : `ExchangePair`
        The exchange pair that the order is made for.
    quantity : `Quantity`
        The quantity of the order.
    portfolio : `Portfolio`
        The portfolio being used in the order.
    criteria : `Callable[[Order, Exchange], bool]`, optional
        The criteria under which the order will be considered executable.
    path_id : str, optional
        The path order id.
    start : int, optional
        The start time of the order.
    end : int, optional
        The end time of the order.

    Raises
    ------
    InvalidOrderQuantity
        Raised if the given quantity has a size of 0.
    """

    def __init__(self,
                 step: int,
                 side: TradeSide,
                 trade_type: TradeType,
                 exchange_pair: 'ExchangePair',
                 quantity: 'Quantity',
                 portfolio: 'Portfolio',
                 price: float,
                 leverage: int = None,
                 criteria: 'Callable[[Order, Exchange], bool]' = None,
                 path_id: str = None,
                 start: int = None,
                 end: int = None,
                 stop_loss: float = None,
                 take_profit: float = None,
                 derivative: bool = False,
                 liquidation = False,
                 broker = None):
        super().__init__()
        Observable.__init__(self)

        quantity = quantity.contain(exchange_pair, side)
        if quantity.size == 0:
            raise InvalidOrderQuantity(quantity)

        self.step = step
        self.side = side
        self.type = trade_type
        self.exchange_pair = exchange_pair
        self.leverage = leverage #self.exchange_pair.exchange.options.leverage
        self.portfolio = portfolio
        self.price = price
        self.criteria = criteria
        self.path_id = path_id or str(uuid.uuid4())
        self.quantity = quantity
        self.start = start or step
        self.end = end
        self.status = OrderStatus.PENDING
        self.derivative = derivative
        self._specs = []
        self._linked = []
        self.trades = []
        self._one_cancel_other = []
        self.broker = broker
        self.liquidation = liquidation


        if self.derivative:
            self.instrument = self.portfolio.base_instrument
            self.quantity = quantity.quantize()
            self.margin = ((self.quantity.size/self.leverage)*self.price*self.portfolio.base_instrument).quantize()
        else:
            self.instrument = self.side.instrument(self.exchange_pair.pair)

        wallet = portfolio.get_wallet(
            self.exchange_pair.exchange.id,
            self.instrument
        )

        if self.path_id not in wallet.locked.keys():
            if self.derivative:
                pass
            else:
                self.quantity = wallet.lock(quantity, self, "LOCK FOR ORDER")

        # here? maybe we need to call this in case if leverage is None??
        if self.derivative:
            self.remaining = self.quantity
        else:
            self.remaining = self.quantity

    @property
    def size(self) -> 'Decimal':
        """The size of the order. (`Decimal`, read-only)"""
        if not self.quantity or self.quantity is None:
            return Decimal(-1)
        return self.quantity.size

    @property
    def current_price(self):
        return self.exchange_pair.exchange.quote_price(self.exchange_pair.pair)

    @property
    def pair(self) -> 'TradingPair':
        """The trading pair of the order. (`TradingPair`, read-only)"""
        return self.exchange_pair.pair

    @property
    def base_instrument(self) -> 'Instrument':
        """The base instrument of the pair being traded."""
        return self.exchange_pair.pair.base

    @property
    def quote_instrument(self) -> 'Instrument':
        """The quote instrument of the pair being traded."""
        return self.exchange_pair.pair.quote

    @property
    def is_buy(self) -> bool:
        """If this is a buy order. (bool, read-only)"""
        return self.side == TradeSide.BUY

    @property
    def is_sell(self) -> bool:
        """If this is a sell order. (bool, read-only)"""
        return self.side == TradeSide.SELL

    @property
    def is_limit_order(self) -> bool:
        """If this is a limit order. (bool, read-only)"""
        return self.type == TradeType.LIMIT

    @property
    def is_market_order(self) -> bool:
        """If this is a market order. (bool, read-only)"""
        return self.type == TradeType.MARKET

    @property
    def is_executable(self) -> bool:
        """If this order is executable. (bool, read-only)"""
        is_satisfied = self.criteria is None or self.criteria(self, self.exchange_pair.exchange)
        clock = self.exchange_pair.exchange.clock
        return is_satisfied and clock.step >= self.start

    @property
    def is_expired(self) -> bool:
        """If this order is expired. (bool, read-only)"""
        if self.end:
            return self.exchange_pair.exchange.clock.step >= self.end
        return False

    @property
    def is_cancelled(self) -> bool:
        """If this order is cancelled. (bool, read-only)"""
        return self.status == OrderStatus.CANCELLED

    @property
    def is_active(self) -> bool:
        """If this order is active. (bool, read-only)"""
        return self.status != OrderStatus.FILLED and self.status != OrderStatus.CANCELLED

    @property
    def is_complete(self) -> bool:
        """If this order is complete. (bool, read-only)"""
        if self.status == OrderStatus.CANCELLED:
            return True

        wallet = self.portfolio.get_wallet(
            self.exchange_pair.exchange.id,
            self.side.instrument(self.exchange_pair.pair)
        )
        quantity = wallet.locked.get(self.path_id, None)

        return (quantity and quantity.size == 0) or self.remaining.size <= 0

    def add_order_spec(self, order_spec: 'OrderSpec') -> 'Order':
        """Adds an order specification to the order.

        Parameters
        ----------
        order_spec : `OrderSpec`
            An order specification.

        Returns
        -------
        `Order`
            The current order.
        """
        self._specs += [order_spec]
        return self

    def execute(self) -> None:
        """Executes the order."""
        self.status = OrderStatus.OPEN

        if self.portfolio.order_listener:
            self.attach(self.portfolio.order_listener)

        for listener in self.listeners or []:
            listener.on_execute(self)

        self.exchange_pair.exchange.execute_order(self, self.portfolio)

    def fill(self, trade: 'Trade') -> None:
        """Fills the order.

        Parameters
        ----------
        trade : `Trade`
            A trade to fill the order.
        """
        self.status = OrderStatus.PARTIALLY_FILLED

        if not self.derivative:
            filled = trade.quantity + trade.commission
            self.remaining -= filled
        else:
            wallet = self.portfolio.get_wallet(
                self.exchange_pair.exchange.id,
                self.instrument
            )

            if trade.is_liquidation:
                q = Quantity(self.instrument, wallet.balance.size) if wallet.balance.size < trade.commission.size else trade.commission
                wallet.withdraw(q, "LIQUIDATION FEES")
            else:
                wallet.withdraw(trade.commission, "TAKER FEES")

            filled = self.quantity
            self.remaining -= filled

        self.trades += [trade]


        for listener in self.listeners or []:
            listener.on_fill(self, trade)

    def complete(self) -> 'Order':
        """Completes an order.

        Returns
        -------
        `Order`
            The completed order.
        """
        self.status = OrderStatus.FILLED

        if self._specs:
            print('oh, we have specs..')
            while self._specs:
                order_spec = self._specs.pop()
                order = order_spec.create_order(self)
                self._linked.append(order)

            for i, o in enumerate(self._linked):
                o._one_cancel_other = self._linked[:i] + self._linked[i+1:]

        for listener in self.listeners or []:
            listener.on_complete(self)

        self.listeners = []
        return self._linked or self.release("COMPLETED")

    def cancel(self, reason: str = "CANCELLED") -> None:
        """Cancels an order.

        Parameters
        ----------
        reason : str, default 'CANCELLED'
            The reason for canceling the order.
        """
        self.status = OrderStatus.CANCELLED

        for listener in self.listeners or []:
            listener.on_cancel(self)

        self.listeners = []

        self.release(reason)

    def release(self, reason: str = "RELEASE (NO REASON)") -> None:
        """Releases all quantities from every wallet that have been allocated
        for this order.

        Parameters
        ----------
        reason : str, default 'RELEASE (NO REASON)'
            The reason for releasing all locked quantities associated with the
            order.
        """
        for wallet in self.portfolio.wallets:
            if self.path_id in wallet.locked.keys():
                quantity = wallet.locked[self.path_id]

                if not self.derivative:
                    if quantity is not None:
                        wallet.unlock(quantity, reason)

                    wallet.locked.pop(self.path_id, None)

    def to_dict(self) -> dict:
        """Creates a dictionary representation of the order.

        Returns
        -------
        dict
            The dictionary representation of the order.
        """
        return {
            "id": self.id,
            "step": self.step,
            "exchange_pair": str(self.exchange_pair),
            "status": self.status,
            "type": self.type,
            "side": self.side,
            "quantity": self.quantity,
            "size": self.size,
            "remaining": self.remaining,
            "price": self.price,
            "criteria": self.criteria,
            "path_id": self.path_id,
            "created_at": self.created_at
        }

    def to_json(self) -> dict:
        """Creates a json dictionary representation of the order.

        Returns
        -------
        dict
            The json dictionary representation of the order
        """
        return {
            "id": str(self.id),
            "step": int(self.step),
            "exchange_pair": str(self.exchange_pair),
            "status": str(self.status),
            "type": str(self.type),
            "side": str(self.side),
            "base_symbol": str(self.exchange_pair.pair.base.symbol),
            "quote_symbol": str(self.exchange_pair.pair.quote.symbol),
            "quantity": str(self.quantity),
            "size": float(self.size),
            "remaining": str(self.remaining),
            "price": float(self.price),
            "criteria": str(self.criteria),
            "path_id": str(self.path_id),
            "created_at": str(self.created_at)
        }

    def __str__(self) -> str:
        data = ['{}={}'.format(k, v) for k, v in self.to_dict().items()]
        return '<{}: {}>'.format(self.__class__.__name__, ', '.join(data))

    def __repr__(self) -> str:
        return str(self)
