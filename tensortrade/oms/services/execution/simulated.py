import decimal
import logging
from decimal import Decimal

from tensortrade.core import Clock
from tensortrade.oms.wallets import Wallet
from tensortrade.oms.exchanges import ExchangeOptions
from tensortrade.oms.orders import Order, Trade, TradeType, TradeSide

from typing import Union

# from tensortrade.oms.wallets.contract import Contract

def get_perp_commission():
    return 0

def execute_derivative_order(order: 'Order',
                      base_wallet: 'Wallet',
                      quote_wallet: 'Wallet',
                      current_price: float,
                      options: 'ExchangeOptions',
                      clock: 'Clock') -> Union[None, 'Trade']:
    """Executes a buy order on the exchange.

    Parameters
    ----------
    order : `Order`
        The order that is being filled.
    base_wallet : `Wallet`
        The wallet of the base instrument.
    quote_wallet : `Wallet`
        The wallet of the quote instrument.
    current_price : float
        The current price of the exchange pair.
    options : `ExchangeOptions`
        The exchange options.
    clock : `Clock`
        The clock for the trading process..

    Returns
    -------
    `Trade`
        The executed trade that was made.
    """

    if order.type == TradeType.LIMIT and order.price < current_price:
        return None

    # filled = order.remaining.contain(order.exchange_pair, order.side)
    #
    # if order.type == TradeType.MARKET:
    #     scale = order.price / max(current_price, order.price)
    #     filled = scale * filled

    commission = get_perp_commission() #options.commission * filled

    trade = Trade(
        order_id=order.id,
        step=clock.step,
        exchange_pair=order.exchange_pair,
        side=order.side, #TradeSide.BUY,
        trade_type=order.type,
        # quantity=transfer.quantity,
        quantity=order.quantity,
        price=order.price,
        commission=commission
    )

    return trade





def execute_buy_order(order: 'Order',
                      base_wallet: 'Wallet',
                      quote_wallet: 'Wallet',
                      current_price: float,
                      options: 'ExchangeOptions',
                      clock: 'Clock') -> Union[None, 'Trade']:
    """Executes a buy order on the exchange.

    Parameters
    ----------
    order : `Order`
        The order that is being filled.
    base_wallet : `Wallet`
        The wallet of the base instrument.
    quote_wallet : `Wallet`
        The wallet of the quote instrument.
    current_price : float
        The current price of the exchange pair.
    options : `ExchangeOptions`
        The exchange options.
    clock : `Clock`
        The clock for the trading process..

    Returns
    -------
    `Trade`
        The executed trade that was made.
    """
    if order.type == TradeType.LIMIT and order.price < current_price:
        return None

    filled = order.remaining.contain(order.exchange_pair, TradeSide.BUY)

    if order.type == TradeType.MARKET:
        scale = order.price / max(current_price, order.price)
        filled = scale * filled

    commission = options.commission * filled

    # If the user has specified a non-zero commission percentage, it has to be higher
    # than the instrument precision, otherwise the minimum precision value is used.
    minimum_commission = Decimal(10) ** -filled.instrument.precision
    if options.commission > 0 and commission < minimum_commission:
        logging.warning("Commission is > 0 but less than instrument precision. "
                        "Setting commission to the minimum allowed amount. "
                        "Consider defining a custom instrument with a higher precision.")
        commission.size = minimum_commission

    quantity = filled - commission

    transfer = Wallet.transfer(
        source=base_wallet,
        target=quote_wallet,
        order=order,
        quantity=quantity,
        commission=commission,
        exchange_pair=order.exchange_pair,
        reason="BUY"
    )

    trade = Trade(
        order_id=order.id,
        step=clock.step,
        exchange_pair=order.exchange_pair,
        side=TradeSide.BUY,
        trade_type=order.type,
        quantity=transfer.quantity,
        price=order.price,
        commission=transfer.commission
    )

    return trade


def execute_sell_order(order: 'Order',
                       base_wallet: 'Wallet',
                       quote_wallet: 'Wallet',
                       current_price: float,
                       options: 'ExchangeOptions',
                       clock: 'Clock') -> Union[None, 'Trade']:
    """Executes a sell order on the exchange.

    Parameters
    ----------
    order : `Order`
        The order that is being filled.
    base_wallet : `Wallet`
        The wallet of the base instrument.
    quote_wallet : `Wallet`
        The wallet of the quote instrument.
    current_price : float
        The current price of the exchange pair.
    options : `ExchangeOptions`
        The exchange options.
    clock : `Clock`
        The clock for the trading process..

    Returns
    -------
    `Trade`
        The executed trade that was made.
    """
    if order.type == TradeType.LIMIT and order.price > current_price:
        return None

    # filled = order.remaining.contain(order.exchange_pair)
    filled = order.remaining.contain(order.exchange_pair, TradeSide.SELL)

    commission = options.commission * filled

    # If the user has specified a non-zero commission percentage, it has to be higher
    # than the instrument precision, otherwise the minimum precision value is used.
    minimum_commission = Decimal(10) ** -filled.instrument.precision
    if options.commission > 0 and commission < minimum_commission:
        logging.warning("Commission is > 0 but less than instrument precision. "
                        "Setting commission to the minimum allowed amount. "
                        "Consider defining a custom instrument with a higher precision.")
        commission.size = minimum_commission

    quantity = filled - commission

    # Transfer Funds from Quote Wallet to Base Wallet
    transfer = Wallet.transfer(
        source=quote_wallet,
        target=base_wallet,
        order=order,
        quantity=quantity,
        commission=commission,
        exchange_pair=order.exchange_pair,
        reason="SELL"
    )

    trade = Trade(
        order_id=order.id,
        step=clock.step,
        exchange_pair=order.exchange_pair,
        side=TradeSide.SELL,
        trade_type=order.type,
        quantity=transfer.quantity,
        price=order.price,
        commission=transfer.commission
    )

    return trade


def execute_order(order: 'Order',
                  base_wallet: 'Wallet',
                  quote_wallet: 'Wallet',
                  current_price: float,
                  options: 'Options',
                  clock: 'Clock') -> 'Trade':
    """Executes an order on the exchange.

    Parameters
    ----------
    order : `Order`
        The order that is being filled.
    base_wallet : `Wallet`
        The wallet of the base instrument.
    quote_wallet : `Wallet`
        The wallet of the quote instrument.
    current_price : float
        The current price of the exchange pair.
    options : `ExchangeOptions`
        The exchange options.
    clock : `Clock`
        The clock for the trading process..

    Returns
    -------
    `Trade`
        The executed trade that was made.
    """
    kwargs = {"order": order,
              "base_wallet": base_wallet,
              "quote_wallet": quote_wallet,
              "current_price": current_price,
              "options": options,
              "clock": clock}

    o = order
    if options.leverage is not None: # by default is 1
        trade = execute_derivative_order(**kwargs)
    elif order.is_buy:
        trade = execute_buy_order(**kwargs)
    elif order.is_sell:
        trade = execute_sell_order(**kwargs)
    else:
        trade = None

    return trade
