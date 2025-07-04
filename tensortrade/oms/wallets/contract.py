from datetime import datetime, timedelta
from enum import Enum
import pytest
from tensortrade.oms.orders import TradeSide
from decimal import Decimal
from tensortrade.oms.instruments import Quantity, ExchangePair
from tensortrade.oms.orders import OrderStatus
class ContractType(Enum):
    OPTION = "Option"
    FUTURE = "Future"
    SWAP = "Swap"
    FORWARD = "Forward"
    BINANCE_PERP = "Binance Perpetual"



class Contract:
    def __init__(
        self,
        order: "Order"
    ):
        self.order = order
        self.pnls = []
        self.m_balances = []
        self.m_margins = []

        self.is_liquidates = []

    @property
    def entry_price(self):
        return self.order.price

    @property
    def leverage(self):
        return self.order.leverage


    @property
    def side(self) -> float:
        return self.order.side

    @property
    def quantity(self):
        return self.order.quantity


    @property
    def margin(self):
        return self.order.margin

    @property
    def current_price(self):
        return self.order.current_price
    @property
    def value(self) -> float:
        # current_price = self.order.exchange_pair.exchange.quote_price(self.order.exchange_pair.pair)
        price_delta = (self.current_price - self.order.price) if self.side == TradeSide.BUY else (self.order.price - self.current_price)
        # pnl = Decimal(str(self.order.n_coins)) * price_delta
        pnl = Decimal(str(self.quantity.size)) * price_delta
        self.pnl_usdt = pnl / self.current_price
        self.pnls += [self.pnl_usdt]


        # if self.is_liquidate:
        #     print('^-.-^')
        # if self.is_liquidate:
        #     print('liquidate me')



        return float(self.pnl_usdt)


    @property
    def is_liquidate(self): #, contract_value, maint_margin_percent=0.005, margin_balance=0.0, liquidation_fee_percent=0.004):
        """
        Checks if a position is subject to liquidation.

        Parameters:
        - contract_value (float): Current notional value of the position (e.g., quantity × price).
        - maint_margin_percent (float): Maintenance margin percent (default 0.5%).
        - margin_balance (float): Current margin balance (wallet balance + unrealized PnL).
        - liquidation_fee_percent (float): Fee charged on liquidation (default 0.4%).

        Returns:
        - dict with maintenance_margin, liquidation_fee, and liquidation_status
        """


        notional = self.order.quantity.size * self.current_price
        margin_balance = self.wallet.balance.size + self.pnl_usdt

        maintenance_margin = notional * Decimal(self.order.exchange_pair.options['info']['maintMarginPercent'])*Decimal('0.01')
        liquidation_fee = self.pnl_usdt * Decimal(self.order.exchange_pair.options['info']['liquidationFee'])


        is_liquidated = margin_balance < maintenance_margin
        self.is_liquidates += [is_liquidated]
        self.m_balances += [margin_balance]
        self.m_margins += [maintenance_margin]

        return is_liquidated



    def liquidate(self):
        self.wallet.unlock(self.margin, "RELEASE MARGIN DUE CLOSE")

        for o in self.order._linked:
            o.status = OrderStatus.CANCELLED

        commission = Quantity(self.order.base_instrument,
                              self.order.margin.size * self.order.leverage * Decimal(
                                  str(self.order.exchange_pair.options['info']['liquidationFee'])))

        if self.wallet.balance.size < abs(Decimal(str(self.value))):
            self.wallet.withdraw(Quantity(self.order.instrument,self.wallet.balance.size), "PnL FROM CONTRACT CLOSE")
        else:
            print('self.wallet.balance.size < Decimal(str(self.value)): ', self.wallet.balance.size, Decimal(str(self.value)))
            self.wallet.withdraw(Quantity(self.order.instrument, -self.value), "PnL FROM CONTRACT CLOSE")
            self.wallet.withdraw(commission, "LIQUIDATION FEE")

        if len(self.order.portfolio.contracts) > 1:
            raise Exception(f' len(self.order.portfolio.contracts) > 1 :{len(self.order.portfolio.contracts)}')

        # self.order.portfolio.contracts.pop()
        # print(f'self.order.portfolio.contracts {len(self.order.portfolio.contracts)}')

    @property
    def wallet(self):
        return self.order.portfolio.get_wallet(
            self.order.exchange_pair.exchange.id,
            self.order.instrument
        )

    def close(self, liquidation = False):
        self.wallet.unlock(self.margin, "RELEASE MARGIN DUE CLOSE")

        for o in self.order._linked:
            o.status = OrderStatus.CANCELLED
            # o.cancel(f'CANCELLED DUE BEING LINKED WITH ORDER {self.order}')

        if self.value > 0:
            self.wallet.deposit(Quantity(self.order.instrument, self.value), "PnL FROM CONTRACT CLOSE")
        else:
            message = "PnL FROM CONTRACT LIQUIDATION" if liquidation else "PnL FROM CONTRACT CLOSE"
            vlue = Decimal(str(abs(self.value)))
            if self.wallet.balance.size <= vlue:
                self.wallet.withdraw(Quantity(self.order.instrument, self.wallet.balance.size), message)
            else:
                self.wallet.withdraw(Quantity(self.order.instrument, -self.value), message)




    def reduce(self, quantity, order):
        """Reduce contract quantity"""
        # raise NotImplementedError('contract.reduce: sorry..')
        # reduce contract locked margin
        # m = copy.deepcopy(self.margin)
        # m.size = quantity.size*self.current_price/self.leverage

        q_before = self.order.quantity.size
        self.order.quantity -= quantity
        q_after = self.order.quantity.size

        before = self.margin.size
        self.wallet.unlock(self.margin, "RELEASE MARGIN DUE REDUCE")
        self.order.margin = ((self.quantity.size/self.leverage)*self.entry_price*self.order.portfolio.base_instrument).quantize()
        # self.order.margin.size = ((self.quantity.size / self.leverage) * self.entry_price * self.order.portfolio.base_instrument).quantize().size
        after = self.margin.size

        r1 = q_after/ q_before
        r2 = after / before
        self.order.margin = self.wallet.lock(self.margin, order, "RE-LOCK MARGIN DUE REDUCE")

        price_delta = (self.current_price - self.order.price) if self.side == TradeSide.BUY else (
                    self.order.price - self.current_price)
        pnl = Decimal(str(quantity.size)) * price_delta
        pnl_usdt = float(pnl / self.current_price)
        # add profit to balance
        self.wallet.deposit(Quantity(self.order.instrument, pnl_usdt), "PROFIT FROM CONTRACT REDUCE")

    def __repr__(self):
        return (f"<Contract {self.order.exchange_pair} side: {self.side} entry: {self.entry_price} value: {self.value} ")


# === Tests ===
@pytest.fixture
def binance_perp_long():
    return Contract(
        symbol="BTCUSDT",
        contract_type=ContractType.BINANCE_PERP,
        underlying="BTC",
        expiration=datetime.max,
        side="LONG",
        leverage=10,
        quantity=0.1,
        entry_price=30000.0
    )

@pytest.fixture
def binance_perp_short():
    return Contract(
        symbol="BTCUSDT",
        contract_type=ContractType.BINANCE_PERP,
        underlying="BTC",
        expiration=datetime.max,
        side="SHORT",
        leverage=10,
        quantity=0.1,
        entry_price=30000.0
    )


def test_binance_perp_long(binance_perp_long):
    assert binance_perp_long.value(31000.0) == 1000.0

def test_binance_perp_short(binance_perp_short):
    assert binance_perp_short.value(29000.0) == 1000.0
