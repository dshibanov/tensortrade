from datetime import datetime, timedelta
from enum import Enum
import pytest


# === Implementation ===

class ContractType(Enum):
    OPTION = "Option"
    FUTURE = "Future"
    SWAP = "Swap"
    FORWARD = "Forward"
    BINANCE_PERP = "Binance Perpetual"



class Contract:
    def __init__(
        self,
        symbol: str,
        contract_type: ContractType,
        underlying: str,
        expiration: datetime,
        strike: float = None,
        # option_type: OptionType = OptionType.NA,
        exchange: str = None,
        currency: str = "USD",
        side: str = None,
        leverage: int = 1,
        quantity: float = None,
        entry_price: float = None,
        **kwargs
    ):
        self.symbol = symbol
        self.contract_type = contract_type
        self.underlying = underlying
        self.expiration = expiration
        self.strike = strike
        # self.option_type = option_type
        self.exchange = exchange
        self.currency = currency
        self.side = side
        self.leverage = leverage
        self.quantity = quantity
        self.entry_price = entry_price


    def value(self, spot_price: float) -> float:
        if self.contract_type == ContractType.BINANCE_PERP:
            if self.side not in ("LONG", "SHORT"):
                raise ValueError("Side must be 'LONG' or 'SHORT'")
            if self.entry_price is None or self.quantity is None:
                raise ValueError("entry_price and quantity must be set for Binance Perpetual contract")
            if self.side == "LONG":
                return (spot_price - self.entry_price) * self.quantity * self.leverage
            else:  # SHORT
                return (self.entry_price - spot_price) * self.quantity * self.leverage

        raise NotImplementedError("Value calculation not implemented for this contract type.")

    def __repr__(self):
        return (f"<Contract {self.symbol} ({self.contract_type.value}) "
                f"{self.option_type.value if self.option_type != OptionType.NA else ''} "
                f"{self.strike if self.strike else ''} "
                f"exp: {self.expiration.strftime('%Y-%m-%d')} "
                f"underlying: {self.underlying}>")


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
