
from decimal import Decimal


class ExchangePair:
    """A pair of financial instruments to be traded on a specific exchange.

    Parameters
    ----------
    exchange : `Exchange`
        An exchange that contains the `pair` for trading.
    pair : `TradingPair`
        A trading pair available on the `exchange`.
    """

    def __init__(self, exchange: "Exchange", pair: "TradingPair"):
        self.exchange = exchange
        self.pair = pair

    def price(self, side) -> "Decimal":
        """The quoted price of the trading pair. (`Decimal`, read-only)"""
        return self.exchange.quote_price(self.pair)

    def inverse_price(self, side) -> "Decimal":
        """The inverse price of the trading pair. (`Decimal, read-only)"""
        quantization = Decimal(10) ** -self.pair.quote.precision
        return Decimal(self.price(side) ** Decimal(-1)).quantize(quantization)

    @property
    def options(self):
        return self.exchange.options.config[f'{self.pair.quote}{self.pair.base}']

    @property
    def step_size(self):
        for f in self.options['info']['filters']:
            if f['filterType'] == 'LOT_SIZE':
                step_size = float(f['stepSize'])
        return step_size

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, ExchangePair):
            if str(self) == str(other):
                return True
        return False

    def __str__(self):
        return "{}:{}".format(self.exchange.name, self.pair)

    def __repr__(self):
        return str(self)
