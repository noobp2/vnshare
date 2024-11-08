from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from vnpy.trader.constant import Exchange


class FREQ(Enum):
    DAILY = '1d'
    WEEKLY = '1w'
    MONTHLY = 'month'
    MINUTE1 = '1m'
    MINUTE5 = '5m'
    MINUTE15 = '15m'
    MINUTE30 = '30m'
    HOURLY = '1h'

class FTYPE(Enum):
    MOM = 'MOM'
    LEV = 'LEV'
    VOL = 'VOL'
    VLM = 'VLM'
    
@dataclass
class FactorData:
    """
    Factor data of a certain trading period.
    """

    symbol: str
    exchange: Exchange
    datetime: datetime

    frequency: FREQ = None
    factor_type: FTYPE = None
    factor_name: str = None
    factor_value: float = 0.0

@dataclass
class FactorOverview:
    """Overviw of bar data for a certain trading period."""
    symbol: str
    exchange: Exchange

    frequency: FREQ = None
    factor_type: FTYPE = None
    factor_name: str = None
    count: int = 0
    start: datetime = None
    end: datetime = None

@dataclass
class RiskData:
    """
    Risk data of a certain trading period.
    """

    date_time: datetime
    group: str
    frequency: FREQ
    horizon: int
    risk_type: str
    risk_name: str
    risk_value: float