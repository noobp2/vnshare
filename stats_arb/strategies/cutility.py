from typing import Callable, Type, Dict, List
from functools import lru_cache
from datetime import date, datetime
import logging
import json

from vnpy.trader.database import BaseDatabase, get_database
from vnpy.trader.utility import BarGenerator
from vnpy_spreadtrading.backtesting import DailyResult
from vnpy.trader.object import BarData
from vnpy.trader.utility import round_to, floor_to, ceil_to, extract_vt_symbol
from vnpy.trader.constant import (Direction, Offset, Exchange,
                                  Interval, Status)
from vnpy_spreadtrading.base import (SpreadData,
                                     LegData,
                                     query_bar_from_rq)

class CBarGenerator(BarGenerator):
    def __init__(
        self,
        on_bar: Callable,
        window: int = 0,
        on_window_bar: Callable = None,
        interval: Interval = Interval.MINUTE,
        end_h: int = 14,
        end_m: int = 59):
        
        super().__init__(on_bar, window, on_window_bar, interval)
        self.day_bar: BarData = None
        self.end_hour: int = end_h
        self.end_min: int = end_m
    
    def update_bar(self, bar: BarData) -> None:
        """
        update 1 minutes bar into generator
        """ 
        if self.interval == Interval.MINUTE:
            self.update_bar_minute_window(bar)
        elif self.interval == Interval.HOUR:
            self.update_bar_hour_window(bar)
        elif self.interval == Interval.DAILY:
            self.update_bar_day_window(bar)
        else:
            pass
    
    def update_bar_day_window(self, bar: BarData) -> None:
        """
        [summary]
        This method is used by backtesting strategy with EOD signal. default market close time is 15:00
        """
        # If not inited, create window bar object
        if not self.day_bar:
            dt = bar.datetime.replace(hour=0, minute=0, second=0, microsecond=0)
            self.day_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=dt,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price,
                close_price=bar.close_price,
                volume=bar.volume,
                turnover=bar.turnover,
                open_interest=bar.open_interest
            )
            return

        finished_bar = None

        # If minute is 59, update minute bar into window bar and push
        if bar.datetime.minute == self.end_min and bar.datetime.hour == self.end_hour:
            self.day_bar.high_price = max(
                self.day_bar.high_price,
                bar.high_price
            )
            self.day_bar.low_price = min(
                self.day_bar.low_price,
                bar.low_price
            )

            self.day_bar.close_price = bar.close_price
            self.day_bar.volume += bar.volume
            self.day_bar.turnover += bar.turnover
            self.day_bar.open_interest = bar.open_interest
            
            self.day_bar.datetime = bar.datetime

            finished_bar = self.day_bar
            self.day_bar = None
        else:
            self.day_bar.high_price = max(
                self.day_bar.high_price,
                bar.high_price
            )
            self.day_bar.low_price = min(
                self.day_bar.low_price,
                bar.low_price
            )

            self.day_bar.close_price = bar.close_price
            self.day_bar.volume += bar.volume
            self.day_bar.turnover += bar.turnover
            self.day_bar.open_interest = bar.open_interest

        # Push finished window bar
        if finished_bar:
            self.on_x_interval_bar(finished_bar)

    def on_x_interval_bar(self, bar: BarData) -> None:
        """"""
        if self.window == 1:
            self.on_window_bar(bar)
        else:
            if not self.window_bar:
                self.window_bar = BarData(
                    symbol=bar.symbol,
                    exchange=bar.exchange,
                    datetime=bar.datetime,
                    gateway_name=bar.gateway_name,
                    open_price=bar.open_price,
                    high_price=bar.high_price,
                    low_price=bar.low_price
                )
            else:
                self.window_bar.high_price = max(
                    self.window_bar.high_price,
                    bar.high_price
                )
                self.window_bar.low_price = min(
                    self.window_bar.low_price,
                    bar.low_price
                )

            self.window_bar.close_price = bar.close_price
            self.window_bar.volume += bar.volume
            self.window_bar.turnover += bar.turnover
            self.window_bar.open_interest = bar.open_interest

            self.interval_count += 1
            if not self.interval_count % self.window:
                self.interval_count = 0
                self.on_window_bar(self.window_bar)
                self.window_bar = None

class logger:
    def __init__(self, app_name: str = __name__, log_file_name: str = 'spread_trading.log'):
        logging.basicConfig(level=logging.INFO)
        self.logger_ = logging.getLogger(app_name)
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file_name)
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        self.logger_.addHandler(c_handler)
        self.logger_.addHandler(f_handler)
    
    def warning(self, msg: str):
        self.logger_.warning(msg)
    
    def error(self, msg: str):
        self.logger_.error(msg)
        
    def info(self, msg: str):
        self.logger_.info(msg)

class CDailyResult(DailyResult):
    def __init__(self, date: date, close_price: float, close_notional: float):
        super().__init__(date, close_price)
        self.close_notional = close_notional
    
    def calculate_pnl(
        self,
        pre_close: float,
        start_pos: float,
        size: int,
        rate: float,
        slippage: float
    ):
        """"""
        # If no pre_close provided on the first day,
        # use value 1 to avoid zero division error
        if pre_close:
            self.pre_close = pre_close
        else:
            self.pre_close = 1

        # Holding pnl is the pnl from holding position at day start
        self.start_pos = start_pos
        self.end_pos = start_pos

        self.holding_pnl = self.start_pos * (self.close_price - self.pre_close) * size

        # Trading pnl is the pnl from new trade during the day
        self.trade_count = len(self.trades)

        for trade in self.trades:
            if trade.direction == Direction.LONG:
                pos_change = trade.volume
            else:
                pos_change = -trade.volume

            self.end_pos += pos_change

            turnover = trade.volume * size * self.close_notional
            self.trading_pnl += pos_change * \
                (self.close_price - trade.price) * size
            self.slippage += trade.volume * size * slippage

            self.turnover += turnover
            self.commission += turnover * rate

        # Net pnl takes account of commission and slippage cost
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission - self.slippage

@lru_cache(maxsize=999)
def load_bar_data(
    spread: SpreadData,
    interval: Interval,
    start: datetime,
    end: datetime,
    pricetick: float = 0
):
    """"""
    database: BaseDatabase = get_database()

    # Load bar data of each spread leg
    leg_bars: Dict[str, Dict] = {}

    for vt_symbol in spread.legs.keys():
        symbol, exchange = extract_vt_symbol(vt_symbol)

        # First, try to query history from RQData
        bar_data: List[BarData] = query_bar_from_rq(
            symbol, exchange, interval, start, end
        )

        # If failed, query history from database
        if not bar_data:
            bar_data = database.load_bar_data(
                symbol, exchange, interval, start, end
            )

        bars: Dict[datetime, BarData] = {bar.datetime: bar for bar in bar_data}
        leg_bars[vt_symbol] = bars
    database.db.close()
    
    # Calculate spread bar data
    spread_bars: List[BarData] = []

    for dt in bars.keys():
        spread_price = 0
        spread_value = 0
        spread_available = True

        leg_data_close = {}
        leg_data_open = {}
        for variable, leg in spread.variable_legs.items():
            leg_bar = leg_bars[leg.vt_symbol].get(dt, None)

            if leg_bar:
                # 缓存该腿当前的价格
                leg_data_close[variable] = leg_bar.close_price
                leg_data_open[variable] = leg_bar.open_price

                # 基于交易乘数累计价值
                trading_multiplier = spread.trading_multipliers[leg.vt_symbol]
                spread_value += abs(trading_multiplier) * leg_bar.close_price
            else:
                spread_available = False

        if spread_available:
            spread_price_close = spread.parse_formula(spread.price_code, leg_data_close)
            spread_price_open = spread.parse_formula(spread.price_code, leg_data_open)
            if pricetick:
                spread_price_close = round_to(spread_price_close, pricetick)
                spread_price_open = round_to(spread_price_open, pricetick)

            spread_bar = BarData(
                symbol=spread.name,
                exchange=exchange.LOCAL,
                datetime=dt,
                interval=interval,
                open_price=spread_price_open,
                high_price=max(spread_price_open,spread_price_close),
                low_price=min(spread_price_open,spread_price_close),
                close_price=spread_price_close,
                gateway_name="SPREAD",
            )
            spread_bar.value = spread_value
            spread_bars.append(spread_bar)
            
    return spread_bars

def load_json_file(filepath:str):
    with open(filepath, mode="r", encoding="UTF-8") as f:
        data = json.load(f)
    return data

def read_trans_from_log(log_file:str = "spread_trading.log",keep_phrases:List[str] = ["algoid"]):
    trans = []
    with open(log_file) as f:
        f = f.readlines()
    for line in f:
        for phrase in keep_phrases:
            if phrase in line:
                trans.append(line)
                break
    return trans       