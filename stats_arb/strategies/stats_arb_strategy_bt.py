from vnpy.trader.utility import BarGenerator, ArrayManager
from vnpy.trader.constant import Interval
from vnpy_spreadtrading import (
    SpreadStrategyTemplate,
    SpreadAlgoTemplate,
    SpreadData,
    OrderData,
    TradeData,
    TickData,
    BarData
)

from .cutility import CBarGenerator,logger
from datetime import datetime


class SprdAtrIndStrategyBT(SpreadStrategyTemplate):
    """"""

    author = "Xiaohui"

    atr_window = 5
    atr_dev = 0.4
    max_pos = 2
    payup = 5
    interval = 5
    start_time = "9:30:00"
    end_time = "15:00:00"
    mat_day_lst = [datetime.strptime('2021-06-18',"%Y-%m-%d").date(),
                   datetime.strptime('2021-07-16',"%Y-%m-%d").date(),
                   datetime.strptime('2021-08-20',"%Y-%m-%d").date(),
                   datetime.strptime('2021-09-17',"%Y-%m-%d").date(),
                   datetime.strptime('2021-10-15',"%Y-%m-%d").date(),
                   datetime.strptime('2021-11-19',"%Y-%m-%d").date(),
                   datetime.strptime('2021-12-17',"%Y-%m-%d").date()]

    spread_pos = 0.0
    buy_price = 0.0
    sell_price = 0.0
    cover_price = 0.0
    short_price = 0.0
    spread_tran = 0
    update_time = None
    buy_algoid = ""
    sell_algoid = ""
    short_algoid = ""
    cover_algoid = ""

    parameters = [
        "atr_window",
        "atr_dev",
        "payup",
        "interval",
        "max_pos"
    ]
    variables = [
        "buy_price",
        "sell_price",
        "cover_price",
        "short_price",
        "spread_pos",     
        "spread_tran",
        "update_time",
        "buy_algoid",
        "sell_algoid",
        "short_algoid",
        "cover_algoid",
    ]

    def __init__(
        self,
        strategy_engine,
        strategy_name: str,
        spread: SpreadData,
        setting: dict
    ):
        """"""
        super().__init__(
            strategy_engine, strategy_name, spread, setting
        )

        self.start_t = datetime.strptime(self.start_time, "%H:%M:%S").time()
        self.end_t = datetime.strptime(self.end_time, "%H:%M:%S").time()
        self.bg = CBarGenerator(self.on_spread_bar,
                                window = 1,
                                on_window_bar = self.on_spread_day_bar,
                                interval = Interval.DAILY)
        self.am = ArrayManager(size = 10)
        self.log_local = logger()

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.load_bar(10)
        self.write_log("策略初始化")

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

        self.update_time = None
        self.buy_algoid = ""
        self.sell_algoid = ""
        self.short_algoid = ""
        self.cover_algoid = ""
        self.put_event()

    def on_spread_data(self):
        """
        update tick
        """
        tick = self.get_spread_tick()
        self.on_spread_tick(tick)
        self.put_event()
    
    def on_spread_tick(self, tick: TickData):
        """update bar"""
        self.bg.update_tick(tick)
        
    def on_spread_bar(self, bar: BarData):
        """Callback when spread 1 min bar updated"""
        
        # Trading is only allowed within given start/end time range
        # self.update_time = bar.datetime.time()
        # if self.update_time < self.start_t or self.update_time >= self.end_t:
        #     self.stop_open_algos()
        #     self.stop_close_algos()
        #     self.put_event()
        #     return

        #close all trades when maturity date
        if bar.datetime.date() in self.mat_day_lst:
            if self.spread_pos >0:
                if not self.sell_algoid:
                    self.sell_algoid = self.start_short_algo(
                        bar.low_price, abs(self.spread_pos), self.payup, self.interval)
            elif self.spread_pos <0:
                if not self.buy_algoid:
                    self.buy_algoid = self.start_long_algo(
                        bar.high_price, abs(self.spread_pos), self.payup, self.interval
                    )
            else:
                pass
            return
        
        #stop trading when max daily transaction reached
        if self.spread_tran >= self.max_pos and self.inited:
            self.bg.update_bar(bar)
            return
        
        # No position
        if not self.spread_pos:
            self.stop_close_algos()

            # Start open algos
            if not self.buy_algoid:
                self.buy_algoid = self.start_long_algo(
                    self.buy_price, self.max_pos, self.payup, self.interval
                )

            if not self.short_algoid:
                self.short_algoid = self.start_short_algo(
                    self.short_price, self.max_pos, self.payup, self.interval
                )

        # Long position
        elif self.spread_pos > 0:
            self.stop_open_algos()

            # Start sell close algo
            if not self.sell_algoid:
                self.sell_algoid = self.start_short_algo(
                    self.sell_price, abs(self.spread_pos), self.payup, self.interval
                )

        # Short position
        elif self.spread_pos < 0:
            self.stop_open_algos()

            # Start cover close algo
            if not self.cover_algoid:
                self.cover_algoid = self.start_long_algo(
                    self.cover_price, abs(self.spread_pos), self.payup, self.interval
                )
        self.bg.update_bar(bar)
        self.put_event()
    
    def on_spread_day_bar(self, bar:BarData):
        self.stop_all_algos()
        self.spread_tran=0
        self.am.update_bar(bar)
        if not self.am.inited:
            return
        
        atr = self.am.atr(self.atr_window)
        mid = bar.close_price
        self.buy_price = mid - atr * self.atr_dev
        self.short_price = mid + atr * self.atr_dev
        self.sell_price = mid
        self.cover_price = mid
        self.put_event()
        
    def on_spread_pos(self):
        """
        Callback when spread position is updated.
        """
        self.spread_pos = self.get_spread_pos()
        self.put_event()

    def on_spread_algo(self, algo: SpreadAlgoTemplate):
        """
        Callback when algo status is updated.
        """
        if not algo.is_active():
            if self.buy_algoid == algo.algoid:
                self.buy_algoid = ""
            elif self.sell_algoid == algo.algoid:
                self.sell_algoid = ""
            elif self.short_algoid == algo.algoid:
                self.short_algoid = ""
            else:
                self.cover_algoid = ""

        self.put_event()

    def on_order(self, order: OrderData):
        """
        Callback when order status is updated.
        """
        pass

    def on_trade(self, trade: TradeData):
        """
        Callback when new trade data is received.
        """
        self.spread_tran += abs(trade.volume)
        self.put_event()

    def stop_open_algos(self):
        """"""
        if self.buy_algoid:
            self.stop_algo(self.buy_algoid)

        if self.short_algoid:
            self.stop_algo(self.short_algoid)

    def stop_close_algos(self):
        """"""
        if self.sell_algoid:
            self.stop_algo(self.sell_algoid)

        if self.cover_algoid:
            self.stop_algo(self.cover_algoid)

class SprdAtrComStrategyBT(SpreadStrategyTemplate):
    """"""

    author = "Xiaohui"

    atr_window = 5
    atr_dev = 0.4
    range_lb = 0.0
    max_pos = 2
    payup = 5
    interval = 5
    end_hour = 23
    end_min = 59

    spread_pos = 0.0
    buy_price = 0.0
    sell_price = 0.0
    cover_price = 0.0
    short_price = 0.0
    spread_tran = 0
    update_time = None
    buy_algoid = ""
    sell_algoid = ""
    short_algoid = ""
    cover_algoid = ""

    parameters = [
        "atr_window",
        "atr_dev",
        "range_lb",
        "payup",
        "interval",
        "max_pos",
        "end_hour",
        "end_min"
    ]
    variables = [
        "buy_price",
        "sell_price",
        "cover_price",
        "short_price",
        "spread_pos",     
        "spread_tran",
        "update_time",
        "buy_algoid",
        "sell_algoid",
        "short_algoid",
        "cover_algoid",
    ]

    def __init__(
        self,
        strategy_engine,
        strategy_name: str,
        spread: SpreadData,
        setting: dict
    ):
        """"""
        super().__init__(
            strategy_engine, strategy_name, spread, setting
        )

        self.bg = CBarGenerator(self.on_spread_bar,
                                window = 1,
                                on_window_bar = self.on_spread_day_bar,
                                interval = Interval.DAILY,
                                end_h = self.end_hour,
                                end_m = self.end_min)
        self.am_day = ArrayManager(size = 10)
        self.am_1min = ArrayManager()
        self.log_local = logger(app_name=spread.name, log_file_name=strategy_name + '.log')

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.load_bar(10)
        self.write_log("策略初始化")

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

        self.update_time = None
        self.buy_algoid = ""
        self.sell_algoid = ""
        self.short_algoid = ""
        self.cover_algoid = ""
        self.put_event()

    def on_spread_data(self):
        """
        update tick
        """
        tick = self.get_spread_tick()
        self.on_spread_tick(tick)
        self.put_event()
    
    def on_spread_tick(self, tick: TickData):
        """update bar"""
        self.bg.update_tick(tick)
        
    def on_spread_bar(self, bar: BarData):
        """Callback when spread 1 min bar updated"""
        
        #stop trading when max daily transaction reached
        # if self.spread_tran >= self.max_pos and self.inited:
        #     self.bg.update_bar(bar)
        #     return
        
        # No position
        if not self.spread_pos:
            self.stop_close_algos()

            # Start open algos
            if not self.buy_algoid:
                self.buy_algoid = self.start_long_algo(
                    self.buy_price, self.max_pos, self.payup, self.interval
                )

            if not self.short_algoid:
                self.short_algoid = self.start_short_algo(
                    self.short_price, self.max_pos, self.payup, self.interval
                )

        # Long position
        elif self.spread_pos > 0:
            self.stop_open_algos()

            # Start sell close algo
            if not self.sell_algoid:
                self.sell_algoid = self.start_short_algo(
                    self.sell_price, abs(self.spread_pos), self.payup, self.interval
                )

        # Short position
        elif self.spread_pos < 0:
            self.stop_open_algos()

            # Start cover close algo
            if not self.cover_algoid:
                self.cover_algoid = self.start_long_algo(
                    self.cover_price, abs(self.spread_pos), self.payup, self.interval
                )
        self.am_1min.update_bar(bar)
        self.bg.update_bar(bar)
        self.put_event()
    
    def on_spread_day_bar(self, bar:BarData):
        self.stop_all_algos()
        self.spread_tran=0
        self.am_day.update_bar(bar)
        if not self.am_day.inited:
            return
        
        atr = self.am_day.atr(self.atr_window)
        # high1 = self.am.high[-1]
        # low1 = self.am.low[-1]
        # high2 = self.am.high[-2]
        # low2 = self.am.low[-2]
        # atr = 0.5*(high1 - low1) + 0.5*(high2 - low2)
        mid = bar.close_price
        if self.am_1min.inited:
            mid = self.am_1min.sma(10)
            
        self.buy_price = round(mid - max(atr * self.atr_dev, self.range_lb), 2)
        self.short_price = round(mid + max(atr * self.atr_dev, self.range_lb), 2)
        self.sell_price = round(mid, 2)
        self.cover_price = round(mid, 2)
        self.put_event()
        
    def on_spread_pos(self):
        """
        Callback when spread position is updated.
        """
        self.spread_pos = self.get_spread_pos()
        self.put_event()

    def on_spread_algo(self, algo: SpreadAlgoTemplate):
        """
        Callback when algo status is updated.
        """
        if not algo.is_active():
            if self.buy_algoid == algo.algoid:
                self.buy_algoid = ""
            elif self.sell_algoid == algo.algoid:
                self.sell_algoid = ""
            elif self.short_algoid == algo.algoid:
                self.short_algoid = ""
            else:
                self.cover_algoid = ""

        self.put_event()

    def on_order(self, order: OrderData):
        """
        Callback when order status is updated.
        """
        pass

    def on_trade(self, trade: TradeData):
        """
        Callback when new trade data is received.
        """
        self.spread_tran += abs(trade.volume)
        self.put_event()

    def stop_open_algos(self):
        """"""
        if self.buy_algoid:
            self.stop_algo(self.buy_algoid)

        if self.short_algoid:
            self.stop_algo(self.short_algoid)

    def stop_close_algos(self):
        """"""
        if self.sell_algoid:
            self.stop_algo(self.sell_algoid)

        if self.cover_algoid:
            self.stop_algo(self.cover_algoid)

class SprdAtrComFrStrategyBT(SpreadStrategyTemplate):
    """"""

    author = "Xiaohui"

    atr_window = 5
    atr_dev = 0.4
    range_lb = 0.0
    max_pos = 2
    payup = 5
    interval = 5
    end_hour = 23
    end_min = 59

    spread_pos = 0.0
    long_price = 0.0
    short_price = 0.0
    spread_tran = 0
    update_time = None
    long_algoid = ""
    short_algoid = ""

    parameters = [
        "atr_window",
        "atr_dev",
        "range_lb",
        "payup",
        "interval",
        "max_pos",
        "end_hour",
        "end_min"
    ]
    variables = [
        "long_price",
        "short_price",
        "spread_pos",     
        "spread_tran",
        "update_time",
        "long_algoid",
        "short_algoid"
    ]

    def __init__(
        self,
        strategy_engine,
        strategy_name: str,
        spread: SpreadData,
        setting: dict
    ):
        """"""
        super().__init__(
            strategy_engine, strategy_name, spread, setting
        )

        self.bg = CBarGenerator(self.on_spread_bar,
                                window = 1,
                                on_window_bar = self.on_spread_day_bar,
                                interval = Interval.DAILY,
                                end_h = self.end_hour,
                                end_m = self.end_min)
        self.am_day = ArrayManager(size = 10)
        self.am_1min = ArrayManager()
        self.log_local = logger(app_name=spread.name, log_file_name=strategy_name + '.log')

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.load_bar(10)
        self.write_log("策略初始化")

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

        self.update_time = None
        self.long_algoid = ""
        self.short_algoid = ""
        self.put_event()

    def on_spread_data(self):
        """
        update tick
        """
        tick = self.get_spread_tick()
        self.on_spread_tick(tick)
        self.put_event()
    
    def on_spread_tick(self, tick: TickData):
        """update bar"""
        self.bg.update_tick(tick)
        
    def on_spread_bar(self, bar: BarData):
        """Callback when spread 1 min bar updated"""
        
        # No position
        if not self.spread_pos:
            # Start algos
            if not self.long_algoid:
                self.long_algoid = self.start_long_algo(
                    self.long_price, self.max_pos, self.payup, self.interval
                )

            if not self.short_algoid:
                self.short_algoid = self.start_short_algo(
                    self.short_price, self.max_pos, self.payup, self.interval
                )
        # Long position
        elif self.spread_pos > 0:
            self.stop_long_algo()
            # Start sell close algo
            if not self.short_algoid:
                self.short_algoid = self.start_short_algo(
                    self.short_price, abs(self.spread_pos), self.payup, self.interval
                )
        # Short position
        elif self.spread_pos < 0:
            self.stop_short_algo()
            # Start cover close algo
            if not self.long_algoid:
                self.long_algoid = self.start_long_algo(
                    self.long_price, abs(self.spread_pos), self.payup, self.interval
                )
        self.am_1min.update_bar(bar)
        self.bg.update_bar(bar)
        self.put_event()
    
    def on_spread_day_bar(self, bar:BarData):
        self.stop_all_algos()
        self.spread_tran=0
        self.am_day.update_bar(bar)
        if not self.am_day.inited:
            return
        
        atr = self.am_day.atr(self.atr_window)
        mid = bar.close_price
        if self.am_1min.inited:
            mid = self.am_1min.sma(10)
            
        self.long_price = round(mid - max(atr * self.atr_dev, self.range_lb), 2)
        self.short_price = round(mid + max(atr * self.atr_dev, self.range_lb), 2)
        self.put_event()
        
    def on_spread_pos(self):
        """
        Callback when spread position is updated.
        """
        self.spread_pos = self.get_spread_pos()
        self.put_event()

    def on_spread_algo(self, algo: SpreadAlgoTemplate):
        """
        Callback when algo status is updated.
        """
        if not algo.is_active():
            if self.long_algoid == algo.algoid:
                self.long_algoid = ""
            elif self.short_algoid == algo.algoid:
                self.short_algoid = ""
            else:
                pass

        self.put_event()

    def on_order(self, order: OrderData):
        """
        Callback when order status is updated.
        """
        pass

    def on_trade(self, trade: TradeData):
        """
        Callback when new trade data is received.
        """
        self.spread_tran += abs(trade.volume)
        self.put_event()
            
    def stop_long_algo(self):
        if self.long_algoid:
            self.stop_algo(self.long_algoid)
            
    def stop_short_algo(self):
        if self.short_algoid:
            self.stop_algo(self.short_algoid)