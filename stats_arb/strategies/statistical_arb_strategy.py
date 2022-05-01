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


class SprdAtrIndStrategy(SpreadStrategyTemplate):
    """"""

    author = "Xiaohui"

    atr_window = 5
    atr_dev = 0.4
    range_lb = 0.0
    max_pos = 2
    payup = 5
    interval = 5
    start_time = "9:30:00"
    end_time = "15:00:00"
    next_mat_day = "2022-01-15"
    eod_hour = 14
    eod_min = 59

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
        "next_mat_day",
        "eod_hour",
        "eod_min"
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
        self.next_mat_d = datetime.strptime(self.next_mat_day,"%Y-%m-%d").date()
        self.is_mat_d = False
        self.is_in_trading_hours = False
        self.algo_starter_triggered = False
        self.algo_event = False
        self.pos_event = False
        self.bg = CBarGenerator(self.on_spread_bar,
                                window = 1,
                                on_window_bar = self.on_spread_day_bar,
                                interval = Interval.DAILY,
                                end_h = self.eod_hour,
                                end_m = self.eod_min)
        self.am_day = ArrayManager(size = 10)
        self.am_1min = ArrayManager()
        #self.log_local = logger()
        self.log_local = logger(app_name=spread.name, log_file_name=strategy_name + '.log')

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.load_bar(30)
        self.is_mat_d = False
        self.algo_starter_triggered = False
        self.spread_tran=0
        self.write_log("策略初始化")

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.is_in_trading_hours = True
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.algo_starter_triggered = False
        self.stop_all_algos()
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
        if self.trading and not self.algo_starter_triggered:
            self.excute_algos()
            self.algo_starter_triggered = True
        
        if self.pos_event and self.algo_event:
            self.excute_algos()
            
        tick = self.get_spread_tick()
        self.on_spread_tick(tick)
        self.put_event()
    
    def on_spread_tick(self, tick: TickData):
        """update bar"""
        self.bg.update_tick(tick)
        
    def on_spread_bar(self, bar: BarData):
        """Callback when spread 1 min bar updated"""
        # Trading is only allowed within given start/end time range
        
        self.update_time = bar.datetime.time()
        if self.update_time < self.start_t or self.update_time >= self.end_t:
            self.spread_tran=0
            self.stop_strategy()
            self.is_in_trading_hours = False
            return
        
        #close all trades when maturity date
        if bar.datetime.date() == self.next_mat_d:
            self.is_mat_d = True
            self.stop_all_algos()
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
        self.am_1min.update_bar(bar)
        self.bg.update_bar(bar)
        self.put_event()
    
    def on_spread_day_bar(self, bar:BarData):
        self.stop_all_algos()
        self.am_day.update_bar(bar)
        if not self.am_day.inited:
            return
        
        atr = self.am_day.atr(self.atr_window)
        mid = bar.close_price
        if self.am_1min.inited:
            mid = self.am_1min.sma(10)
            
        self.buy_price = round(mid - max(atr * self.atr_dev, self.range_lb),3)
        self.short_price = round(mid + max(atr * self.atr_dev, self.range_lb),3)
        self.sell_price = round(mid,3)
        self.cover_price = round(mid,3)
        self.reset_all_algos()
        self.excute_algos()
        self.put_event()
        
    def on_spread_pos(self):
        """
        Callback when spread position is updated.
        """
        spread_pos_new = self.get_spread_pos()
        if not self.spread_pos == spread_pos_new:
            self.pos_event = True
            self.write_log("sprd_pos_pre:{0};sprd_pos_new:{1};trans:{2};buy:{3};sell:{4};short:{5};cover{6}"
                       .format(self.spread_pos, spread_pos_new, self.spread_tran,self.buy_price,self.sell_price,self.short_price,self.cover_price))
        self.spread_pos = spread_pos_new
            
        self.put_event()
    
    def excute_algos(self):
        #not in trading hours
        if not self.is_in_trading_hours:
            self.reset_excution_event()
            self.stop_strategy()
            return
            
        # maturity date, close all positions
        if self.is_mat_d:
            self.reset_excution_event()
            self.put_event()
            return
        
        if self.spread_tran >= self.max_pos and self.inited:
            self.reset_excution_event()
            self.put_event()
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
        self.reset_excution_event()
        self.put_event()
        
    def on_spread_algo(self, algo: SpreadAlgoTemplate):
        """
        Callback when algo status is updated.
        """
        if not algo.is_active():
            if algo.traded_volume > 0:
                self.algo_event = True
                self.write_log("algoid:{0};direction:{1};price:{2};volume:{3};status:{4};traded_volume:{5};traded_price:{6}"
                               .format(algo.algoid,algo.direction,algo.price,algo.volume,algo.status,algo.traded_volume,algo.traded_price))
            self.update_spread_trans(algo)
            self.reset_algo(algo.algoid)

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
    
    def stop_strategy(self):
        self.strategy_engine.stop_strategy(self.strategy_name)
    
    def update_spread_trans(self, algo: SpreadAlgoTemplate):
        if algo.algoid == self.buy_algoid or algo.algoid == self.short_algoid:
            self.spread_tran += algo.traded_volume
    
    def reset_excution_event(self):
        self.pos_event = False
        self.algo_event = False
        
    def reset_algo(self, algoid: str):
        if self.buy_algoid == algoid:
            self.buy_algoid = ""
        elif self.sell_algoid == algoid:
            self.sell_algoid = ""
        elif self.short_algoid == algoid:
            self.short_algoid = ""
        elif self.cover_algoid == algoid:
            self.cover_algoid = ""
        else:
            pass
    
    def reset_all_algos(self):
        self.buy_algoid = ""
        self.sell_algoid = ""
        self.short_algoid = ""
        self.cover_algoid = ""
        
    def write_log(self, msg: str):
        self.log_local.info(msg)
        return super().write_log(msg)

class SprdAtrComStrategy(SpreadStrategyTemplate):
    """"""

    author = "Xiaohui"

    atr_window = 3
    atr_dev = 0.4
    range_lb = 0.0
    max_pos = 1
    payup = 1
    interval = 5
    eod_hour = 13
    eod_min = 40
    rolling_day = "2022-03-01"
    rolling_time = "21:00:00"
    start_time = "09:00:00"
    end_time = "01:00:00"

    spread_pos = 0.0
    buy_price = 0.0
    sell_price = 0.0
    cover_price = 0.0
    short_price = 0.0
    spread_tran = 0
    mid_time = None
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
        "eod_hour",
        "eod_min",
        "rolling_day",
        "rolling_time",
        "end_time"
    ]
    variables = [
        "buy_price",
        "sell_price",
        "cover_price",
        "short_price",
        "spread_pos",     
        "spread_tran",
        "mid_time",
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
        self.rolling_t = datetime.strptime(self.rolling_day + ' '+ self.rolling_time,"%Y-%m-%d %H:%M:%S")
        self.close_pos = False
        self.is_in_trading_hours = False
        self.algo_starter_triggered = False
        self.algo_event = False
        self.pos_event = False
        self.bg = CBarGenerator(self.on_spread_bar,
                                window = 1,
                                on_window_bar = self.on_spread_day_bar,
                                interval = Interval.DAILY,
                                end_h = self.eod_hour,
                                end_m = self.eod_min)
        self.am_day = ArrayManager(size = 10)
        self.am_1min = ArrayManager()
        self.log_local = logger(app_name=spread.name, log_file_name=strategy_name + '.log')

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.load_bar(30)
        self.close_pos = False
        self.algo_starter_triggered = False
        self.write_log("策略初始化")

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.is_in_trading_hours = False
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.algo_starter_triggered = False
        self.stop_all_algos()
        self.buy_algoid = ""
        self.sell_algoid = ""
        self.short_algoid = ""
        self.cover_algoid = ""
        self.write_log("策略停止")
        self.put_event()

    def on_spread_data(self):
        """
        update tick
        """
        if self.trading and not self.algo_starter_triggered:
            self.excute_algos()
            self.algo_starter_triggered = True
        
        if self.pos_event and self.algo_event:
            self.excute_algos()
            
        tick = self.get_spread_tick()
        self.on_spread_tick(tick)
        self.put_event()
    
    def on_spread_tick(self, tick: TickData):
        """update bar"""
        self.bg.update_tick(tick)
        
    def on_spread_bar(self, bar: BarData):
        """Callback when spread 1 min bar updated"""
        #close all trades when maturity date
        if bar.datetime.replace(tzinfo=None) >= self.rolling_t:
            self.close_pos = True
        
        if not self.in_trade_hours(bar.datetime.time()):
            self.stop_strategy()
            self.is_in_trading_hours = False
            return
        
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
            
        self.buy_price = round(mid - max(atr * self.atr_dev, self.range_lb), 2)
        self.short_price = round(mid + max(atr * self.atr_dev, self.range_lb), 2)
        self.sell_price = round(mid, 2)
        self.cover_price = round(mid, 2)
        self.mid_time = bar.datetime.replace(tzinfo=None)
        self.reset_all_algos()
        self.excute_algos()
        self.put_event()
        
    def on_spread_pos(self):
        """
        Callback when spread position is updated.
        """
        spread_pos_new = self.get_spread_pos()
        if not self.spread_pos == spread_pos_new:
            self.pos_event = True
            self.write_log("sprd_pos_pre:{0};sprd_pos_new:{1};trans:{2};buy:{3};sell:{4};short:{5};cover{6}"
                       .format(self.spread_pos, spread_pos_new, self.spread_tran,self.buy_price,self.sell_price,self.short_price,self.cover_price))
        self.spread_pos = spread_pos_new  
        self.put_event()
        
    def excute_algos(self):
        #not in trading hours
        if not self.is_in_trading_hours:
            self.reset_excution_event()
            self.stop_strategy()
            return
            
        # rolling date, close all positions
        if self.close_pos:
            self.stop_open_algos()
            if self.spread_pos >0:
                if not self.sell_algoid:
                    self.sell_algoid = self.start_short_algo(
                        self.sell_price, abs(self.spread_pos), self.payup, self.interval)
            elif self.spread_pos <0:
                if not self.buy_algoid:
                    self.buy_algoid = self.start_long_algo(
                        self.buy_price, abs(self.spread_pos), self.payup, self.interval
                    )
            else:
                pass
            self.reset_excution_event()
            self.put_event()
            return
        
        if self.spread_tran >= self.max_pos and self.inited:
            self.reset_excution_event()
            self.put_event()
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
        self.reset_excution_event()    
        self.put_event()
        
    def on_spread_algo(self, algo: SpreadAlgoTemplate):
        """
        Callback when algo status is updated.
        """
        if not algo.is_active():
            if algo.traded_volume > 0:
                self.algo_event = True
                self.write_log("algoid:{0};direction:{1};price:{2};volume:{3};status:{4};traded_volume:{5};traded_price:{6}"
                               .format(algo.algoid,algo.direction,algo.price,algo.volume,algo.status,algo.traded_volume,algo.traded_price))
            self.reset_algo(algo.algoid)

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
    
    def reset_algo(self, algoid: str):
        if self.buy_algoid == algoid:
            self.buy_algoid = ""
        elif self.sell_algoid == algoid:
            self.sell_algoid = ""
        elif self.short_algoid == algoid:
            self.short_algoid = ""
        elif self.cover_algoid == algoid:
            self.cover_algoid = ""
        else:
            pass
    
    def reset_all_algos(self):
        self.buy_algoid = ""
        self.sell_algoid = ""
        self.short_algoid = ""
        self.cover_algoid = ""
    
    def write_log(self, msg: str):
        self.log_local.info(msg)
        return super().write_log(msg)
    
    def reset_excution_event(self):
        self.pos_event = False
        self.algo_event = False
    
    def stop_strategy(self):
        self.strategy_engine.stop_strategy(self.strategy_name)
    
    def in_trading_hours(self, bar_time:datetime.time):
        result = True
        if self.start_t > self.end_t:
            if bar_time < self.start_t and bar_time >= self.end_t:
                result = False
            else:
                result = True
        else:
            if bar_time < self.start_t or bar_time >= self.end_t:
                result = False
            else:
                result = True
        return result

class SprdAtrComFrStrategy(SpreadStrategyTemplate):
    """"""

    author = "Xiaohui"

    atr_window = 3
    atr_dev = 0.4
    range_lb = 0.0
    max_pos = 1
    payup = 1
    interval = 5
    eod_hour = 13
    eod_min = 40
    rolling_day = "2022-03-01"
    rolling_time = "21:00:00"
    start_time = "09:00:00"
    end_time = "01:00:00"

    spread_pos = 0.0
    long_price = 0.0
    short_price = 0.0
    spread_tran = 0
    mid_time = None
    long_algoid = ""
    short_algoid = ""

    parameters = [
        "atr_window",
        "atr_dev",
        "range_lb",
        "payup",
        "interval",
        "max_pos",
        "eod_hour",
        "eod_min",
        "rolling_day",
        "rolling_time",
        "end_time"
    ]
    variables = [
        "long_price",
        "short_price",
        "spread_pos",     
        "spread_tran",
        "mid_time",
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
        self.start_t = datetime.strptime(self.start_time, "%H:%M:%S").time()
        self.end_t = datetime.strptime(self.end_time, "%H:%M:%S").time()
        self.rolling_t = datetime.strptime(self.rolling_day + ' '+ self.rolling_time,"%Y-%m-%d %H:%M:%S")
        self.close_pos = False
        self.is_in_trading_hours = False
        self.algo_starter_triggered = False
        self.algo_event = False
        self.pos_event = False
        self.bg = CBarGenerator(self.on_spread_bar,
                                window = 1,
                                on_window_bar = self.on_spread_day_bar,
                                interval = Interval.DAILY,
                                end_h = self.eod_hour,
                                end_m = self.eod_min)
        self.am_day = ArrayManager(size = 10)
        self.am_1min = ArrayManager()
        self.log_local = logger(app_name=spread.name, log_file_name=strategy_name + '.log')

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.load_bar(30)
        self.close_pos = False
        self.algo_starter_triggered = False
        self.write_log("策略初始化")

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.is_in_trading_hours = True
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.algo_starter_triggered = False
        self.stop_all_algos()
        self.long_algoid = ""
        self.short_algoid = ""
        self.write_log("策略停止")
        self.put_event()

    def on_spread_data(self):
        """
        update tick
        """
        if self.trading and not self.algo_starter_triggered:
            self.excute_algos()
            self.algo_starter_triggered = True
        
        if self.pos_event and self.algo_event:
            self.excute_algos()
            
        tick = self.get_spread_tick()
        self.on_spread_tick(tick)
        self.put_event()
    
    def on_spread_tick(self, tick: TickData):
        """update bar"""
        self.bg.update_tick(tick)
        
    def on_spread_bar(self, bar: BarData):
        """Callback when spread 1 min bar updated"""
        #close all trades when maturity date
        if bar.datetime.replace(tzinfo=None) >= self.rolling_t:
            self.close_pos = True
        
        if not self.in_trading_hours(bar.datetime.time()):
            self.stop_strategy()
            self.is_in_trading_hours = False
            return
        
        self.am_1min.update_bar(bar)
        self.bg.update_bar(bar)
        self.put_event()
    
    def on_spread_day_bar(self, bar:BarData):
        self.stop_all_algos()
        self.am_day.update_bar(bar)
        if not self.am_day.inited:
            return
        
        atr = self.am_day.atr(self.atr_window)
        mid = bar.close_price
        if self.am_1min.inited:
            mid = self.am_1min.sma(10)
            
        self.long_price = round(mid - max(atr * self.atr_dev, self.range_lb), 2)
        self.short_price = round(mid + max(atr * self.atr_dev, self.range_lb), 2)
        self.mid_time = bar.datetime.replace(tzinfo=None)
        self.reset_all_algos()
        self.excute_algos()
        self.put_event()
        
    def on_spread_pos(self):
        """
        Callback when spread position is updated.
        """
        spread_pos_new = self.get_spread_pos()
        if not self.spread_pos == spread_pos_new:
            self.write_log("sprd_pos_pre:{0};sprd_pos_new:{1};trans:{2};long:{3};short:{4}"
                       .format(self.spread_pos, spread_pos_new, self.spread_tran,self.long_price,self.short_price))
            self.pos_event = True
        self.spread_pos = spread_pos_new    
        self.put_event()
    
    def excute_algos(self):
        #not in trading hours
        if not self.is_in_trading_hours:
            self.reset_excution_event()
            self.stop_strategy()
            return
            
        # rolling date, close all positions
        if self.close_pos:
            if self.spread_pos >0:
                self.stop_long_algo()
                if not self.short_algoid:
                    self.short_algoid = self.start_short_algo(
                        self.short_price, abs(self.spread_pos), self.payup, self.interval)
            elif self.spread_pos <0:
                self.stop_short_algo()
                if not self.long_algoid:
                    self.long_algoid = self.start_long_algo(
                        self.long_price, abs(self.spread_pos), self.payup, self.interval
                    )
            else:
                self.stop_all_algos()
            self.reset_excution_event()
            self.put_event()
            return
        
        # No position
        if not self.spread_pos:
            # Start open algos
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
            #stop long algo
            self.stop_long_algo()
            # Start short algo
            if not self.short_algoid:
                self.short_algoid = self.start_short_algo(
                    self.short_price, abs(self.spread_pos), self.payup, self.interval
                )

        # Short position
        elif self.spread_pos < 0:
            #stop short algo
            self.stop_short_algo()
            # Start cover close algo
            if not self.long_algoid:
                self.long_algoid = self.start_long_algo(
                    self.long_price, abs(self.spread_pos), self.payup, self.interval
                )
        self.reset_excution_event()    
        self.put_event()
        
    def on_spread_algo(self, algo: SpreadAlgoTemplate):
        """
        Callback when algo status is updated.
        """
        if not algo.is_active():
            if algo.traded_volume > 0:
                self.write_log("algoid:{0};direction:{1};price:{2};volume:{3};status:{4};traded_volume:{5};traded_price:{6}"
                               .format(algo.algoid,algo.direction,algo.price,algo.volume,algo.status,algo.traded_volume,algo.traded_price))
            if self.long_algoid == algo.algoid:
                self.long_algoid = ""
            elif self.short_algoid == algo.algoid:
                self.short_algoid = ""
            else:
                pass
            self.algo_event = True

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
        
    def stop_long_algo(self):
        if self.long_algoid:
            self.stop_algo(self.long_algoid)
            
    def stop_short_algo(self):
        if self.short_algoid:
            self.stop_algo(self.short_algoid)
    
    def reset_all_algos(self):
        self.long_algoid = ""
        self.short_algoid = ""
    
    def reset_excution_event(self):
        self.pos_event = False
        self.algo_event = False
                
    def write_log(self, msg: str):
        self.log_local.info(msg)
        return super().write_log(msg)
    
    def stop_strategy(self):
        self.strategy_engine.stop_strategy(self.strategy_name)
    
    def in_trading_hours(self, bar_time:datetime.time):
        result = True
        if self.start_t > self.end_t:
            if bar_time < self.start_t and bar_time >= self.end_t:
                result = False
            else:
                result = True
        else:
            if bar_time < self.start_t or bar_time >= self.end_t:
                result = False
            else:
                result = True
        return result