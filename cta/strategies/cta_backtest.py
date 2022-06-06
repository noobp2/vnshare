from pyparsing import null_debug_action
from vnpy.trader.utility import BarGenerator, ArrayManager,round_to
from vnpy.trader.constant import Interval,Offset,Direction
from vnpy_ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)

from .cutility import CBarGenerator,logger
from datetime import datetime

import talib
import numpy as np


class StochRsiScalpStrategy(CtaTemplate):
    """"""
    author = "Xiaohui"

    rsi_signal = 20
    rsi_fast_window = 7
    rsi_slow_window = 20
    k_fast_window = 9
    k_slow_window = 3
    fixed_size = 1
    captial_alloc = 100000
    size = 200
    pricetick = 0.2
    risk_tol = 1
    atr_tol = 1
    win_ratio = 2

    rsi_value = 0
    rsi_long = 0
    rsi_short = 0
    rsi_ma = 0
    k_value = 0
    d_value = 0
    fast_ma = 0
    slow_ma = 0
    ma_trend = 0
    stop_loss_limit = 0
    atr = 0

    parameters = ["rsi_signal", "rsi_fast_window","rsi_slow_window",
                  "k_fast_window","k_slow_window",
                  "fixed_size","captial_alloc",
                  "size", "pricetick","risk_tol","atr_tol","win_ratio"]

    variables = ["rsi_value", "rsi_long", "rsi_short","rsi_ma",
                 "k_value","d_value",
                 "fast_ma", "slow_ma", "ma_trend","stop_loss_limit","atr"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.rsi_ma = 50
        self.rsi_long = self.rsi_ma - self.rsi_signal
        self.rsi_short = self.rsi_ma + self.rsi_signal
        self.rsi_arry = None
        self.am1 = ArrayManager()

        self.bg5 = BarGenerator(self.on_bar, 5, self.on_5min_bar)
        self.am5 = ArrayManager()

        self.bg15 = BarGenerator(self.on_bar, 15, self.on_15min_bar)
        self.am15 = ArrayManager()
        
        self.traded_price = 0
        self.stop_loss_limit = round_to((self.captial_alloc * self.risk_tol/100)/self.size, self.pricetick)
        self.pre_ref_price = 0
        self.watch_kd = False

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.load_bar(10)

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

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        self.bg5.update_tick(tick)

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.cancel_all()
        self.bg5.update_bar(bar)
        self.bg15.update_bar(bar)
        
        self.am1.update_bar(bar)
        
        if not self.am1.inited:
            return
        
        self.rsi_arry = self.am1.rsi(self.rsi_fast_window, True)
        self.rsi_value = self.rsi_arry[-1]
        
        rsi_ma_arry = talib.SMA(self.rsi_arry, self.rsi_slow_window)
        self.rsi_ma = rsi_ma_arry[-1]
        
        self.k_value, self.d_value = self.am1.stoch(self.k_fast_window,self.k_slow_window, 0, self.k_slow_window, 0)
        
        self.atr = self.am1.atr(self.rsi_slow_window)
        stop_loss_signal = self.stop_loss(bar)
        exit_gain_signal = self.exit_gain(bar)
        open_pos_signal = self.open_position()
        
        #do not open position if atr is low
        if self.atr < bar.close_price * 0.0001:
            open_pos_signal = 0

        if self.pos == 0:
            if open_pos_signal > 0:
                self.buy(bar.close_price + 5 * self.pricetick, self.fixed_size)
            elif open_pos_signal < 0:
                self.short(bar.close_price - 5 * self.pricetick, self.fixed_size)
        
        elif self.pos > 0:
            if stop_loss_signal or exit_gain_signal:
                self.sell(bar.close_price - 5 * self.pricetick, abs(self.pos))

        elif self.pos < 0:
            if stop_loss_signal or exit_gain_signal:
                self.cover(bar.close_price + 5 * self.pricetick, abs(self.pos))
        
        self.put_event()

    def on_5min_bar(self, bar: BarData):
        """"""
        if  not self.am1.inited:
            return

        if self.rsi_ma > 60:
            self.ma_trend = 1
        elif self.rsi_ma < 40:
            self.ma_trend = -1
        else:
            self.ma_trend = 0
        
        if self.rsi_ma > 60 or self.rsi_ma < 40: 
            self.rsi_long = max(self.rsi_ma - self.rsi_signal, 0)
            self.rsi_short = min(self.rsi_ma + self.rsi_signal, 100)
        
        self.put_event()
        # self.cancel_all()

        # self.am5.update_bar(bar)
        # if not self.am5.inited:
        #     return

        # if not self.ma_trend:
        #     return

        # self.rsi_value = self.am5.rsi(self.rsi_window)

        # if self.pos == 0:
        #     if self.ma_trend > 0 and self.rsi_value <= self.rsi_long:
        #         self.buy(bar.close_price + 5, self.fixed_size)
        #     elif self.ma_trend < 0 and self.rsi_value >= self.rsi_short:
        #         self.short(bar.close_price - 5, self.fixed_size)

        # elif self.pos > 0:
        #     if self.ma_trend < 0 or self.rsi_value > 50:
        #         self.sell(bar.close_price - 5, abs(self.pos))

        # elif self.pos < 0:
        #     if self.ma_trend > 0 or self.rsi_value < 50:
        #         self.cover(bar.close_price + 5, abs(self.pos))

        # self.put_event()

    def on_15min_bar(self, bar: BarData):
        """"""
        pass
        # self.am15.update_bar(bar)
        # if not self.am15.inited:
        #     return

        # self.fast_ma = self.am15.sma(self.fast_window)
        # self.slow_ma = self.am15.sma(self.slow_window)

        # if self.fast_ma > self.slow_ma:
        #     self.ma_trend = 1
        # else:
        #     self.ma_trend = -1

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        pass

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        self.traded_price = trade.price
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass
    
    def stop_loss(self,bar:BarData):
        close_position = False
        ref_price = bar.close_price
        sl_price = min(self.atr * self.atr_tol, self.stop_loss_limit)
        if self.pos > 0:
            if (ref_price - self.traded_price) < - sl_price:
                close_position = True
        elif self.pos <0:
            if (self.traded_price - ref_price) < - sl_price:
                close_position = True
        else:
            pass
        return close_position
    
    def exit_gain(self,bar:BarData):
        exit_position = False
        current_ref_price = self.am1.sma(3)
        exit_signal_price = self.am1.sma(10)
        eg_price = self.atr * self.atr_tol * self.win_ratio
        
        if self.pos > 0:
            if self.pre_ref_price > exit_signal_price and current_ref_price < exit_signal_price:
                exit_position = True
            if self.rsi_value > self.rsi_short:
                exit_position = True
            if self.pre_ref_price - current_ref_price > eg_price:
                exit_position = True
            if self.k_value < self.d_value:
                exit_position = True
        elif self.pos < 0:
            if self.pre_ref_price < exit_signal_price and current_ref_price > exit_signal_price:
                exit_position = True
            if self.rsi_value < self.rsi_long:
                exit_position = True
            if current_ref_price - self.pre_ref_price > eg_price:
                exit_position = True
            if self.k_value > self.d_value:
                exit_position = True
        else:
            pass
        
        #sl_price = min(self.atr, self.stop_loss_limit)
        #if floating_pnl > sl_price * 2:
        #   exit_position = True
            
        self.pre_ref_price = current_ref_price
        return exit_position
    
    def open_position(self):
        direction = 0
        
        #set up kd watch indicator
        if self.ma_trend >= 0 and self.rsi_value < self.rsi_long:
            self.watch_kd = True
        elif self.ma_trend <= 0 and self.rsi_value > self.rsi_short:
            self.watch_kd = True
        
        #long short signal
        if self.watch_kd:
            if self.ma_trend >= 0 and self.k_value > self.d_value:
                direction = 1
                self.watch_kd = False
            elif self.ma_trend <= 0 and self.k_value < self.d_value:
                direction = -1
                self.watch_kd = False
            
        return direction


class RsiMAScalpStrategy(CtaTemplate):
    """"""
    author = "Xiaohui"

    rsi_signal = 20
    rsi_fast_window = 7
    rsi_slow_window = 20
    ma_window = 3
    fixed_size = 1
    captial_alloc = 100000
    size = 200
    pricetick = 0.2
    risk_tol = 1
    atr_tol = 1
    win_ratio = 2

    rsi_value = 0
    rsi_long = 0
    rsi_short = 0
    rsi_ma = 0
    ma_price = 0
    ma_trend = 0
    stop_loss_limit = 0
    atr = 0
    sl_price = 0
    eg_price = 0
    n_atr_max = 0

    parameters = ["rsi_signal", "rsi_fast_window","rsi_slow_window","ma_window",
                  "fixed_size","captial_alloc",
                  "size", "pricetick","risk_tol","atr_tol","win_ratio"]

    variables = ["rsi_value", "rsi_long", "rsi_short","rsi_ma","ma_price", "ma_trend",
                 "stop_loss_limit","atr","sl_price","eg_price","n_atr_max"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.rsi_ma = 50
        self.rsi_long = self.rsi_ma - self.rsi_signal
        self.rsi_short = self.rsi_ma + self.rsi_signal
        self.rsi_arry = None
        self.am1 = ArrayManager()

        self.bg5 = BarGenerator(self.on_bar, 5, self.on_5min_bar)
        self.am5 = ArrayManager()

        self.bg15 = BarGenerator(self.on_bar, 15, self.on_15min_bar)
        self.am15 = ArrayManager()
        
        self.traded_price = 0
        self.stop_loss_limit = round_to((self.captial_alloc * self.risk_tol/100)/self.size, self.pricetick)
        self.pre_ref_price = 0
        self.rsi_view = 0

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.load_bar(10)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.sl_price = 0.0
        self.eg_price = 0.0
        self.n_atr_max = 0
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        self.bg5.update_tick(tick)

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.cancel_all()
        self.bg5.update_bar(bar)
        self.bg15.update_bar(bar)
        
        self.am1.update_bar(bar)
        
        if not self.am1.inited:
            return
        
        self.ma_price = self.am1.sma(self.ma_window)
        
        self.atr = self.am1.atr(self.rsi_slow_window)
        stop_loss_signal = self.stop_loss(bar)
        exit_gain_signal = self.exit_gain(bar)
        open_pos_signal = self.open_position(bar)
        
        #do not open position if atr is low
        if self.atr < bar.close_price * 0.0001:
            open_pos_signal = 0

        if self.pos == 0:
            if open_pos_signal > 0:
                self.buy(bar.close_price + 5 * self.pricetick, self.fixed_size)
            elif open_pos_signal < 0:
                self.short(bar.close_price - 5 * self.pricetick, self.fixed_size)
        
        elif self.pos > 0:
            if stop_loss_signal or exit_gain_signal:
                self.sell(bar.close_price - 5 * self.pricetick, abs(self.pos))

        elif self.pos < 0:
            if stop_loss_signal or exit_gain_signal:
                self.cover(bar.close_price + 5 * self.pricetick, abs(self.pos))
        
        self.put_event()

    def on_5min_bar(self, bar: BarData):
        """"""
        self.am5.update_bar(bar)
        
        if not self.am5.inited:
            return
        
        self.rsi_arry = self.am5.rsi(self.rsi_fast_window, True)
        self.rsi_value = self.rsi_arry[-1]
        rsi_ma_arry = talib.SMA(self.rsi_arry, self.rsi_slow_window)
        self.rsi_ma = rsi_ma_arry[-1]
        
        if self.rsi_ma > 60:
            self.ma_trend = 1
        elif self.rsi_ma < 40:
            self.ma_trend = -1
        else:
            self.ma_trend = 0
        
        if self.rsi_ma > 60 or self.rsi_ma < 40: 
            self.rsi_long = max(self.rsi_ma - self.rsi_signal, 0)
            self.rsi_short = min(self.rsi_ma + self.rsi_signal, 100)
        
        if self.ma_trend >= 0 and self.rsi_value < self.rsi_long:
            self.rsi_view = 1
        elif self.ma_trend <= 0 and self.rsi_value > self.rsi_short:
            self.rsi_view = -1
            
        self.put_event()
        # self.cancel_all()

        # self.am5.update_bar(bar)
        # if not self.am5.inited:
        #     return

        # if not self.ma_trend:
        #     return

        # self.rsi_value = self.am5.rsi(self.rsi_window)

        # if self.pos == 0:
        #     if self.ma_trend > 0 and self.rsi_value <= self.rsi_long:
        #         self.buy(bar.close_price + 5, self.fixed_size)
        #     elif self.ma_trend < 0 and self.rsi_value >= self.rsi_short:
        #         self.short(bar.close_price - 5, self.fixed_size)

        # elif self.pos > 0:
        #     if self.ma_trend < 0 or self.rsi_value > 50:
        #         self.sell(bar.close_price - 5, abs(self.pos))

        # elif self.pos < 0:
        #     if self.ma_trend > 0 or self.rsi_value < 50:
        #         self.cover(bar.close_price + 5, abs(self.pos))

        # self.put_event()

    def on_15min_bar(self, bar: BarData):
        """"""
        pass
        # self.am15.update_bar(bar)
        # if not self.am15.inited:
        #     return

        # self.fast_ma = self.am15.sma(self.fast_window)
        # self.slow_ma = self.am15.sma(self.slow_window)

        # if self.fast_ma > self.slow_ma:
        #     self.ma_trend = 1
        # else:
        #     self.ma_trend = -1

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        pass

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        self.traded_price = trade.price
        pnl_tol = min(self.atr * self.atr_tol, self.stop_loss_limit)
        pnl_tar = self.atr * self.atr_tol * self.win_ratio
        
        #initiate stop_price and exit_price
        if trade.offset == Offset.OPEN:
            if trade.direction == Direction.LONG:
                self.sl_price = round_to(self.traded_price - pnl_tol, self.pricetick)
                self.eg_price = round_to(self.traded_price + pnl_tar, self.pricetick)
            elif trade.direction == Direction.SHORT:
                self.sl_price = round_to(self.traded_price + pnl_tol, self.pricetick)
                self.eg_price = round_to(self.traded_price - pnl_tar, self.pricetick)
            else:
                pass
            self.n_atr_max = 0
            
        if trade.offset == Offset.CLOSE or trade.offset == Offset.CLOSETODAY or trade.offset == Offset.CLOSEYESTERDAY:
            self.sl_price = 0.0
            self.eg_price = 0.0
    
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass
    
    def stop_loss(self,bar:BarData):
        close_position = False
        ref_price = bar.close_price
        
        self.reset_sl_price(bar)
        
        if self.sl_price <= 1.0:
            return False
        
        if self.pos > 0:
            if ref_price < self.sl_price:
                close_position = True
        elif self.pos < 0:
            if ref_price > self.sl_price:
                close_position = True
        else:
            pass
            
        return close_position
    
    def exit_gain(self,bar:BarData):
        exit_position = False
        
        if self.pos > 0:
            if self.rsi_value > self.rsi_short:
                exit_position = True
            if bar.close_price > self.eg_price:
                exit_position = True
        elif self.pos < 0:
            if self.rsi_value < self.rsi_long:
                exit_position = True
            if bar.close_price < self.eg_price:
                exit_position = True
        else:
            pass
        
        return exit_position
    
    def open_position(self,bar:BarData):
        direction = 0
        
        #long short signal
        if self.rsi_view > 0:
            if bar.close_price > self.ma_price:
                direction = 1
                self.rsi_view = 0
        
        if self.rsi_view < 0:
            if bar.close_price < self.ma_price:
                direction = -1
                self.rsi_view = 0
            
        return direction
    
    def reset_sl_price(self, bar: BarData):
        n_atr_curr = 0
        if self.pos > 0:
            n_atr_curr = np.floor(max(bar.close_price - self.traded_price, 0)/self.atr)
            self.n_atr_max = max(n_atr_curr, self.n_atr_max)
            self.sl_price = self.traded_price + (self.n_atr_max - self.atr_tol) * self.atr
        elif self.pos < 0:
            n_atr_curr = np.floor(max(self.traded_price - bar.close_price, 0)/self.atr)
            self.n_atr_max = max(n_atr_curr, self.n_atr_max)
            self.sl_price = self.traded_price - (self.n_atr_max - self.atr_tol) * self.atr
        else:
            pass
        self.sl_price = round_to(self.sl_price, self.pricetick)
        
        self.put_event()
