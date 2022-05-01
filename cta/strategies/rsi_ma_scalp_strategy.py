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

from .cutility import logger, CoolDownTimer, SignalArrayManager
from datetime import datetime 
import talib
import numpy as np

class RsiMAScalpStrategy(CtaTemplate):
    """"""
    author = "Xiaohui"

    rsi_signal = 20
    rsi_fast_window = 6
    rsi_slow_window = 20
    dynamic_rsi = 0
    trend_filter = 0
    ma_window = 3
    fixed_size = 1
    captial_alloc = 100000
    size = 10
    pricetick = 1.0
    risk_tol = 1.0
    atr_multi = 1
    win_ratio = 2
    atr_th = 0.1
    mngr_discretion = 0

    rsi_value = 0
    rsi_long = 0
    rsi_short = 0
    rsi_ma = 0
    trend = 0
    stop_loss_limit = 0
    atr = 0
    cd = ''
    sl_price = 0
    eg_price = 0
    n_atr_max = 0

    parameters = ["rsi_signal", "rsi_fast_window","rsi_slow_window",
                  "dynamic_rsi","trend_filter","ma_window",
                  "fixed_size","captial_alloc",
                  "size", "pricetick","risk_tol","atr_multi","atr_th","win_ratio", "mngr_discretion"]

    variables = ["rsi_value", "rsi_long", "rsi_short","rsi_ma",
                 "trend","stop_loss_limit","atr", "cd", "sl_price", "eg_price", "n_atr_max"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.rsi_ma = 50
        self.rsi_long = self.rsi_ma - self.rsi_signal
        self.rsi_short = self.rsi_ma + self.rsi_signal
        self.rsi_arry = None
        self.p_am = SignalArrayManager(180)
        self.am1 = ArrayManager()

        self.bg5 = BarGenerator(self.on_bar, 5, self.on_5min_bar)
        self.am5 = ArrayManager()

        self.bg15 = BarGenerator(self.on_bar, 15, self.on_15min_bar)
        self.am15 = ArrayManager()
        
        self.traded_price = 0
        self.stop_loss_limit = round_to((self.captial_alloc * self.risk_tol/100)/self.size, self.pricetick)
        self.pre_ref_price = 0
        self.direction = 0
        self.pos_lock = 0
        
        self.cooldown = CoolDownTimer(5)
        self.log_local = logger(app_name=vt_symbol, log_file_name=strategy_name + '.log')

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.cd = ''
        self.n_atr_max = 0
        self.write_log("策略初始化")
        self.load_bar(10)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.pos_lock = self.pos
        self.trend = self.mngr_discretion
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.cd = ''
        self.sl_price = 0.0
        self.eg_price = 0.0
        self.n_atr_max = 0
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        self.bg5.update_tick(tick)
        
        #update last tick price
        self.p_am.update_signal(tick.last_price)
        
        open_pos_signal = self.open_position()
        
        #do not open position if atr is low
        if self.atr < tick.last_price * self.atr_th/100:
            open_pos_signal = 0
        
        if self.pos != self.pos_lock:
            self.put_event()
            return

        if self.pos == 0 and not self.cooldown.initiated:
            if open_pos_signal > 0:
                self.buy(tick.last_price + 5 * self.pricetick, self.fixed_size)
                self.pos_lock += self.fixed_size
                self.write_params_log()
            elif open_pos_signal < 0:
                self.short(tick.last_price - 5 * self.pricetick, self.fixed_size)
                self.pos_lock -= self.fixed_size
                self.write_params_log()
        
        elif self.pos > 0:
            if self.stop_loss(tick) or self.exit_gain(tick):
                self.sell(tick.last_price - 5 * self.pricetick, abs(self.pos))
                self.pos_lock -= abs(self.pos)
                self.write_params_log()

        elif self.pos < 0:
            if self.stop_loss(tick) or self.exit_gain(tick):
                self.cover(tick.last_price + 5 * self.pricetick, abs(self.pos))
                self.pos_lock += abs(self.pos)
                self.write_params_log()
        
        self.put_event()

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
        
        if self.cooldown.initiated:
            self.cooldown.update_bartime(bar.datetime)
            self.cd = self.cooldown.time_left()
        
        self.rsi_arry = self.am1.rsi(self.rsi_fast_window, True)
        self.rsi_value = self.rsi_arry[-1]
        self.rsi_value = round(self.rsi_value,2)
        self.update_view()
        
        rsi_ma_arry = talib.SMA(self.rsi_arry, self.rsi_slow_window)
        self.rsi_ma = rsi_ma_arry[-1]
        self.rsi_ma = round(self.rsi_ma,2)
        
        self.atr = self.am1.atr(self.rsi_slow_window)
        self.atr = round(self.atr,3)
        
        self.put_event()

    def on_5min_bar(self, bar: BarData):
        """"""
        if  not self.am1.inited:
            return
        
        if self.mngr_discretion > -1 and self.mngr_discretion < 1:      
            if self.rsi_ma > 60:
                self.trend = 1
            elif self.rsi_ma < 40:
                self.trend = -1
            else:
                self.trend = 0
        
        if self.dynamic_rsi < 1:
            self.put_event()
            return
        
        if self.rsi_ma > 60: 
            self.rsi_long = 60 - self.rsi_signal
            self.rsi_short = 60 + self.rsi_signal
        elif self.rsi_ma < 40:
            self.rsi_long = 40 - self.rsi_signal
            self.rsi_short = 40 + self.rsi_signal
        else:
            self.rsi_long = 50 - self.rsi_signal
            self.rsi_short = 50 + self.rsi_signal
        
        self.put_event()

    def on_15min_bar(self, bar: BarData):
        """"""
        pass

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        self.put_event()

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        self.traded_price = trade.price
        self.write_log("orderid:{0};tradeid:{1};direction:{2};offset:{3};volume:{4};traded_price:{5}"
                .format(trade.orderid, trade.tradeid, trade.direction, trade.offset, trade.volume, trade.price))
        
        pnl_tol = min(self.atr * self.atr_multi, self.stop_loss_limit)
        pnl_tar = self.atr * self.atr_multi * self.win_ratio
        
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
    
    def stop_loss(self, tick: TickData):
        close_position = False
        ref_price = tick.last_price
        
        if self.sl_price <= 1.0:
            return False
        
        if self.pos > 0:
            if ref_price < self.sl_price:
                close_position = True
        elif self.pos <0:
            if ref_price > self.sl_price:
                close_position = True
        else:
            pass
        
        if close_position:
            self.cooldown.start(tick.datetime)
            
        return close_position
    
    def exit_gain(self, tick: TickData):
        exit_position = False
        
        if self.eg_price <= 1.0:
            return False
        
        if self.pos > 0:
            if self.rsi_value > self.rsi_short or tick.last_price > self.eg_price:
                exit_position = True
        elif self.pos < 0:
            if self.rsi_value < self.rsi_long or tick.last_price < self.eg_price:
                exit_position = True
        else:
            pass
        
        return exit_position
    
    def open_position(self):
        ls_signal = 0
        
        if not self.p_am.inited:
            return ls_signal
        
        sma = self.p_am.sma(self.ma_window, array= True)
        
        #long short signal(statistical)
        #lb = 120 #lookback 120 ticks
        #cl = 0.6 # confidence level 60%
        
        # if self.direction > 0 and self.p_am.greater_than(sma, lb, cl):
        #     ls_signal = 1
        #     #swith off direction signal to avoid multiple position open with in 1 min
        #     self.direction = 0
        # elif self.direction < 0 and self.p_am.less_than(sma, lb, cl):
        #     ls_signal = -1
        #     #swith off direction signal to avoid multiple position open with in 1 min
        #     self.direction = 0
        
        #long short signal(simple)
        if self.direction > 0 and self.p_am.signal_array[-1] > sma[-1]:
            ls_signal = 1
            #swith off direction signal to avoid multiple position open with in 1 min
            self.direction = 0
        elif self.direction < 0 and self.p_am.signal_array[-1] < sma[-1]:
            ls_signal = -1
            #swith off direction signal to avoid multiple position open with in 1 min
            self.direction = 0
            
        return ls_signal
    
    def update_view(self):
        
        if self.trend_filter > 0:
            #set up price watch indicator(with trend filter)
            if self.trend >= 0 and self.rsi_value < self.rsi_long:
                self.direction = 1
            elif self.trend <= 0 and self.rsi_value > self.rsi_short:
                self.direction = -1
            else:
                self.direction = 0
        else:
            #set up price watch indicator(simple)
            if self.rsi_value < self.rsi_long:
                self.direction = 1
            elif self.rsi_value > self.rsi_short:
                self.direction = -1
            else:
                self.direction = 0
    
    def write_log(self, msg: str):
        self.log_local.info(msg)
        return super().write_log(msg)
    
    def write_params_log(self):
        self.write_log("trend:{0};rsi_current:{1};rsi_ma:{2};rsi_long:{3};rsi_short:{4};pos:{5}"
        .format(self.trend, self.rsi_value, self.rsi_ma, self.rsi_long, self.rsi_short, self.pos))

class RsiMAScalpQuickStrategy(RsiMAScalpStrategy):
    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        
    def stop_loss(self,tick: TickData):
        close_position = False
        ref_price = tick.last_price
        
        self.reset_sl_price(tick)
        
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
        
        if close_position:
            self.cooldown.start(tick.datetime)
            
        return close_position
    
    def reset_sl_price(self, tick: TickData):
        n_atr_curr = 0
        if self.pos > 0:
            n_atr_curr = np.floor(max(tick.last_price - self.traded_price, 0)/self.atr)
            self.n_atr_max = max(n_atr_curr, self.n_atr_max)
            self.sl_price = self.traded_price + (self.n_atr_max - self.atr_multi) * self.atr
        elif self.pos < 0:
            n_atr_curr = np.floor(max(self.traded_price - tick.last_price, 0)/self.atr)
            self.n_atr_max = max(n_atr_curr, self.n_atr_max)
            self.sl_price = self.traded_price - (self.n_atr_max - self.atr_multi) * self.atr
        else:
            pass
        self.sl_price = round_to(self.sl_price, self.pricetick)
        
        self.put_event()
      

class RsiMAScalpSlowStrategy(RsiMAScalpStrategy):
    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        
    def stop_loss(self,tick: TickData):
        close_position = False
        ref_price = tick.last_price
        
        self.reset_sl_price(tick)
        
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
        
        if close_position:
            self.cooldown.start(tick.datetime)
            
        return close_position
    
    def exit_gain(self,tick: TickData):
        exit_position = False
        
        self.reset_eg_price()
        
        if self.eg_price <= 1.0:
            return False
        
        if self.pos > 0:
            if tick.last_price > self.eg_price:
                exit_position = True
        elif self.pos < 0:
            if tick.last_price < self.eg_price:
                exit_position = True
        else:
            pass
        
        return exit_position
    
    def reset_sl_price(self, tick: TickData):
        n_atr_curr = 0
        if self.pos > 0:
            n_atr_curr = np.floor(max(tick.last_price - self.traded_price, 0)/self.atr)
            self.n_atr_max = max(n_atr_curr, self.n_atr_max)
            self.sl_price = self.traded_price + self.calc_sl_factor() * self.atr
        elif self.pos < 0:
            n_atr_curr = np.floor(max(self.traded_price - tick.last_price, 0)/self.atr)
            self.n_atr_max = max(n_atr_curr, self.n_atr_max)
            self.sl_price = self.traded_price - self.calc_sl_factor() * self.atr
        else:
            pass
        self.sl_price = round_to(self.sl_price, self.pricetick)
        
        self.put_event()
        
    def reset_eg_price(self, bar:BarData):
        self.put_event()
    
    def calc_sl_factor(self):
        f = 0.0
        atr_range = self.win_ratio * self.atr_multi
        m1_atr = np.floor(atr_range /4)
        m2_atr = np.floor(atr_range / 2)
        
        if self.n_atr_max <= self.atr_multi:
            f = self.n_atr_max - self.atr_multi
        elif self.n_atr_max > self.atr_multi and self.n_atr_max <= m1_atr:
            f = 0.0
        elif self.n_atr_max > m1_atr and self.n_atr_max <= m2_atr:
            f = 1.0
        elif self.n_atr_max > m2_atr and self.n_atr_max <= atr_range:
            f = self.n_atr_max / 2
        else:
            f = self.n_atr_max - 1
        return f