from collections import defaultdict
from datetime import date, datetime
from typing import Callable, Type, Dict, List
from functools import partial

import numpy as np
import multiprocessing
from pandas import DataFrame
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vnpy.trader.constant import (Direction, Offset, Exchange,
                                  Interval, Status)
from vnpy.trader.object import TradeData, BarData, TickData

from vnpy_spreadtrading.template import SpreadStrategyTemplate, SpreadAlgoTemplate
from vnpy_spreadtrading.base import (SpreadData,
                                     LegData,
                                     BacktestingMode,
                                     load_tick_data)
from vnpy.trader.utility import round_to, floor_to, ceil_to, extract_vt_symbol
from vnpy.trader.optimize import (OptimizationSetting, 
                                  check_optimization_setting, 
                                  run_bf_optimization, 
                                  run_ga_optimization)

from .cutility import load_bar_data

class SpreadBacktestEngine:
    """"""

    gateway_name = "BACKTESTING"

    def __init__(self):
        """"""
        self.spread: SpreadData = None
        self.spread_setting={}
        
        self.start = None
        self.end = None
        self.rate = 0
        self.slippage = 0
        self.size = 1
        self.pricetick = 0
        self.capital = 1_000_000
        self.mode = BacktestingMode.BAR

        self.strategy_class: Type[SpreadStrategyTemplate] = None
        self.strategy: SpreadStrategyTemplate = None
        self.tick: TickData = None
        self.bar: BarData = None
        self.datetime = None

        self.interval = None
        self.days = 0
        self.callback = None
        self.history_data = []

        self.algo_count = 0
        self.algos = {}
        self.active_algos = {}

        self.trade_count = 0
        self.trades = {}

        self.logs = []

        self.daily_results = {}
        self.daily_df = None

    def output(self, msg):
        """
        Output message of backtesting engine.
        """
        print(f"{datetime.now()}\t{msg}")

    def clear_data(self):
        """
        Clear all data of last backtesting.
        """
        self.strategy = None
        self.tick = None
        self.bar = None
        self.datetime = None

        self.algo_count = 0
        self.algos.clear()
        self.active_algos.clear()

        self.trade_count = 0
        self.trades.clear()

        self.logs.clear()
        self.daily_results.clear()

    def set_parameters(
        self,
        interval: Interval,
        start: datetime,
        rate: float,
        slippage: float,
        size: float,
        pricetick: float,
        capital: int = 0,
        end: datetime = None,
        mode: BacktestingMode = BacktestingMode.BAR,
        spread: SpreadData = None
    ):
        """"""
        self.spread = spread
        self.interval = Interval(interval)
        self.rate = rate
        self.slippage = slippage
        self.size = size
        self.pricetick = pricetick
        self.start = start
        self.capital = capital
        self.end = end
        self.mode = mode
    
    def add_spread(self, spread_setting: dict):
        var_list = [elem['variable'] for elem in spread_setting['leg_settings']]
        vt_symbol_list = [elem['vt_symbol'] for elem in spread_setting['leg_settings']]
        direction_list = [elem['trading_direction'] for elem in spread_setting['leg_settings']]
        multi_list = [elem['trading_multiplier'] for elem in spread_setting['leg_settings']]
        var_dict = dict(zip(var_list, vt_symbol_list))
        var_direct_dict = dict(zip(var_list, direction_list))
        trade_multi_dict = dict(zip(vt_symbol_list, multi_list))
        legs_list = list([LegData(elem['vt_symbol']) for elem in spread_setting['leg_settings']])
        self.spread = SpreadData(spread_setting['name'],
                                 legs= legs_list,
                                 variable_symbols= var_dict,
                                 variable_directions=var_direct_dict,
                                 price_formula=spread_setting['price_formula'],
                                 trading_multipliers=trade_multi_dict,
                                 active_symbol=spread_setting['active_symbol'],
                                 min_volume= spread_setting['min_volume'])
        self.spread_setting = spread_setting
    
    def add_strategy(self, strategy_class: type, setting: dict):
        """"""
        self.strategy_class = strategy_class

        self.strategy = strategy_class(
            self,
            strategy_class.__name__,
            self.spread,
            setting
        )

    def load_data(self):
        """"""
        self.output("开始加载历史数据")

        if not self.end:
            self.end = datetime.now()

        if self.start >= self.end:
            self.output("起始日期必须小于结束日期")
            return

        if self.mode == BacktestingMode.BAR:
            self.history_data = load_bar_data(
                self.spread,
                self.interval,
                self.start,
                self.end,
                self.pricetick
            )
        else:
            self.history_data = load_tick_data(
                self.spread,
                self.start,
                self.end
            )

        self.output(f"历史数据加载完成，数据量：{len(self.history_data)}")

    def run_backtesting(self):
        """"""
        if self.mode == BacktestingMode.BAR:
            func = self.new_bar
        else:
            func = self.new_tick

        self.strategy.on_init()

        # Use the first [days] of history data for initializing strategy
        day_count = 0
        ix = 0

        for ix, data in enumerate(self.history_data):
            if self.datetime and data.datetime.day != self.datetime.day:
                day_count += 1
                if day_count >= self.days:
                    break

            self.datetime = data.datetime
            self.callback(data)

        self.strategy.inited = True
        self.output("策略初始化完成")

        self.strategy.on_start()
        self.strategy.trading = True
        self.output("开始回放历史数据")

        # Use the rest of history data for running backtesting
        for data in self.history_data[ix:]:
            func(data)

        self.output("历史数据回放结束")

    def calculate_result(self):
        """"""
        self.output("开始计算逐日盯市盈亏")

        if not self.trades:
            self.output("成交记录为空，无法计算")
            return

        # Add trade data into daily reuslt.
        for trade in self.trades.values():
            d = trade.datetime.date()
            daily_result = self.daily_results[d]
            daily_result.add_trade(trade)

        # Calculate daily result by iteration.
        pre_close = 0
        start_pos = 0

        for daily_result in self.daily_results.values():
            daily_result.calculate_pnl(
                pre_close,
                start_pos,
                self.size,
                self.rate,
                self.slippage
            )

            pre_close = daily_result.close_price
            start_pos = daily_result.end_pos

        # Generate dataframe
        results = defaultdict(list)

        for daily_result in self.daily_results.values():
            for key, value in daily_result.__dict__.items():
                results[key].append(value)

        self.daily_df = DataFrame.from_dict(results).set_index("date")

        self.output("逐日盯市盈亏计算完成")
        return self.daily_df

    def calculate_statistics(self, df: DataFrame = None, output=True):
        """"""
        self.output("开始计算策略统计指标")

        # Check DataFrame input exterior
        if df is None:
            df = self.daily_df

        # Check for init DataFrame
        if df is None:
            # Set all statistics to 0 if no trade.
            start_date = ""
            end_date = ""
            total_days = 0
            profit_days = 0
            loss_days = 0
            end_balance = 0
            max_drawdown = 0
            max_ddpercent = 0
            max_drawdown_duration = 0
            total_net_pnl = 0
            daily_net_pnl = 0
            total_commission = 0
            daily_commission = 0
            total_slippage = 0
            daily_slippage = 0
            total_turnover = 0
            daily_turnover = 0
            total_trade_count = 0
            daily_trade_count = 0
            total_return = 0
            annual_return = 0
            daily_return = 0
            return_std = 0
            sharpe_ratio = 0
            return_drawdown_ratio = 0
        else:
            # Calculate balance related time series data
            df["balance"] = df["net_pnl"].cumsum() + self.capital
            df["return"] = np.log(df["balance"] / df["balance"].shift(1)).fillna(0)
            df["highlevel"] = (
                df["balance"].rolling(
                    min_periods=1, window=len(df), center=False).max()
            )
            df["drawdown"] = df["balance"] - df["highlevel"]
            df["ddpercent"] = df["drawdown"] / df["highlevel"] * 100

            # Calculate statistics value
            start_date = df.index[0]
            end_date = df.index[-1]

            total_days = len(df)
            profit_days = len(df[df["net_pnl"] > 0])
            loss_days = len(df[df["net_pnl"] < 0])

            end_balance = df["balance"].iloc[-1]
            max_drawdown = df["drawdown"].min()
            max_ddpercent = df["ddpercent"].min()
            max_drawdown_end = df["drawdown"].idxmin()
            max_drawdown_start = df["balance"][:max_drawdown_end].idxmax()
            max_drawdown_duration = (max_drawdown_end - max_drawdown_start).days

            total_net_pnl = df["net_pnl"].sum()
            daily_net_pnl = total_net_pnl / total_days

            total_commission = df["commission"].sum()
            daily_commission = total_commission / total_days

            total_slippage = df["slippage"].sum()
            daily_slippage = total_slippage / total_days

            total_turnover = df["turnover"].sum()
            daily_turnover = total_turnover / total_days

            total_trade_count = df["trade_count"].sum()
            daily_trade_count = total_trade_count / total_days

            total_return = (end_balance / self.capital - 1) * 100
            annual_return = total_return / total_days * 240
            daily_return = df["return"].mean() * 100
            return_std = df["return"].std() * 100

            if return_std:
                sharpe_ratio = daily_return / return_std * np.sqrt(240)
            else:
                sharpe_ratio = 0

            return_drawdown_ratio = -total_return / max_ddpercent

        # Output
        if output:
            self.output("-" * 30)
            self.output(f"首个交易日：\t{start_date}")
            self.output(f"最后交易日：\t{end_date}")

            self.output(f"总交易日：\t{total_days}")
            self.output(f"盈利交易日：\t{profit_days}")
            self.output(f"亏损交易日：\t{loss_days}")

            self.output(f"起始资金：\t{self.capital:,.2f}")
            self.output(f"结束资金：\t{end_balance:,.2f}")

            self.output(f"总收益率：\t{total_return:,.2f}%")
            self.output(f"年化收益：\t{annual_return:,.2f}%")
            self.output(f"最大回撤: \t{max_drawdown:,.2f}")
            self.output(f"百分比最大回撤: {max_ddpercent:,.2f}%")
            self.output(f"最长回撤天数: \t{max_drawdown_duration}")

            self.output(f"总盈亏：\t{total_net_pnl:,.2f}")
            self.output(f"总手续费：\t{total_commission:,.2f}")
            self.output(f"总滑点：\t{total_slippage:,.2f}")
            self.output(f"总成交金额：\t{total_turnover:,.2f}")
            self.output(f"总成交笔数：\t{total_trade_count}")

            self.output(f"日均盈亏：\t{daily_net_pnl:,.2f}")
            self.output(f"日均手续费：\t{daily_commission:,.2f}")
            self.output(f"日均滑点：\t{daily_slippage:,.2f}")
            self.output(f"日均成交金额：\t{daily_turnover:,.2f}")
            self.output(f"日均成交笔数：\t{daily_trade_count}")

            self.output(f"日均收益率：\t{daily_return:,.2f}%")
            self.output(f"收益标准差：\t{return_std:,.2f}%")
            self.output(f"Sharpe Ratio：\t{sharpe_ratio:,.2f}")
            self.output(f"收益回撤比：\t{return_drawdown_ratio:,.2f}")

        statistics = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "max_drawdown_duration": max_drawdown_duration,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "daily_commission": daily_commission,
            "total_slippage": total_slippage,
            "daily_slippage": daily_slippage,
            "total_turnover": total_turnover,
            "daily_turnover": daily_turnover,
            "total_trade_count": total_trade_count,
            "daily_trade_count": daily_trade_count,
            "total_return": total_return,
            "annual_return": annual_return,
            "daily_return": daily_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "return_drawdown_ratio": return_drawdown_ratio,
        }

        return statistics

    def show_chart(self, df: DataFrame = None):
        """"""
        # Check DataFrame input exterior
        if df is None:
            df = self.daily_df

        # Check for init DataFrame
        if df is None:
            return

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Balance", "Drawdown", "Daily Pnl", "Pnl Distribution"],
            vertical_spacing=0.06
        )

        balance_line = go.Scatter(
            x=df.index,
            y=df["balance"],
            mode="lines",
            name="Balance"
        )
        drawdown_scatter = go.Scatter(
            x=df.index,
            y=df["drawdown"],
            fillcolor="red",
            fill='tozeroy',
            mode="lines",
            name="Drawdown"
        )
        pnl_bar = go.Bar(y=df["net_pnl"], name="Daily Pnl")
        pnl_histogram = go.Histogram(x=df["net_pnl"], nbinsx=100, name="Days")

        fig.add_trace(balance_line, row=1, col=1)
        fig.add_trace(drawdown_scatter, row=2, col=1)
        fig.add_trace(pnl_bar, row=3, col=1)
        fig.add_trace(pnl_histogram, row=4, col=1)

        fig.update_layout(height=1000, width=1000)
        fig.show()

    def update_daily_close(self, price: float):
        """"""
        d = self.datetime.date()

        daily_result = self.daily_results.get(d, None)
        if daily_result:
            daily_result.close_price = price
        else:
            self.daily_results[d] = DailyResult(d, price)

    def new_bar(self, bar: BarData):
        """"""
        self.bar = bar
        self.datetime = bar.datetime
        self.cross_limit_algo()

        self.strategy.on_spread_bar(bar)

        self.update_daily_close(bar.close_price)

    def new_tick(self, tick: TickData):
        """"""
        self.tick = tick
        self.datetime = tick.datetime
        self.cross_algo()

        self.spread.bid_price = tick.bid_price_1
        self.spread.bid_volume = tick.bid_volume_1
        self.spread.ask_price = tick.ask_price_1
        self.spread.ask_volume = tick.ask_volume_1
        self.spread.datetime = tick.datetime

        self.strategy.on_spread_data()

        self.update_daily_close(tick.last_price)

    def cross_algo(self):
        """
        Cross limit order with last bar/tick data.
        """
        if self.mode == BacktestingMode.BAR:
            long_cross_price = self.bar.close_price
            short_cross_price = self.bar.close_price
        else:
            long_cross_price = self.tick.ask_price_1
            short_cross_price = self.tick.bid_price_1

        for algo in list(self.active_algos.values()):
            # Check whether limit orders can be filled.
            long_cross = (
                algo.direction == Direction.LONG
                and algo.price >= long_cross_price
            )

            short_cross = (
                algo.direction == Direction.SHORT
                and algo.price <= short_cross_price
            )

            if not long_cross and not short_cross:
                continue

            # Push order udpate with status "all traded" (filled).
            algo.traded = algo.volume
            algo.status = Status.ALLTRADED
            self.strategy.update_spread_algo(algo)

            self.active_algos.pop(algo.algoid)

            # Push trade update
            self.trade_count += 1

            if long_cross:
                trade_price = long_cross_price
                pos_change = algo.volume
            else:
                trade_price = short_cross_price
                pos_change = -algo.volume

            trade = TradeData(
                symbol=self.spread.name,
                exchange=Exchange.LOCAL,
                orderid=algo.algoid,
                tradeid=str(self.trade_count),
                direction=algo.direction,
                price=trade_price,
                volume=algo.volume,
                datetime=self.datetime,
                gateway_name=self.gateway_name,
            )

            if self.mode == BacktestingMode.BAR:
                trade.value = self.bar.value
            else:
                trade.value = trade_price
            
            self.strategy.on_trade(trade)
            self.spread.net_pos += pos_change
            self.strategy.on_spread_pos()

            self.trades[trade.vt_tradeid] = trade

    def cross_limit_algo(self):
            """
            Cross limit order with last bar/tick data.
            """
            if self.mode == BacktestingMode.BAR:
                long_cross_price = self.bar.low_price
                short_cross_price = self.bar.high_price
                long_best_price = self.bar.open_price
                short_best_price = self.bar.open_price
            else:
                long_cross_price = self.tick.ask_price_1
                short_cross_price = self.tick.bid_price_1
                long_best_price = long_cross_price
                short_best_price = short_cross_price

            for algo in list(self.active_algos.values()):
                # Check whether limit orders can be filled.
                long_cross = (
                    algo.direction == Direction.LONG
                    and algo.price >= long_cross_price
                )

                short_cross = (
                    algo.direction == Direction.SHORT
                    and algo.price <= short_cross_price
                )

                if not long_cross and not short_cross:
                    continue

                # Push order udpate with status "all traded" (filled).
                algo.traded = algo.volume
                algo.status = Status.ALLTRADED
                self.strategy.update_spread_algo(algo)

                self.active_algos.pop(algo.algoid)

                # Push trade update
                self.trade_count += 1

                if long_cross:
                    trade_price = min(floor_to(algo.price, self.pricetick), long_best_price)
                    pos_change = algo.volume
                else:
                    trade_price = max(ceil_to(algo.price, self.pricetick), short_best_price)
                    pos_change = -algo.volume

                trade = TradeData(
                    symbol=self.spread.name,
                    exchange=Exchange.LOCAL,
                    orderid=algo.algoid,
                    tradeid=str(self.trade_count),
                    direction=algo.direction,
                    price=trade_price,
                    volume=algo.volume,
                    datetime=self.datetime,
                    gateway_name=self.gateway_name,
                )

                if self.mode == BacktestingMode.BAR:
                    trade.value = self.bar.value
                else:
                    trade.value = trade_price

                self.strategy.on_trade(trade)
                self.spread.net_pos += pos_change
                self.strategy.on_spread_pos()

                self.trades[trade.vt_tradeid] = trade

    def load_bar(
        self, spread: SpreadData, days: int, interval: Interval, callback: Callable
    ):
        """"""
        self.days = days
        self.callback = callback

    def load_tick(self, spread: SpreadData, days: int, callback: Callable):
        """"""
        self.days = days
        self.callback = callback

    def start_algo(
        self,
        strategy: SpreadStrategyTemplate,
        spread_name: str,
        direction: Direction,
        price: float,
        volume: float,
        payup: int,
        interval: int,
        lock: bool,
        extra: dict
    ) -> str:
        """"""
        self.algo_count += 1
        algoid = str(self.algo_count)

        algo = SpreadAlgoTemplate(
            self,
            algoid,
            self.spread,
            direction,
            price,
            volume,
            payup,
            interval,
            lock,
            extra
        )

        self.algos[algoid] = algo
        self.active_algos[algoid] = algo

        return algoid

    def stop_algo(
        self,
        strategy: SpreadStrategyTemplate,
        algoid: str
    ):
        """"""
        if algoid not in self.active_algos:
            return
        algo = self.active_algos.pop(algoid)

        algo.status = Status.CANCELLED
        self.strategy.update_spread_algo(algo)

    def send_order(
        self,
        strategy: SpreadStrategyTemplate,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        stop: bool,
        lock: bool
    ):
        """"""
        pass

    def cancel_order(self, strategy: SpreadStrategyTemplate, vt_orderid: str):
        """
        Cancel order by vt_orderid.
        """
        pass

    def write_strategy_log(self, strategy: SpreadStrategyTemplate, msg: str):
        """
        Write log message.
        """
        msg = f"{self.datetime}\t{msg}"
        self.logs.append(msg)

    def send_email(self, msg: str, strategy: SpreadStrategyTemplate = None):
        """
        Send email to default receiver.
        """
        pass

    def put_strategy_event(self, strategy: SpreadStrategyTemplate):
        """
        Put an event to update strategy status.
        """
        pass

    def write_algo_log(self, algo: SpreadAlgoTemplate, msg: str):
        """"""
        pass
    
    def run_bf_optimization(self, optimization_setting: OptimizationSetting, output=True):
        """"""
        if not check_optimization_setting(optimization_setting):
            return

        evaluate_func: callable = wrap_evaluate(self, optimization_setting.target_name)
        results = run_bf_optimization(
            evaluate_func,
            optimization_setting,
            get_target_value,
            output=self.output
        )

        if output:
            for result in results:
                msg: str = f"参数：{result[0]}, 目标：{result[1]}"
                self.output(msg)

        return results

    run_optimization = run_bf_optimization

    def run_ga_optimization(self, optimization_setting: OptimizationSetting, output=True):
        """"""
        if not check_optimization_setting(optimization_setting):
            return

        evaluate_func: callable = wrap_evaluate(self, optimization_setting.target_name)
        results = run_ga_optimization(
            evaluate_func,
            optimization_setting,
            get_target_value,
            output=self.output
        )

        if output:
            for result in results:
                msg: str = f"参数：{result[0]}, 目标：{result[1]}"
                self.output(msg)

        return results

class DailyResult:
    """"""

    def __init__(self, date: date, close_price: float):
        """"""
        self.date = date
        self.close_price = close_price
        self.pre_close = 0

        self.trades = []
        self.trade_count = 0

        self.start_pos = 0
        self.end_pos = 0

        self.turnover = 0
        self.commission = 0
        self.slippage = 0

        self.trading_pnl = 0
        self.holding_pnl = 0
        self.total_pnl = 0
        self.net_pnl = 0

    def add_trade(self, trade: TradeData):
        """"""
        self.trades.append(trade)

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

            turnover = trade.volume * size * trade.value
            self.trading_pnl += pos_change * \
                (self.close_price - trade.price) * size
            self.slippage += trade.volume * size * slippage

            self.turnover += turnover
            self.commission += turnover * rate

        # Net pnl takes account of commission and slippage cost
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission - self.slippage

def evaluate(
    target_name: str,
    strategy_class: SpreadStrategyTemplate,
    spread_setting: dict,
    interval: Interval,
    start: datetime,
    rate: float,
    slippage: float,
    size: float,
    pricetick: float,
    capital: int,
    end: datetime,
    mode: BacktestingMode,
    setting: dict
):
    """
    Function for running in multiprocessing.pool
    """
    engine = SpreadBacktestEngine()
    engine.clear_data()
    engine.set_parameters(
        interval=interval,
        start=start,
        rate=rate,
        slippage=slippage,
        size=size,
        pricetick=pricetick,
        capital=capital,
        end=end,
        mode=mode
    )
    
    engine.add_spread(spread_setting)
    engine.add_strategy(strategy_class, setting)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    statistics = engine.calculate_statistics(output=False)

    target_value = statistics[target_name]
    return (str(setting), target_value, statistics)


def wrap_evaluate(engine: SpreadBacktestEngine, target_name: str) -> callable:
    """
    Wrap evaluate function with given setting from backtesting engine.
    """
    func: callable = partial(
        evaluate,
        target_name,
        engine.strategy_class,
        engine.spread_setting,
        engine.interval,
        engine.start,
        engine.rate,
        engine.slippage,
        engine.size,
        engine.pricetick,
        engine.capital,
        engine.end,
        engine.mode
    )
    return func


def get_target_value(result: list) -> float:
    """
    Get target value for sorting optimization results.
    """
    return result[1]

# @lru_cache(maxsize=999)
# def load_bar_data(
# spread: SpreadData,
# interval: Interval,
# start: datetime,
# end: datetime,
# pricetick: float = 0
# ):
# """"""
# database: BaseDatabase = get_database()

# # Load bar data of each spread leg
# leg_bars: Dict[str, Dict] = {}

# for vt_symbol in spread.legs.keys():
#     symbol, exchange = extract_vt_symbol(vt_symbol)

#     # First, try to query history from RQData
#     bar_data: List[BarData] = query_bar_from_rq(
#         symbol, exchange, interval, start, end
#     )

#     # If failed, query history from database
#     if not bar_data:
#         bar_data = database.load_bar_data(
#             symbol, exchange, interval, start, end
#         )

#     bars: Dict[datetime, BarData] = {bar.datetime: bar for bar in bar_data}
#     leg_bars[vt_symbol] = bars

# # Calculate spread bar data
# spread_bars: List[BarData] = []

# for dt in bars.keys():
#     spread_price = 0
#     spread_value = 0
#     spread_available = True

#     leg_data_close = {}
#     leg_data_open = {}
#     for variable, leg in spread.variable_legs.items():
#         leg_bar = leg_bars[leg.vt_symbol].get(dt, None)

#         if leg_bar:
#             # 缓存该腿当前的价格
#             leg_data_close[variable] = leg_bar.close_price
#             leg_data_open[variable] = leg_bar.open_price

#             # 基于交易乘数累计价值
#             trading_multiplier = spread.trading_multipliers[leg.vt_symbol]
#             spread_value += abs(trading_multiplier) * leg_bar.close_price
#         else:
#             spread_available = False

#     if spread_available:
#         spread_price_close = spread.parse_formula(spread.price_code, leg_data_close)
#         spread_price_open = spread.parse_formula(spread.price_code, leg_data_open)
#         if pricetick:
#             spread_price_close = round_to(spread_price_close, pricetick)
#             spread_price_open = round_to(spread_price_open, pricetick)

#         spread_bar = BarData(
#             symbol=spread.name,
#             exchange=exchange.LOCAL,
#             datetime=dt,
#             interval=interval,
#             open_price=spread_price_open,
#             high_price=max(spread_price_open,spread_price_close),
#             low_price=min(spread_price_open,spread_price_close),
#             close_price=spread_price_close,
#             gateway_name="SPREAD",
#         )
#         spread_bar.value = spread_value
#         spread_bars.append(spread_bar)

# return spread_bars