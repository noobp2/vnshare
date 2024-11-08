from collections import defaultdict
from datetime import date, datetime
from re import S
from typing import Callable, Type, Dict, List
from functools import partial

import numpy as np
import json
import os
from pandas import DataFrame
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vnpy.trader.constant import (Direction, Offset, Exchange,Interval, Status)
from vnpy.trader.object import TradeData, BarData, TickData
from vnpy.trader.utility import round_to, floor_to, ceil_to, extract_vt_symbol

from .tutils import CDailyResult,load_bar_data,read_trans_from_log,load_json_file
from vnpy_spreadtrading.template import SpreadStrategyTemplate
from vnpy_spreadtrading.base import (SpreadData,
                                     LegData,
                                     BacktestingMode,
                                     load_tick_data)

class SpreadPerformanceEngine:
    """"""

    gateway_name = "PERFORMANCE"

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
        
        self.strategy_name: str = ''
        self.strategy_base_path: str = ''
        self.exchange = Exchange.LOCAL
        self.spread_name = ''
        
        self.bar: BarData = None
        self.datetime = None

        self.interval = None
        self.days = 0
        self.callback = None
        self.history_data = []

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

        self.trade_count = 0
        self.trades.clear()

        self.logs.clear()
        self.daily_results.clear()

    def set_parameters(
        self,
        strategy_name : str,
        interval: Interval,
        start: datetime,
        rate: float,
        slippage: float,
        size: float,
        pricetick: float,
        capital: int = 0,
        end: datetime = None,
        mode: BacktestingMode = BacktestingMode.BAR,
        spread: SpreadData = None,
        strategy_base_path : str = Path.cwd()
    ):
        """"""
        self.strategy_name = strategy_name
        self.strategy_base_path = strategy_base_path
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
        self.spread_name = self.get_spread_name()
        self.set_spread()
    
    def set_spread(self):
        spread_setting_path = os.path.join(self.strategy_base_path,'.vntrader','spread_trading_setting.json')
        sprd_settings = load_json_file(spread_setting_path)
        sprd_setting = [d for d in sprd_settings if d['name'] == self.spread_name]
        sprd_setting_dict = sprd_setting[0]
        self.add_spread(sprd_setting_dict)
        
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
    
    def load_trades(self):
        log_file = self.strategy_name + '.log'
        log_file_path = os.path.join(self.strategy_base_path,log_file)
        kp_phrases = ["algoid"]
        trans = read_trans_from_log(log_file_path,kp_phrases)
        #get datetime
        trans_datetime=[t.split(' - ')[0] for t in trans]
        #get algos arrys
        trans_algos=[t.split(' - ')[3] for t in trans]
        trans_algoids = [a.split(';')[0].split(':')[1] for a in trans_algos]
        trans_directions = [a.split(';')[1].split(':')[1].split('.')[1] for a in trans_algos]
        trans_prices = [float(a.split(';')[2].split(':')[1]) for a in trans_algos]
        trans_volumes = [float(a.split(';')[3].split(':')[1]) for a in trans_algos]
        trans_status = [a.split(';')[4].split(':')[1].split('.')[1] for a in trans_algos]
        trans_trade_volumes = [float(a.split(';')[5].split(':')[1]) for a in trans_algos]
        trans_trade_prices = [float(a.split(';')[6].split(':')[1]) for a in trans_algos]
        #build dataframe
        trans_df = DataFrame()
        trans_df['datetime'] = trans_datetime
        trans_df['algoid'] = trans_algoids
        trans_df['direction'] = trans_directions
        trans_df['price'] = trans_prices
        trans_df['volume'] = trans_volumes
        trans_df['status'] = trans_status
        trans_df['traded_volume'] = trans_trade_volumes
        trans_df['traded_price'] = trans_trade_prices
        
        for idx,row in trans_df.iterrows():
            trade = TradeData(
                        symbol=self.spread_name,
                        exchange=self.exchange,
                        orderid=row['algoid'],
                        tradeid=str(idx),
                        direction=Direction[row['direction']],
                        price=row['traded_price'],
                        volume=row['traded_volume'],
                        datetime=datetime.fromisoformat(row['datetime'].replace(',','.')),
                        gateway_name=self.gateway_name,
                    )
            self.trades[trade.vt_tradeid] = trade
        self.trade_count = len(self.trades)
        self.output(f"trades loaded：{self.trade_count}")
    
    def get_trades_df(self):
        gwy = [val.gateway_name for (key, val) in self.trades.items()]
        symbol = [val.symbol for (key, val) in self.trades.items()]
        exch = [val.exchange.value for (key, val) in self.trades.items()]
        orderid = [val.orderid for (key, val) in self.trades.items()]
        tradeid = [int(val.tradeid) for (key, val) in self.trades.items()]
        direction = [val.direction.name for (key, val) in self.trades.items()]
        offset = [val.offset.name for (key, val) in self.trades.items()]
        price = [float(val.price) for (key, val) in self.trades.items()]
        datetime_ = [val.datetime for (key, val) in self.trades.items()]
        vol = [val.volume for (key, val) in self.trades.items()]
        trade_df =DataFrame()
        trade_df['gateway']=gwy
        trade_df['symbol']=symbol
        trade_df['exch']=exch
        trade_df['orderid']=orderid
        trade_df['tradeid']=tradeid
        trade_df['direction']=direction
        trade_df['offset']=offset
        trade_df['price']=price
        trade_df['datetime']=datetime_
        trade_df['volume']=vol
        return trade_df
    
    def run_eod(self):
        """"""
        func = self.new_bar
        ix = 0
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
            if d in self.daily_results.keys():
                daily_result = self.daily_results[d]
                daily_result.add_trade(trade)
            else:
                self.output("trade {0} not in scope".format(trade.orderid))

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

    def update_daily_close(self, price: float, notional: float):
        """"""
        d = self.datetime.date()

        daily_result = self.daily_results.get(d, None)
        if daily_result:
            daily_result.close_price = price
            daily_result.close_notional = notional
        else:
            self.daily_results[d] = CDailyResult(d, price, notional)

    def new_bar(self, bar: BarData):
        """"""
        self.bar = bar
        self.datetime = bar.datetime
        self.update_daily_close(bar.close_price, bar.value)
    
    def get_spread_name(self):
        result = ''
        log_file = self.strategy_name + '.log'
        log_file_path = os.path.join(self.strategy_base_path,log_file)
        kp_phrases = ["INFO"]
        trans = read_trans_from_log(log_file_path,kp_phrases)
        #get datetime
        sprds=[t.split(' - ')[1] for t in trans]
        if sprds:
            result = sprds[0]
        return result