from collections import defaultdict
from datetime import date, datetime, timedelta
import pytz
from pytz import timezone
from typing import Callable, Type, Dict, List
from enum import Enum
from functools import partial

import numpy as np
import pandas as pd
import re
import json
import os
from pandas import DataFrame
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import akshare as ak
import talib

from vnpy.trader.constant import (Direction, Offset, Exchange,Interval, Status)
from vnpy.trader.object import BarData
from vnpy.trader.utility import round_to, floor_to, ceil_to, extract_vt_symbol
from vnpy_spreadtrading.base import (SpreadData,LegData)
import vnpy_sqlite.sqlite_database as sqlite

import guerilla.core.config as cfg
import guerilla.dataservice.source.tq as tq
from guerilla.dataservice.dsUtil import SOURCE,FREQ, save_to_vn_sqlite, read_from_vn_sqlite

from .tutils import load_bar_data,read_trans_from_log,load_json_file

d_format_long = "%Y-%m-%d %H:%M:%S"
d_format_short = "%Y-%m-%d"
pattern_letters_re = re.compile(r'[A-Za-z]')
pattern_number_re = re.compile(r'[0-9]')

def parseBar(row):
    return BarData(symbol = row['order_book_id'],\
                   exchange = Exchange(row['exchange']),\
                   datetime = datetime.strptime(row['datetime'],d_format_long),\
                  volume = float(row['volume']),\
                  open_price = float(row['open']),\
                  high_price = float(row['high']),\
                  low_price = float(row['low']),\
                  close_price = float(row['close']),\
                  interval = Interval.MINUTE,\
                  gateway_name = row['gateway'])

def remove_future_data(dt_now: datetime, data_df:DataFrame):
        latest_datetime = dt_now.replace(tzinfo=None)
        if not len(data_df):
            return DataFrame()
        
        if 'date' in data_df.columns:
            data_df['datetime_key'] = data_df['date'].apply(lambda x:datetime.strptime(x,d_format_short))
        elif 'datetime' in data_df.columns:
            data_df['datetime_key'] = data_df['datetime'].apply(lambda x:datetime.strptime(x,d_format_long))
        data_df = data_df.set_index('datetime_key')
        
        return data_df.loc[data_df.index < latest_datetime]

def fetchBars(field_name: str, bars: List[BarData]):
    return [getattr(b,field_name) for b in bars]

class CTADataFeeder:
    def __init__(self, strategy_base_path : str = Path.cwd(), 
                 fut_c_hist_path:str = '', 
                 refetch: bool = True, 
                 src: SOURCE = SOURCE.TQ, 
                 freq:FREQ = FREQ.MINUTE1) -> None:
        self.strategy_base_path = strategy_base_path
        self.fut_contract_path = fut_c_hist_path
        self.refetch = refetch
        self.cfg_ = cfg.config_parser()
        if not self.fut_contract_path:
            self.fut_contract_path =os.path.join(self.cfg_["dataset"]["base_path"],self.cfg_["dataset"]["fut_ct_folder"])
            print('using default fut contract path: {0}'.format(self.fut_contract_path))
        self.freq = freq
    
    def fetch_all_tickers(self):
        cta_setting_path = os.path.join(self.strategy_base_path,'.vntrader','cta_strategy_setting.json')
        cta_settings = load_json_file(cta_setting_path)
        tickers = [v['vt_symbol'] for (k,v) in cta_settings.items()]
        
        if self.refetch:
            self.refetch_data(tickers)
        
        s_date_str = datetime.strftime(datetime.now() - timedelta(weeks=13),d_format_short)
        save_to_vn_sqlite(tickers, freq=self.freq, start_date=s_date_str)
        return True
    
    def refetch_data(self, tickers: List[str]):
        u = self.cfg_["dataprovider"]["tq"]["user"]
        p = self.cfg_["dataprovider"]["tq"]["pwd"]
        tqwkr = tq.worker(user_name=u,password=p)
        tqwkr.start()
        for t in tickers:
            ticker, exch_str = t.split('.')
            s = tqwkr.fetch_future_hist_data(ticker=ticker,
                                             exch=exch_str,
                                             freq=self.freq,
                                             basepath=self.fut_contract_path,
                                             slp=3,
                                             refetch=True)
            print('ticker {0} refetch finished with status {1}'.format(t,s))
        tqwkr.stop()
        
class SpreadDataFeeder:
    def __init__(self, strategy_base_path : str = Path.cwd(),
                 fut_c_hist_path:str = '', 
                 refetch: bool = True, 
                 src: SOURCE = SOURCE.TQ, 
                 freq:FREQ = FREQ.MINUTE1):
        self.spreads : Dict[str, SpreadData] = {}
        self.spread_settings={}
        self.spreadBars : Dict[str,List[BarData]] = {}
        self.legBars : Dict[str,List[BarData]] = {}
        self.legs : List[str] = []
        self.refetch = refetch
        self.strategy_base_path = strategy_base_path
        self.fut_contract_path = fut_c_hist_path
        if not self.fut_contract_path:
            cfg_ = cfg.config_parser()
            self.fut_contract_path =os.path.join(cfg_["dataset"]["base_path"],cfg_["dataset"]["fut_ct_folder"])
            print('using default fut contract path: {0}'.format(self.fut_contract_path))
        self.freq = freq
        
        utc_now = datetime.now(pytz.utc)
        china_tz = timezone('Asia/Shanghai')
        self.datetime_bj = utc_now.astimezone(china_tz)
        self.lookback = 10
    
    def set_spreads(self):
        spread_setting_path = os.path.join(self.strategy_base_path,'.vntrader','spread_trading_setting.json')
        sprd_settings = load_json_file(spread_setting_path)
        self.spread_settings = sprd_settings
        for setting in sprd_settings:
            self.add_spread(setting)
        
    def add_spread(self, spread_setting: dict):
        var_list = [elem['variable'] for elem in spread_setting['leg_settings']]
        vt_symbol_list = [elem['vt_symbol'] for elem in spread_setting['leg_settings']]
        direction_list = [elem['trading_direction'] for elem in spread_setting['leg_settings']]
        multi_list = [elem['trading_multiplier'] for elem in spread_setting['leg_settings']]
        var_dict = dict(zip(var_list, vt_symbol_list))
        var_direct_dict = dict(zip(var_list, direction_list))
        trade_multi_dict = dict(zip(vt_symbol_list, multi_list))
        legs_list = list([LegData(elem['vt_symbol']) for elem in spread_setting['leg_settings']])
        
        for leg in vt_symbol_list:
            if leg not in self.legs:
                self.legs.append(leg)
                
        self.spreads[spread_setting['name']] = SpreadData(spread_setting['name'],
                                 legs= legs_list,
                                 variable_symbols= var_dict,
                                 variable_directions=var_direct_dict,
                                 price_formula=spread_setting['price_formula'],
                                 trading_multipliers=trade_multi_dict,
                                 active_symbol=spread_setting['active_symbol'],
                                 min_volume= spread_setting['min_volume'])

    def fetch_legs_data(self):
        save_to_vn_sqlite(self.legs, freq=self.freq)
    
    def load_leg_bars(self):
        start_time = self.datetime_bj - timedelta(days=10)
        end_time = self.datetime_bj
        sql_db = sqlite.SqliteDatabase()
        for leg in self.legs:
            ticker, exch = extract_vt_symbol(leg)
            self.legBars[leg] = sql_db.load_bar_data(symbol = ticker,
                                        exchange = Exchange(exch),
                                        interval = Interval.MINUTE,
                                        start = start_time,
                                        end = end_time)
            bars = self.legBars[leg]
            days_loaded = (bars[-1].datetime - bars[0].datetime).days
            print("ticker {0} loaded:{1} counts for {2} days.".format(ticker, len(bars),days_loaded))
        sql_db.db.close()
    
    def load_spread_bars(self):
        start_time = self.datetime_bj - timedelta(days=10)
        end_time = self.datetime_bj
        for nm,sprdD in self.spreads.items():
            self.spreadBars[nm] = load_bar_data(sprdD,Interval.MINUTE,start_time,end_time)
            bars = self.spreadBars[nm]
            days_loaded = (bars[-1].datetime - bars[0].datetime).days
            print("spread {0} loaded:{1} counts for {2} days.".format(nm, len(bars), days_loaded))
    
    def check_spreads(self):
        #load data points from sqlite db
        self.load_leg_bars()
        self.load_spread_bars()
        #plot
        self.show_charts()
    
    def show_charts(self):
        sprd_lst = list(self.spreads.keys())
        rows = len(sprd_lst)
        fig, axs = plt.subplots(rows,2, figsize=(20, 10), constrained_layout=True)
        for r in range(rows):
            sname = sprd_lst[r]
            #plot1
            leg_lst = list(self.spreads[sname].legs.keys())
            if rows == 1:
                ax = axs[0]
            else:
                ax = axs[r,0]
            for l in leg_lst:
                y = fetchBars("volume", self.legBars[l])
                y_smooth =talib.SMA(np.asarray(y),60)
                ax.plot(y_smooth,label=l)
            ax.legend()
            ax.set_title(sname + ' Legs Volume MA60')
            ax.set_ylabel('volume')
            #plot2
            if rows == 1:
                ax = axs[1]
            else:
                ax = axs[r,1]
            y = fetchBars("close_price", self.spreadBars[sname])
            y_smooth =talib.SMA(np.asarray(y),15)
            ax.plot(y,label='spot')
            ax.plot(y_smooth,label='ma15')
            ax.legend()
            ax.set_title(sname + ' Spread Close Spot/MA15')
            ax.set_ylabel('point')
            
    
        
        