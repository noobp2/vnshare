from typing import Dict
from .fbase import FactorBase,FactorVisualEngine
from guerilla.dataservice.dsUtil import FREQ,SOURCE
import alphalens as al
import guerilla.core.config as cfg
import os
import pandas as pd
from datetime import datetime, timedelta

class backtestEngine:

    market:str = "china_fut"
    freq = FREQ.HOURLY
    source_ = SOURCE.DEF
    factorEngine = None
    f_setting = {}
    ticker_group_map = {}
    num_buckets = 5
    horizon = (1,5,10)
    bin_type = "sigma"
    static_bin = []
    start_date = datetime(1900,1,1)
    end_date = datetime(1900, 1, 1)
    
    def __init__(self,
                 factor:FactorBase, 
                 factor_setting:Dict = {},
                 bt_setting:Dict = {},
                 market:str = "china_fut",
                 freq:FREQ = FREQ.HOURLY, 
                 s:SOURCE = SOURCE.DEF,
                 start_date:datetime = datetime.now() - timedelta(weeks=156),
                 end_date:datetime = datetime.now()) -> None:
        self.market = market
        self.factorEngine = factor
        self.f_setting = factor_setting
        self.freq = freq
        self.source_ = s
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_group_map = self.read_group_tag()

        if "num_buckets" in bt_setting.keys():
            self.num_buckets = bt_setting["num_buckets"]
        if "horizon" in bt_setting.keys():
            self.horizon = bt_setting["horizon"]
        if "bin_type" in bt_setting.keys():
            self.bin_type = bt_setting["bin_type"]
        if "static_bin" in bt_setting.keys():
            self.static_bin = bt_setting["static_bin"]


    def run_single_asset_analysis(self, ticker:str, show: bool = True):
        f_instant = self.factorEngine(ticker = ticker,f = self.freq, s = self.source_,g = self.ticker_group_map[ticker])
        f_instant.init_data()
        f_instant.calc_factor()
        fv = FactorVisualEngine(f=self.freq,num_q=self.num_buckets, horizon=self.horizon, bin_type= self.bin_type, static_bin=self.static_bin)
        fv.init_data(f_instant.get_visual_data(start_date = self.start_date, end_date = self.end_date))

        if show:
            fv.plot_tear_sheet()
        
        return fv.factor_data

    def run_all_market_analysis(self, tickers:list = ['all'], show:bool = True):
        ticker_list = tickers
        if 'all' in tickers:
            ticker_list = self.ticker_group_map.keys()
        visual_multi_data = []
        f_stats = []
        fv = FactorVisualEngine(f=self.freq,num_q=self.num_buckets, horizon=self.horizon, bin_type= self.bin_type, static_bin=self.static_bin)
        
        for t in ticker_list:
            f_instant = self.factorEngine(ticker = t,f = self.freq, s = self.source_,g = self.ticker_group_map[t])
            f_instant.init_data()
            f_instant.calc_factor()
            v_data = f_instant.get_visual_data(start_date = self.start_date, end_date = self.end_date)
            #calc single factor analysis
            fv.init_data(v_data)
            #dump fv factor data to excel
            stats_dict = fv.calc_factor_stats()
            stats_dict['ticker'] = t
            f_stats.append(stats_dict)
            visual_multi_data.append(v_data.copy())
        
        #dump f_stats to excel
        f_stats_df = pd.DataFrame(f_stats)
        #cross sectional factor analysis
        visual_multi_data = pd.concat(visual_multi_data)
        fv.init_data(visual_multi_data)

        if show:
            print("Individual Ticker IC.")
            al.utils.print_table(f_stats_df)
            fv.plot_tear_sheet_c1()
        
        return f_stats_df



    def read_group_tag(self):
        config = cfg.config_parser()
        base_path = config["dataset"]["base_path"]
        tag_file_path = os.path.join(base_path,config["dataset"]["fut_info_folder"],"ticker_tag_info.csv")
        tag_df = pd.read_csv(tag_file_path)
        tickers = tag_df[tag_df['group_'] == self.market]['ticker_name']
        grp =  tag_df[tag_df['group_'] == self.market]['g1']
        return dict(zip(tickers,grp))