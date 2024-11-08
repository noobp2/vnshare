import pandas as pd
import numpy as np
import re
from typing import List
from datetime import date
from datetime import datetime as dt
from datetime import time as tm
from datetime import timedelta as td
from .dmutil import read_trading_info_df,getActiveTimeBuckets,generateDateTimeKey,parseTime
from guerilla.core.utility import logger
from guerilla.core.constants import d_format_long

pattern_number_re = re.compile(r'[0-9]')

def flatFillPrev(prev_row, current_datetime: dt):
        result = prev_row.copy()
        result['datetime'] = dt.strftime(current_datetime,d_format_long)
        result['open'] = result['close']
        result['high'] = result['close']
        result['low'] = result['close']
        result['volume'] = 0
        return result
    
class tsFillEngine:
    def __init__(self, start_time = dt(1900,1,1,9,0,0),end_time = dt.now()):
        self.c_trade_time_dict = {}
        self.gen_trade_time_dict = {}
        self.logger_ = logger(__name__)
        self.start_datetime = start_time
        self.end_datetime = end_time
    
    def load_trading_time(self, fut_trading_time_path: str = ".//dataset//future_base_info//"):
        fut_info_df = read_trading_info_df(fut_trading_time_path)
        #build trading hours map contract and gen ticker
        gen_df = fut_info_df[['generic_ticker_sina','time_buckets_obj']].drop_duplicates(subset = ['generic_ticker_sina'])
        c_df = fut_info_df[['contract_code_sina','time_buckets_obj']].drop_duplicates(subset = ['contract_code_sina'])
        self.gen_trade_time_dict = dict(zip(gen_df['generic_ticker_sina'],gen_df['time_buckets_obj']))
        self.c_trade_time_dict = dict(zip(c_df['contract_code_sina'],c_df['time_buckets_obj']))
    
    def get_trading_time(self, ticker: str):
        result = []
        genkey = pattern_number_re.sub(' ', ticker).split(' ')[0].upper() + '0'
        if ticker in self.c_trade_time_dict.keys():
            result = self.c_trade_time_dict[ticker]
        elif genkey in self.gen_trade_time_dict.keys():
            result = self.gen_trade_time_dict[genkey]
        else:
            self.logger_.error('cannot locate the trading time for {0}'.format(ticker))  
        return result
    
    def flatFillData(self, fut_data_df, time_bucket_lst):
        column_lst = ['datetime','open','high','low','close','volume','hold']
        s_datetime = max(fut_data_df.index[0],self.start_datetime)
        e_datetime = min(fut_data_df.index[-1],self.end_datetime)
        result_df = pd.DataFrame(columns=column_lst)
        dates = fut_data_df['date'].drop_duplicates()
        for d in dates:
            temp_df = fut_data_df[fut_data_df['date'] == d]
            tbs_active = getActiveTimeBuckets(time_bucket_lst, temp_df['time'])
            dt_keys = generateDateTimeKey(tbs_active, d, s_datetime, e_datetime)
            for dt_key in dt_keys:
                if dt_key in temp_df.index:
                    result_df = result_df.append(temp_df[column_lst].loc[dt_key])
                elif len(result_df) > 0:
                    result_df = result_df.append(flatFillPrev(result_df.iloc[-1],dt_key))
                else:
                    pass
        return result_df
    
    def process(self,fut_data,ticker):
        time_buckets_lst = self.get_trading_time(ticker)
        fut_data_df = fut_data
        fut_data_df['dt_key'] = fut_data_df['datetime'].apply(lambda x: dt.strptime(x,d_format_long))
        fut_data_df['date'] = fut_data_df['dt_key'].apply(lambda x: x.date())
        fut_data_df['time'] = fut_data_df['dt_key'].apply(lambda x: x.time())
        fut_data_df = fut_data_df.set_index('dt_key')
        result_df = self.flatFillData(fut_data_df, time_buckets_lst)
        return result_df
    
    
class tsFilterEngine:
    
    def __init__(self,
                 volFilter: bool = True,
                 timeFilter: bool = True,
                 stdmulti: float = 5,
                 time_filter_list: List[str] = [],
                 fields: List[str] = ['close','open','high','low']):
        self.stdmulti = stdmulti
        self.time_filter_list =[parseTime(t) for t in time_filter_list]
        self.fields = fields
        self.volfilter = volFilter
        self.timefilter = timeFilter
        self.data_df = pd.DataFrame()
    
    def set_time_filter(self, time_filter_list: List[str] = []):
        self.time_filter_list =[parseTime(t) for t in time_filter_list]
        
    def removeOutliers(self):
        for f in self.fields:
            lb,ub = self.calcband(self.data_df[f])
            outlier_inds = self.data_df.loc[(self.data_df[f] <= lb) | (self.data_df[f] >= ub)].index
            self.data_df = self.data_df.loc[[ind for ind in self.data_df.index if ind not in outlier_inds]]
    
    def removeTimeStamps(self):
        self.data_df = self.data_df.loc[[ind for ind in self.data_df.index if ind.time() not in self.time_filter_list]]
    
    def calcband(self, arry):
        lb = np.mean(arry) - self.stdmulti * np.std(arry)
        ub = np.mean(arry) + self.stdmulti * np.std(arry)
        return lb,ub
    
    def process(self, data):
        self.data_df = data
        self.data_df['datetime_key'] = self.data_df['datetime'].apply(lambda x:dt.strptime(x,d_format_long))
        self.data_df = self.data_df.set_index('datetime_key')
        if self.volfilter:
            self.removeOutliers()
        
        if self.timefilter:
            self.removeTimeStamps()
        
        return self.data_df
        
        
        
        