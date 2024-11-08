import pandas as pd
from datetime import datetime
import yfinance as yf
import pytz
from pytz import timezone
import time
import ta
import os
import re
from guerilla.core.utility import logger
from guerilla.core.constants import d_format_short
from ..dsUtil import dsTaskManager,TASKSTATUS,SOURCE, publish_to_csv

#re expression
pattern_number_re = re.compile(r'[0-9]')
pattern_letters_re = re.compile(r'[A-Za-z]')

class worker:
    def __init__(self) -> None:
        self.datetime_local = datetime.now()
        utc_now = datetime.now(pytz.utc)
        china_tz = timezone('Asia/Shanghai')
        
        self.datetime_bj = utc_now.astimezone(china_tz)
        self.tskmgr = dsTaskManager(self.datetime_bj,src=SOURCE.YH.value)
        self.logger_ = logger(__name__)
        print('yahoo worker ready.')
    
    def start(self):
        self.tskmgr.initiate_log()
        print('yahoo worker started..')
    
    def stop(self):
        self.tskmgr.push_log()
        print('yahoo worker stopped.')
    
    def create_master_file(self, file_path:str):
        master_df = pd.DataFrame([{"tickers":"spy"},{"tickers":"qqq"}])
        master_df.to_csv(file_path,index=False)
        
    
    def get_master_list(self, file_path:str):
        master_df = pd.read_csv(file_path)
        return master_df['tickers'].tolist()
        
    def fetch_daily_hist_data(self, ticker: str,
                              start_date: datetime=datetime.strptime('2000-01-01',d_format_short),
                              end_date: datetime=datetime.now(),
                              basepath:str = './/dateset//ticker_generic_hist//',
                              source:str = 'yahoo',
                              slp: int = 1,
                              refetch:bool = False):
        
        ticker_task = ticker + '_1d'
        start_str = start_date.strftime(d_format_short)
        end_str = end_date.strftime(d_format_short)
        
        if not self.tskmgr.fetch_required(ticker_task ) and not refetch:
            return
        
        filepath = os.path.join(basepath,source,'1d')
        success = False
        hist_df:pd.DataFrame = pd.DataFrame()
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        ticker_obj = yf.Ticker(ticker)
        try:
            hist_df = ticker_obj.history(start=start_str,end=end_str)
            hist_df.reset_index(inplace=True)
            hist_df.columns = [str.lower(c) for c in hist_df.columns]
            hist_df['date'] = hist_df['date'].apply(lambda x:datetime.strptime(x.strftime(d_format_short),d_format_short))
            hist_df.sort_values(by='date',inplace=True)
            hist_df.set_index('date',inplace=True)
            target_file = os.path.join(filepath,ticker + '.csv')
            publish_to_csv(target_file, hist_df, use_short_date_format=True)
            success = True
            self.tskmgr.update_log(ticker_task,TASKSTATUS.DONE)
            time.sleep(slp)
        except:
            self.logger_.warning("An exception occurred when fetching {0}".format(ticker))
            self.tskmgr.update_log(ticker_task,TASKSTATUS.FAILED)
            success = False
        return success, hist_df