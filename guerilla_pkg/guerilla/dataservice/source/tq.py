
import pandas as pd
from datetime import datetime
import pytz
from pytz import timezone
import time
import os
from tqsdk import TqApi, TqAuth,tafunc
from guerilla.core.utility import logger
from guerilla.core.constants import d_format_ms,pattern_number_re,pattern_letters_re
from guerilla.dataservice.dsUtil import dsTaskManager,TASKSTATUS,SOURCE,FREQ, save_to_csv,publish_to_csv,remove_past_data,map_ticker_custom,map_exchange_ch

datetime_cutoff = datetime(1971,1,1,0,0,0)

def map_tq_freq(freq:FREQ):
    result = 60
    if freq == FREQ.MINUTE15:
        result = 60 * 15
    elif freq == FREQ.MINUTE30:
        result = 60 * 30
    elif freq == FREQ.HOURLY:
        result = 60 * 60
    elif freq == FREQ.DAILY:
        result = 86400
    elif freq == FREQ.WEEKLY:
        result = 86400 * 7
    elif freq == FREQ.MONTHLY:
        result = 86400 * 28
    else:
        pass
     
    return result

class worker:
    def __init__(self, user_name:str = "user", password:str = "xxxx", worker_name:str = 'tq') -> None:
        self.datetime_local = datetime.now()
        utc_now = datetime.now(pytz.utc)
        china_tz = timezone('Asia/Shanghai')
        self.datetime_bj = utc_now.astimezone(china_tz)
        self.tskmgr = dsTaskManager(self.datetime_bj,src=SOURCE.TQ.value,taskname=worker_name)
        self.logger_ = logger(__name__+ '_' + worker_name)
        self.api = None
        try:
            self.api = TqApi(auth=TqAuth(user_name, password))
            print('tqsdk worker ready.')
        except:
            print('tqsdk worker initiation failed.')
    
    def start(self):
        self.tskmgr.initiate_log()
        print('tqsdk worker started..')
    
    def stop(self):
        self.api.close()
        self.tskmgr.push_log()
        print('tqsdk worker stopped.')
    
    def fetch_future_hist_data(self, ticker: str,\
                            exch:str,\
                            freq: FREQ,\
                            data_len:int = 8000,
                            basepath:str = './/dateset//future_generic_hist//',\
                            source:str = 'tq',\
                            slp: int = 1,\
                            refetch:bool = False):
        ticker_task = ticker + '_' + freq.value
        if not self.tskmgr.fetch_required(ticker_task) and not refetch:
            return
        
        filepath = os.path.join(basepath,source,freq.value)
        success = False
        hist_df = pd.DataFrame()
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        tq_freq = map_tq_freq(freq)
        exch_code = map_exchange_ch(exch)
            
        try:
            hist_df = self.api.get_kline_serial(exch_code + '.' + ticker, tq_freq,data_length=data_len)
            hist_df['datetime']=hist_df['datetime'].map(tafunc.time_to_str)
            hist_df = remove_past_data(datetime_cutoff,hist_df,d_format_ms)
            target_file = os.path.join(filepath,map_ticker_custom(ticker,exch) + '.csv')
            #save_to_csv(target_file, hist_df, is_daily=False)
            publish_to_csv(target_file, hist_df)
            success = True
            self.tskmgr.update_log(ticker_task,TASKSTATUS.DONE)
            time.sleep(slp)
            
        except Exception as e:        
            self.logger_.warning("An exception occurred when fetching {0}, {1}:{2}".format(ticker,freq.value,e))
            self.tskmgr.update_log(ticker_task,TASKSTATUS.FAILED)
            success = False
        return success, hist_df
    
    def fetch_dominant_contract_map(self, contracts_list:list[tuple],
                                    basepath:str = './/dateset//future_base_info//',
                                    source:str = 'tq'):
        task_key = 'dominant_ct_map'
        if not self.tskmgr.fetch_required(task_key):
            return
        
        filepath = os.path.join(basepath,source)
        success = False
        result_df = pd.DataFrame()
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            
        symbol_list = ['KQ.m@' + map_exchange_ch(exch) + '.' + pattern_number_re.sub('',ticker) for ticker,exch in contracts_list]
        symbol_list = list(set(symbol_list))

        result_df = self.api.query_his_cont_quotes(symbol=symbol_list, n=8000)
        cols_new = [c.replace('KQ.m@', '') for c in result_df.columns]
        result_df.columns = cols_new
        result_df= result_df[result_df['date'] > datetime(2000,7,14)]
        target_file = os.path.join(filepath,'dominant_contract_hist.csv')
        result_df.to_csv(target_file,index=False)
        self.tskmgr.update_log(task_key,TASKSTATUS.DONE)
        success = True
        return success, result_df