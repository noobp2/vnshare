import pandas as pd
import akshare as ak
from datetime import datetime
import pytz
from pytz import timezone
import time
import ta
import os
import re
from guerilla.core.utility import logger
from guerilla.core.constants import pattern_letters_re, pattern_number_re
from ..dsUtil import dsTaskManager,TASKSTATUS,save_to_csv,remove_future_data

#function definations
def func_contract_date(row):
    result =''
    if row['交易所名称']=='郑州商品交易所':
        result = datetime.strptime(('2'+ pattern_letters_re.sub('',row['合约代码'])),"%y%m")
    else:
        result = datetime.strptime((pattern_letters_re.sub('',row['合约代码'])),"%y%m")
    return result

def func_is_active(row):
    return (row['价格更新时间']<= row['maturity_month'])
    #return (row['maturity_month'].month >= row['价格更新时间'].month) & (row['maturity_month'].year >= row['价格更新时间'].year)

def func_sina_contract_code(row):
    result =''
    if row['交易所名称']=='郑州商品交易所':
        result = pattern_number_re.sub('',row['合约代码']).upper() + '2'+ pattern_letters_re.sub('',row['合约代码'])
    else:
        result = pattern_number_re.sub('',row['合约代码']).upper() + pattern_letters_re.sub('',row['合约代码'])
    return result

def map_sina_freq(freq:str):
    freq_s = freq
    if 'm' in freq:
        freq_s = pattern_letters_re.sub('', freq)
    elif 'h' in freq:
        freq_s = '60'
    else:
        pass
     
    return freq_s

class worker:
    def __init__(self) -> None:
        self.datetime_local = datetime.now()
        
        utc_now = datetime.now(pytz.utc)
        china_tz = timezone('Asia/Shanghai')
        
        self.datetime_bj = utc_now.astimezone(china_tz)
        self.tskmgr = dsTaskManager(self.datetime_bj)
        self.logger_ = logger(__name__)
        print('sina worker ready')
    
    def start(self):
        self.tskmgr.initiate_log()
        print('sina worker started')
    
    def stop(self):
        self.tskmgr.push_log()
        print('sina worker stopped')
    
    def fetch_future_basic_info(self,basepath: str = './/dateset//future_base_info//'):
        success = False
        filepath = os.path.join(basepath,'sina')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        utc_now = datetime.now(pytz.utc)
        china_tz = timezone('Asia/Hong_Kong')
        china_now = utc_now.astimezone(china_tz)
        date_str = china_now.strftime("%Y%m%d")
        #fees info
        futures_fees_info_df = ak.futures_fees_info()
        #contract details info
        contracts_detail_dict = {}

        try:
            futures_contract_info_shfe_df = ak.futures_contract_info_shfe(date=date_str)
            futures_contract_info_shfe_df.to_csv(os.path.join(filepath,'contracts_shfe_'+ china_now.strftime("%Y-%m-%d") + '.csv'),index=False)
            contract_mat_shfe_df = futures_contract_info_shfe_df[["合约代码","上市日","到期日"]]
            contracts_detail_dict['shfe'] = contract_mat_shfe_df
        except:
            self.logger_.warning("Failed to fetch shfe contract info")
        try:
            futures_contract_info_ine_df = ak.futures_contract_info_ine(date=date_str)
            futures_contract_info_ine_df.to_csv(os.path.join(filepath,'contracts_ine_'+ china_now.strftime("%Y-%m-%d") + '.csv'),index=False)
            contract_mat_ine_df = futures_contract_info_ine_df[["合约代码","上市日","到期日"]]
            contracts_detail_dict['ine'] = contract_mat_ine_df
        except:
            self.logger_.warning("Failed to fetch ine contract info")
        
        try:
            futures_contract_info_dce_df = ak.futures_contract_info_dce()
            futures_contract_info_dce_df.to_csv(os.path.join(filepath,'contracts_dce_'+ china_now.strftime("%Y-%m-%d") + '.csv'),index=False)
            contract_mat_dce_df = futures_contract_info_dce_df[["合约代码","开始交易日","最后交易日"]]
            contract_mat_dce_df.columns = ["合约代码","上市日","到期日"]
            contracts_detail_dict['dce'] = contract_mat_dce_df
        except:
            self.logger_.warning("Failed to fetch dce contract info")
        
        try:
            futures_contract_info_czce_df = ak.futures_contract_info_czce(date=date_str)
            futures_contract_info_czce_df.to_csv(os.path.join(filepath,'contracts_czce_'+ china_now.strftime("%Y-%m-%d") + '.csv'),index=False)
            contract_mat_czce_df = futures_contract_info_czce_df[["合约代码","第一交易日","最后交易日待国家公布2025年节假日安排后进行调整"]]
            contract_mat_czce_df.columns = ["合约代码","上市日","到期日"]
            contracts_detail_dict['czce'] = contract_mat_czce_df
        except:
            self.logger_.warning("Failed to fetch czce contract info")
        
        try:
            futures_contract_info_gfex_df = ak.futures_contract_info_gfex()
            futures_contract_info_gfex_df.to_csv(os.path.join(filepath,'contracts_gfex_'+ china_now.strftime("%Y-%m-%d") + '.csv'),index=False)
            contract_mat_gfex_df = futures_contract_info_gfex_df[["合约代码","开始交易日","最后交易日"]]
            contract_mat_gfex_df.columns = ["合约代码","上市日","到期日"]
            contracts_detail_dict['gfex'] = futures_contract_info_gfex_df
        except:
            self.logger_.warning("Failed to fetch gfex contract info")
        
        #ignore cffex for now because of server issue, it can be done locally...
        #try:
        #    futures_contract_info_cffex_df = ak.futures_contract_info_cffex(date=date_str)
        #    futures_contract_info_cffex_df.to_csv(os.path.join(filepath,'contracts_cffex_'+ china_now.strftime("%Y-%m-%d") + '.csv'),index=False)
        #    contract_mat_cffex_df = futures_contract_info_cffex_df[["合约代码","上市日","最后交易日"]]
        #    contract_mat_cffex_df.columns = ["合约代码","上市日","到期日"]
        #    contracts_detail_dict['cffex'] = contract_mat_cffex_df
        #except:
        #    self.logger_.warning("Failed to fetch cffex contract info")
        
        contract_mat_df = pd.concat(list(contracts_detail_dict.values()))
        
        info_df = pd.merge(futures_fees_info_df,contract_mat_df,how='left',left_on='合约代码',right_on='合约代码')
        info_df['fetch_time_cn']=china_now
        #info_df['is_active']=info_df.apply(func_is_active, axis=1)
        info_df.to_csv(os.path.join(filepath,'future_info_'+ china_now.strftime("%Y-%m-%d") + '.csv'),index=False)
        
        return success, info_df

    def fetch_future_basic_info_9qi(self, source: str = 'sina',\
                            basepath: str = './/dateset//future_base_info//'):
        success = False
        info_df = None
        dummy_ticker = 'all_info_0'
        filepath = os.path.join(basepath,source)
        
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            
        if not self.tskmgr.fetch_required(dummy_ticker):
            info_df = pd.read_csv(os.path.join(filepath,'future_info_'+ self.datetime_bj.strftime("%Y-%m-%d") + '.csv'))
            return True,info_df
            
        info_df = ak.futures_comm_info(symbol="所有")
        info_df['generic_ticker_sina']=[pattern_number_re.sub('',c).upper()+'0' for c in info_df['合约代码']]
        info_df['contract_code_sina']=info_df.apply(func_sina_contract_code,axis=1)
        info_df['underlying']=[pattern_number_re.sub('',c) for c in info_df['合约名称']]
        info_df['maturity_month']=info_df.apply(func_contract_date,axis=1)
        info_df['价格更新时间']=[datetime.fromisoformat(s) for s in info_df['价格更新时间']]
        info_df['is_active']=info_df.apply(func_is_active, axis=1)
        info_df['units']=100*info_df['保证金-每手']/(info_df['保证金-买开']*info_df['现价'])
        info_df['fetch_time_cn']=self.datetime_bj
        info_df.to_csv(os.path.join(filepath,'future_info_'+ self.datetime_bj.strftime("%Y-%m-%d") + '.csv'),index=False)
        self.tskmgr.update_log(dummy_ticker,TASKSTATUS.DONE)
        success = True
        return success, info_df

    def fetch_future_hist_data(self, ticker: str,\
                            freq: str,\
                            basepath:str = './/dateset//future_generic_hist//',\
                            source:str = 'sina',\
                            slp: int = 1):
        ticker_task = ticker + '_' + freq
        if not self.tskmgr.fetch_required(ticker_task):
            return
        
        filepath = os.path.join(basepath,source,freq)
        success = False
        hist_df = pd.DataFrame()
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        sina_freq = map_sina_freq(freq)
        try:
            if sina_freq == '1d':
                hist_df = ak.futures_zh_daily_sina(symbol=ticker)
            elif sina_freq in ['1','5','15','30','60']:
                hist_df = ak.futures_zh_minute_sina(symbol=ticker, period=sina_freq)
            else:
                hist_df = None
                self.logger_.warning("freq {0} not defined in sina api for ticker {1}".format(freq,ticker))
            hist_df = remove_future_data(self.datetime_bj, hist_df)
            target_file = os.path.join(filepath,ticker + '.csv')
            save_to_csv(target_file, hist_df, is_daily=(freq == '1d'))
            success = True
            self.tskmgr.update_log(ticker_task,TASKSTATUS.DONE)
            time.sleep(slp)
        except:
            self.logger_.warning("An exception occurred when fetching {0}".format(ticker))
            self.tskmgr.update_log(ticker_task,TASKSTATUS.FAILED)
            success = False
        return success, hist_df