from typing import List
from enum import Enum
import pandas as pd
import guerilla.core.config as cfg
import os
from datetime import datetime,timedelta
from guerilla.core.constants import d_format_long,d_format_short,d_format_ms,pattern_number_re,pattern_letters_re
import vnpy_sqlite.sqlite_database as sqlite
from vnpy.trader.object import BarData
from vnpy.trader.constant import Direction, Offset, Exchange, Interval

class TASKSTATUS(Enum):
    DONE = 'DONE'
    FAILED = 'FAILED'
    INITIATED = 'INITIATED'
    NA = 'NOT APPLICABLE'
    
class SOURCE(Enum):
    SINA = 'sina'
    RQ = 'rq'
    YH = 'yahoo'
    TQ = 'tq'
    TB = 'tb'
    CUS = 'custom'
    CC = 'custom_clean'
    DEF = 'default'

class FREQ(Enum):
    DAILY = '1d'
    WEEKLY = '1w'
    MONTHLY = 'month'
    MINUTE1 = '1m'
    MINUTE5 = '5m'
    MINUTE15 = '15m'
    MINUTE30 = '30m'
    HOURLY = '1h'
    
class dsTaskManager:
    
    def __init__(self, date_time: datetime = datetime.now(),src: str = 'sina', taskname:str = 'task') -> None:
        config_ = cfg.config_parser()
        self.logbook_path = os.path.join(config_["dataset"]["base_path"],taskname + '_logbook.csv')
        self.logbook_cache = os.path.join(config_["dataset"]["base_path"],taskname + '_logbook_cache.csv')
        self.tdate = date_time.date()
        self.source = src
        self.logbook = pd.DataFrame(columns=['date','ticker','source','status'])
    
    def initiate_log(self):
        lb_df = self.get_logbook()
        lb_df = lb_df[(lb_df['date'] == self.tdate) & (lb_df['source'] == self.source)].drop_duplicates()
        lb_cache_Df = self.get_logbook(cache=True)
        lb_cache_Df = lb_cache_Df[(lb_cache_Df['date'] == self.tdate) & (lb_cache_Df['source'] == self.source)].drop_duplicates()
        self.logbook = pd.concat([lb_df,lb_cache_Df]).drop_duplicates(subset=['date','ticker','source'],keep='last')
        #save log to cache
        self.logbook.to_csv(self.logbook_cache,index= False)
    
    def update_log(self, ticker: str, status : TASKSTATUS = TASKSTATUS.INITIATED):
        if(len(self.logbook[self.logbook.ticker == ticker])):
            self.logbook.loc[self.logbook.ticker == ticker, 'status']=status.value
        else:
            #self.logbook = self.logbook.append({'date':self.tdate, 'ticker': ticker, 'source': self.source, 'status': status.value },ignore_index=True)
            self.logbook = pd.concat([self.logbook, pd.DataFrame.from_records({'date':self.tdate, 'ticker': ticker, 'source': self.source, 'status': status.value },index=[0])])
        self.logbook.to_csv(self.logbook_cache,index= False)
    
    def push_log(self):
        if os.path.exists(self.logbook_path):
            pre_log_df = self.get_logbook()
            concat_df = pd.concat([pre_log_df,self.logbook])
            concat_df = concat_df.drop_duplicates(subset=['date','ticker','source'], keep='last')
            concat_df.to_csv(self.logbook_path,index= False)
        else:
            self.logbook.to_csv(self.logbook_path,index= False)
        #remove cache file
        if os.path.exists(self.logbook_cache):
            os.remove(self.logbook_cache)
    
    def get_status(self, ticker):
        status = TASKSTATUS.INITIATED.value
        if(len(self.logbook[self.logbook.ticker == ticker])):
            status = self.logbook[(self.logbook.ticker == ticker)]['status'].to_list()[0]
        return TASKSTATUS(status)
    
    def get_logbook(self, cache: bool = False):
        logpath = self.logbook_path
        if cache:
            logpath = self.logbook_cache
        
        if os.path.exists(logpath):
            lg_df = pd.read_csv(logpath)
            lg_df['date'] = lg_df['date'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d").date())
        else:
            lg_df=pd.DataFrame([],columns=['date','ticker','source','status'])     
        return lg_df
    
    def fetch_required(self, ticker):
        required = False
        status = self.get_status(ticker)
        if (status in [TASKSTATUS.INITIATED, TASKSTATUS.FAILED]):
            required = True
        return required
    
    def reset_status(self):
        if(len(self.logbook)):
            self.logbook['status']=TASKSTATUS.INITIATED.value

def save_to_csv(fileName,data_df, is_daily = False):
    """This method is going to be replaced by publish_to_csv.[to be deprecated]

    Args:
        fileName (_type_): _description_
        data_df (_type_): _description_
        is_daily (bool, optional): _description_. Defaults to False.
    """
    if(os.path.exists(fileName)):
        hist_df = pd.read_csv(fileName)
        
        if 'date' in hist_df.columns:
            hist_df['date'] = hist_df['date'].apply(lambda x:datetime.strptime(x,d_format_short))
        elif 'datetime' in hist_df.columns:
            hist_df['datetime'] = hist_df['datetime'].apply(lambda x:datetime.strptime(str(x),d_format_long))
            
        concat_df=pd.concat([hist_df,data_df])
        if is_daily:
            concat_df=concat_df.drop_duplicates(subset=['date'], keep='last')
        else:
            concat_df=concat_df.drop_duplicates(subset=['datetime'], keep='last')
    else:
        concat_df=data_df
    concat_df.to_csv(fileName,index=False)
    #second round clean to remove dups.
    concat_df = pd.read_csv(fileName)
    concat_df = concat_df.drop_duplicates(keep='last')
    concat_df.to_csv(fileName,index=False)

def publish_to_csv(fileName, data_df, use_short_date_format=False):
    """
    Publishes a DataFrame to a CSV file, optionally removing duplicates based on date or datetime.
    Args:
        fileName (str): The path and name of the CSV file to publish to.
        data_df (pandas.DataFrame): The DataFrame to be published.
        is_daily (bool, optional): Specifies whether to remove duplicates based on date (True) or datetime (False).
            Defaults to False.

    Returns:
        None
    """
    date_col = 'datetime'
    if use_short_date_format:
        date_col = 'date'
    data_df[date_col] = pd.to_datetime(data_df[date_col])
    data_df.set_index(date_col, inplace=True)

    if os.path.exists(fileName):
        exist_df = pd.read_csv(fileName,parse_dates=[date_col],index_col=date_col)
        #exist_df.to_csv('existing_csv_check.csv',index=True)
           
        concat_df=pd.concat([exist_df,data_df])
        concat_df = concat_df[~concat_df.index.duplicated(keep='last')]
    else:
        concat_df = data_df

    concat_df.to_csv(fileName,index=True)

def save_to_vn_sqlite(vn_tickers:List[str],
                   s:SOURCE = SOURCE.TQ,
                   start_date:str = '2000-01-01',
                   freq:FREQ = FREQ.MINUTE1,
                   verbose:bool = False):
    
    def push_db(bars:List[BarData]):
        sql_db = sqlite.SqliteDatabase()
        pushed = sql_db.save_bar_data(bars)
        closed = sql_db.db.close()
        return pushed,closed
    
    def parseBar(row,interval = Interval.MINUTE, short_date_format = False):
        date_col = 'datetime'
        if short_date_format:
            date_col = 'date'
            
        return BarData(symbol = row['ticker_name'],\
                    exchange = Exchange(row['exchange']),\
                    datetime = datetime.strptime(str(row[date_col]),d_format_long),\
                    volume = float(row['volume']),\
                    open_price = float(row['open']),\
                    high_price = float(row['high']),\
                    low_price = float(row['low']),\
                    close_price = float(row['close']),\
                    interval = interval,\
                    gateway_name = row['gateway'])
    
    vn_interval = map_vn_interval(freq)
    start_datetime = datetime.strptime(start_date,d_format_short)
    
    for vn_t in vn_tickers:
        t = vn_t.split('.')[0]
        exch = vn_t.split('.')[1]
        t_alias = map_ticker_custom(t,exch)
        ticker_df,_ = fetch_ticker_data(ticker=t_alias,frequency=freq)
        
        is_short_date_format = False
        if 'date' in ticker_df.columns:
            is_short_date_format = True
        
        if len(ticker_df):
            ticker_df['exchange'] = exch
            ticker_df['ticker_name'] = t
            ticker_df['gateway'] = s.value
            bar = ticker_df.apply(parseBar,args = (vn_interval,is_short_date_format,), axis = 1).to_list()
            bar = [b for b in bar if b.datetime > start_datetime]
            success,c = push_db(bar)
            if verbose:
                print("ticker {0} loaded to db:{1}; Count {2}; db closed:{3}".format(t,success,len(bar),c))

def read_from_vn_sqlite(vn_tickers:List[str],
                     freq:FREQ = FREQ.DAILY,
                     start_date:str = '2000-01-01',
                     end_date:str = datetime.now().strftime(d_format_short),
                     verbose:bool = False):
    s_date = datetime.strptime(start_date,d_format_short)
    e_date = datetime.strptime(end_date,d_format_short)
    vn_interval = map_vn_interval(freq)
    sql_db = sqlite.SqliteDatabase()
    result_df = []
    for vn_t in vn_tickers:
        t = vn_t.split('.')[0]
        exch = vn_t.split('.')[1]
        bars = sql_db.load_bar_data(symbol = t, exchange = Exchange(exch), interval = vn_interval, start = s_date, end = e_date)
        bar_df = pd.DataFrame([vars(b) for b in bars])
        result_df.append(bar_df)
        if verbose:
            print("ticker {0} loaded from db; Count {1}".format(t,len(bars)))
    sql_db.db.close()
    
    result_df = pd.concat(result_df)
    return result_df          

def fetch_ticker_data(ticker, frequency:FREQ = FREQ.DAILY, s:SOURCE = SOURCE.DEF, verbose:bool = False):
    config_ = cfg.config_parser()
    base_path = config_["dataset"]["base_path"]
    f_lst = get_list_files(base_path)
    target_lst = []
    
    result = pd.DataFrame()
    message = ''
    tkey_low = '\\' + ticker.lower() + '.csv'
    tkey_up = '\\' + ticker.upper() + '.csv'
    freq_key = '\\' + frequency.value + '\\'

    if s == SOURCE.DEF:
        #search all sources
        target_lst = [f for f in f_lst if (tkey_low in f or tkey_up in f) and freq_key in f]
    elif s == SOURCE.TB:
        tkey_tb = '\\' + ticker.upper() + '.'
        target_lst = [f for f in f_lst if (tkey_tb in f) and (freq_key in f) and (s.value in f)]
    else:
        target_lst = [f for f in f_lst if (tkey_low in f or tkey_up in f) and freq_key in f and s.value in f]
        
    if len(target_lst) == 0:
        message = "no data found for {0}".format(ticker)
    elif len(target_lst) == 1:
        result = pd.read_csv(target_lst[0])
        message = 'success:{0} -- fetched'.format(target_lst[0])
        #print('{0} -- fetched'.format(target_lst[0]))
    else:
        message = "multiple dataset found for {0}, using {1}".format(ticker, target_lst[0])
        result =  pd.read_csv(target_lst[0])
    
    #post processing dataset
    if s == SOURCE.TB and len(result) > 1:
        result.columns = ['datetime','open','high','low','close','volume','money','open_oi']
        result['close_oi'] = result['open_oi']
    
    if 'date' in result.columns:
        result['date'] = result['date'].apply(lambda x:datetime.strptime(x,d_format_short))
    elif 'datetime' in result.columns and frequency == FREQ.DAILY:
        try:
            result['datetime'] = result['datetime'].apply(lambda x:datetime.strptime(str(x),d_format_short))
        except:
            result['datetime'] = result['datetime'].apply(lambda x:datetime.strptime(str(x),d_format_long))
    elif 'datetime' in result.columns:
        result['datetime'] = result['datetime'].apply(lambda x:datetime.strptime(str(x),d_format_long))
    else:
        pass
    
    #add 1min to TQ data
    if s == SOURCE.TQ and len(result) > 1:
        result['datetime'] = result['datetime'] + timedelta(minutes=1)
    
    if verbose:
        print(message)
    
    return result, message

def fetch_sina_fut_baseinfo(file_path:str):
    result_df = pd.DataFrame()
    
    if os.path.exists(file_path):
        result_df = pd.read_csv(file_path)
    
    return result_df

def get_list_files(directory:str):
    result = []
    for path, currentDirectory, files in os.walk(directory):
        for file in files:
            result.append(os.path.join(path, file))
    return result
 
def remove_future_data(dt_now: datetime, data_df:pd.DataFrame):
    latest_datetime = dt_now.replace(tzinfo=None)
    if not len(data_df):
        return pd.DataFrame()
    
    if 'date' in data_df.columns:
        data_df['datetime_key'] = data_df['date'].apply(lambda x:datetime.strptime(x,d_format_short))
    elif 'datetime' in data_df.columns:
        data_df['datetime_key'] = data_df['datetime'].apply(lambda x:datetime.strptime(x,d_format_long))
    data_df = data_df.set_index('datetime_key')
    
    return data_df.loc[data_df.index < latest_datetime]

def remove_past_data(dt_threshold: datetime, data_df:pd.DataFrame, dformat:str = d_format_long):
    threshold_datetime = dt_threshold.replace(tzinfo=None)
    r_df = data_df.copy()
    if not len(r_df):
        return pd.DataFrame()
    
    date_col = 'datetime'
    if 'date' in r_df.columns:
        date_col = 'date'
    
    r_df[date_col] = r_df[date_col].apply(lambda x:datetime.strptime(x,dformat))
    r_df = r_df[r_df[date_col] > threshold_datetime]
    r_df[date_col] = r_df[date_col].apply(lambda x:datetime.strftime(x,d_format_long))
    
    return r_df

def map_exchange_ch(name_ch:str):
    result = name_ch
    if name_ch =="上海期货交易所":
        result = "SHFE"
    elif name_ch =="大连商品交易所":
        result = "DCE"
    elif name_ch =="郑州商品交易所":
        result = "CZCE"
    elif name_ch =="上海国际能源交易中心":
        result = "INE"
    elif name_ch =="广州期货交易所":
        result = "GFEX"
    elif name_ch =="中国金融期货交易所":
        result = "CFFEX"
    else:
        pass
    return result

def map_ticker_custom(ticker:str, exchange:str):
    result = ticker
    if exchange == "郑州商品交易所" or exchange == "CZCE":
        prefix = '2'
        if (int(pattern_letters_re.sub('',ticker)) >=600):
            prefix = '1'
        result = pattern_number_re.sub('',ticker).upper() + prefix + pattern_letters_re.sub('',ticker)
    return result

def map_vn_interval(freq:FREQ):
    result = Interval.MINUTE
    
    if freq == FREQ.MINUTE1:
        result = Interval.MINUTE
    elif freq == FREQ.DAILY:
        result = Interval.DAILY
    elif freq == FREQ.WEEKLY:
        result = Interval.WEEKLY
    elif freq == FREQ.HOURLY:
        result = Interval.HOUR
    else:
        print("unsupported freq {0} in vn".format(freq))

    return result