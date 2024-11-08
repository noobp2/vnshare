import pandas as pd
import numpy as np
from enum import Enum
import glob
import os
from typing import List
from datetime import date
from datetime import datetime as dt
from datetime import time as tm
from datetime import timedelta as td

class STITCH_CODE(Enum):
    VANILLA = 'A'
    DIFF = 'D'
    RATIO = 'R'

def get_latest_file(path: str,ftype: str = '*.csv'):
    list_of_files = glob.glob(os.path.join(path,ftype)) # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def parseTime(t: str):
    #t is time in string format: HH:MM
    hour = 0
    minute = 0
    t_lst = t.split(':')
    if len(t_lst) ==2:
        hour = int(t_lst[0])
        minute = int(t_lst[1])
    else:
        print('failed to parse time obj, using 00:00')
    return tm(hour, minute)

def getActiveTimeBuckets(tb_lst, time_lst):
    result = []
    for tb in tb_lst:
        if len([t for t in time_lst if (t > tb[0] and t <= tb[1])]):
            result.append(tb)
    return result

def convertTimeBuckets(tbs: str,coffeeBreak: bool = False):
    t_buckets = tbs.split(' ')
    if coffeeBreak & ('10:15' not in t_buckets) & ('10:30' not in t_buckets) & (len(t_buckets)>=2):
        t_buckets_n = [t_buckets[0], '10:15', '10:30', t_buckets[1]]
        t_buckets_n.extend(t_buckets[2:])
        t_buckets = t_buckets_n
    result = []
    if (len(t_buckets) % 2 == 0) & (len(t_buckets) > 0):
        for i in range(int(len(t_buckets)/2)):
            start_t = parseTime(t_buckets.pop(0))
            end_t = parseTime(t_buckets.pop(0))
            if start_t > end_t:
                result.append((start_t, tm(23,59)))
                result.append((tm(0,0), end_t))
            else:
                result.append((start_t,end_t))
    else:
        print('time buckets input issue:{0}'.format(tbs))
    return result

def generateDateTimeKey(tb_lst, current_date: date, start_datetime: dt, end_datetime: dt):
    result = []
    tdelta_arry = np.arange(1,360)
    for tp in tb_lst:
        dt_s = dt.combine(current_date,tp[0])
        dt_e= dt.combine(current_date,tp[1])
        #if dt_s > dt_e:
        #    dt_s = dt.combine(current_date,tm(0,0))
            
        dt_keys = [dt_s + td(minutes = int(tdelta))\
                   for tdelta in tdelta_arry \
                   if ((dt_s + td(minutes = int(tdelta))) <= dt_e)]
        result.extend(dt_keys)
    result = [r for r in result if (r>=start_datetime and r<=end_datetime)]
    return sorted(result)

def parseTimeBucketsCol(row):
    if row["交易所名称"] == "中国金融期货交易所":
        return convertTimeBuckets(row["trade_time_buckets"])
    else:
        return convertTimeBuckets(row["trade_time_buckets"],True)

def read_trading_info_df(fut_trading_time_path: str = ".//dataset//future_base_info//"):
    fut_info_sina_path = os.path.join(fut_trading_time_path,'sina')
    latest_info_file = get_latest_file(fut_info_sina_path)
    latest_t_hours_file = get_latest_file(fut_trading_time_path)
    fut_info_df = pd.read_csv(latest_info_file)
    fut_hours_df = pd.read_csv(latest_t_hours_file)
    fut_hours_df = fut_hours_df.fillna('')
    fut_info_aug_df = fut_info_df.merge(fut_hours_df[['generic_ticker_sina','trading_hours','trade_time_buckets']],
                                        how = 'left', on='generic_ticker_sina')
    fut_info_aug_df['time_buckets_obj'] = fut_info_aug_df.apply(parseTimeBucketsCol, axis = 1)
    return fut_info_aug_df