import guerilla.core.config as cfg
import os
import pandas as pd
from datetime import datetime
from datetime import timedelta
import argparse
from guerilla.core.utility import logger
from guerilla.core.constants import d_format_short

import guerilla.dataservice.source.sina as ds
import guerilla.dataservice.source.yahoo as dy
import guerilla.dataservice.source.tq as tq
from guerilla.dataservice.dsUtil import SOURCE, FREQ, fetch_sina_fut_baseinfo


def main():
    log = logger(__name__,'app.log')
    parser = argparse.ArgumentParser()
    freq_choices = [e.value for e in FREQ]
    parser.add_argument("-s", "--source",type = str, choices=['sina', 'rq', 'yahoo', 'tq'], help="input data source")
    parser.add_argument("-f","--freq",type = str, choices=freq_choices, help="input data frequency")
    parser.add_argument("-g","--group",type = str, choices=['g1','g2','all'], help="input data frequency")
    args = parser.parse_args()
    src = SOURCE.SINA
    frq = 'all'
    grp = 'all'
    if args.source:
        src = SOURCE(args.source)
    if args.freq:
        frq = args.freq
    if args.group:
        grp = args.group

    if (src == SOURCE.SINA):
        fetch_sina(frq)
    elif (src == SOURCE.YH):
        fetch_yahoo()
    elif (src == SOURCE.TQ):
        fetch_tq(frq,grp=grp)
    else:
        log.info("source not implemented.")

def fetch_sina(frq:str = "all", config = cfg.config_parser(), sleep_time = 3):
    dswkr = ds.worker()
    log = logger(__name__,'app.log')
    log.info('app started')
    dswkr.start()
    #fetch future base information
    log.info('base future info fetch started...')
    base_fut_info_path = os.path.join(config["dataset"]["base_path"],\
        config["dataset"]["fut_info_folder"])
    s, futures_comm_info_df = dswkr.fetch_future_basic_info(basepath=base_fut_info_path)
    log.info('base future info fetch finished.')
    
    generic_future_sina_lst = futures_comm_info_df['generic_ticker_sina'].drop_duplicates().to_list()
    base_fut_gen_path = os.path.join(config["dataset"]["base_path"],\
        config["dataset"]["fut_gen_folder"])
    contract_fut_sina_lst=futures_comm_info_df[(futures_comm_info_df.is_active)]['contract_code_sina'].drop_duplicates().to_list()
    base_fut_ct_path = os.path.join(config["dataset"]["base_path"],\
        config["dataset"]["fut_ct_folder"])
    
    #fetch 1 min generic fut contract from sina
    if frq in [FREQ.MINUTE1.value, 'all']:
        log.info('1 min generic future history fetch started...')
        for ticker in generic_future_sina_lst:
            s = dswkr.fetch_future_hist_data(ticker,freq=FREQ.MINUTE1.value,basepath=base_fut_gen_path,slp=sleep_time)
        log.info('1 min generic future history fetch finished.')
        
        #fetch 1 min fut contract from sina
        log.info('1 min contract future history fetch started...')
        for ticker in contract_fut_sina_lst:
            s = dswkr.fetch_future_hist_data(ticker,freq=FREQ.MINUTE1.value,basepath=base_fut_ct_path,slp=sleep_time)
        log.info('1 min contract future history fetch finished.')
    
    if frq in [FREQ.DAILY.value,'all']:
        #fetch 1d generic fut contract from sina
        log.info('1d generic future history fetch started...')
        for ticker in generic_future_sina_lst:
            s = dswkr.fetch_future_hist_data(ticker,freq=FREQ.DAILY.value,basepath=base_fut_gen_path,slp=sleep_time)
        log.info('1d generic future history fetch finished.')
        
        #fetch 1d fut contract from sina
        log.info('1 day contract future history fetch started...')
        for ticker in contract_fut_sina_lst:
            s = dswkr.fetch_future_hist_data(ticker,freq=FREQ.DAILY.value,basepath=base_fut_ct_path,slp=sleep_time)
        log.info('1 day contract future history fetch finished.')
    
    if frq in [FREQ.MINUTE5.value,'all']:
        #fetch 5 min generic fut contract from sina
        log.info('5 min generic future history fetch started...')
        for ticker in generic_future_sina_lst:
            s = dswkr.fetch_future_hist_data(ticker,freq=FREQ.MINUTE5.value,basepath=base_fut_gen_path,slp=sleep_time)
        log.info('5 min generic future history fetch finished.')

    if frq in [FREQ.MINUTE15.value,'all']:
        #fetch 15 min generic fut contract from sina
        log.info('15 min generic future history fetch started...')
        for ticker in generic_future_sina_lst:
            s = dswkr.fetch_future_hist_data(ticker,freq=FREQ.MINUTE15.value,basepath=base_fut_gen_path,slp=sleep_time)
        log.info('15 min generic future history fetch finished.')
    
    if frq in [FREQ.MINUTE30.value,'all']:
        #fetch 15 min generic fut contract from sina
        log.info('30 min generic future history fetch started...')
        for ticker in generic_future_sina_lst:
            s = dswkr.fetch_future_hist_data(ticker,freq=FREQ.MINUTE30.value,basepath=base_fut_gen_path,slp=sleep_time)
        log.info('30 min generic future history fetch finished.')
    
    if frq in [FREQ.HOURLY.value,'all']:
        #fetch 15 min generic fut contract from sina
        log.info('1h generic future history fetch started...')
        for ticker in generic_future_sina_lst:
            s = dswkr.fetch_future_hist_data(ticker,freq=FREQ.HOURLY.value,basepath=base_fut_gen_path,slp=sleep_time)
        log.info('1h generic future history fetch finished.')
    
    dswkr.stop()
    log.info('app finished.')

def fetch_yahoo(config = cfg.config_parser(), sleep_time = 1):
    
    base_path = os.path.join(config["dataset"]["base_path"],config["dataset"]["ticker_gen_folder"])
    master_path = os.path.join(base_path,'yahoo')
    master_file = os.path.join(master_path,config["dataset"]["yh_ticker_list_file"])
    end_date = datetime.now()
    start_date = datetime.strptime('2000-01-01',d_format_short)
    
    dywkr = dy.worker()
    #set default master file if not exist
    if not os.path.exists(master_path):
        os.makedirs(master_path)
    
    if not os.path.exists(master_file):
        dywkr.create_master_file(master_file)
    
    tickers = dywkr.get_master_list(master_file)
     
    log = logger(__name__,'app.log')
    log.info('fetch 1d timeseries from yahoo started..')
    dywkr.start()
    for ticker in tickers:
        s = dywkr.fetch_daily_hist_data(ticker=ticker,
                                basepath=base_path,
                                start_date=start_date,
                                end_date=end_date,
                                slp=sleep_time)
    dywkr.stop()
    log.info('fetch 1d timeseries from yahoo finished.')

def fetch_tq(frq:str = "all", config = cfg.config_parser(), sleep_time = 3, grp:str = 'all'):
    log = logger(__name__,'app.log')
    log.info('app started')
    #fetch future base information
    dswkr = ds.worker()
    dswkr.start()
    log.info('base future info fetch started...')
    base_fut_info_path = os.path.join(config["dataset"]["base_path"],\
        config["dataset"]["fut_info_folder"])
    s, futures_comm_info_df = dswkr.fetch_future_basic_info(basepath=base_fut_info_path)
    log.info('base future info fetch finished.')
    dswkr.stop()
    
    contract_fut_tq_lst=list(zip(futures_comm_info_df["合约代码"],futures_comm_info_df["交易所名称"]))
    
    lookback = 40
    date_now = datetime.now()
    for d in range(lookback):
        date_check = datetime.strftime(date_now - timedelta(days=lookback - d),d_format_short)
        file_path = os.path.join(base_fut_info_path,'sina','future_info_' + date_check + '.csv')
        contract_extr_df = fetch_sina_fut_baseinfo(file_path=file_path)
        if len(contract_extr_df):
            contract_lst = list(zip(contract_extr_df["合约代码"],contract_extr_df["交易所名称"]))
            contract_fut_tq_lst.extend(contract_lst)
            contract_fut_tq_lst = list(set(contract_fut_tq_lst))

    base_fut_ct_path = os.path.join(config["dataset"]["base_path"],config["dataset"]["fut_ct_folder"])
    
    u = config["dataprovider"]["tq"]["user"]
    p = config["dataprovider"]["tq"]["pwd"]
    tqwkr = tq.worker(user_name=u,password=p)
    tqwkr.start()
    
    if grp == 'g1':
        #dominant contract only
        contracts_df = pd.read_csv(os.path.join(base_fut_info_path,'tq','dominant_contract_hist.csv'))
        c_dict = contracts_df.tail(1).iloc[0].to_dict()
        contract_fut_tq_lst = [(v.split('.')[1],v.split('.')[0]) for k,v in c_dict.items() if k != 'date']
    
    if frq == 'month':
        s = tqwkr.fetch_dominant_contract_map(contracts_list=contract_fut_tq_lst,basepath=base_fut_info_path)
        tqwkr.stop()
        return

    if frq == 'all':
        for f in FREQ:
            log.info('{0} contract future history fetch started...'.format(f.value))
            for (t,ex) in contract_fut_tq_lst:
                s = tqwkr.fetch_future_hist_data(ticker=t,exch=ex, freq=f,basepath=base_fut_ct_path,slp=sleep_time)
            log.info('{0} contract future history fetch finished.'.format(f.value))
    else:
        log.info('{0} contract future history fetch started...'.format(frq))
        for (t,ex) in contract_fut_tq_lst:
            s = tqwkr.fetch_future_hist_data(ticker=t,exch=ex, freq=FREQ(frq),basepath=base_fut_ct_path,slp=sleep_time)
        log.info('{0} contract future history fetch finished.'.format(frq))
    
    tqwkr.stop()
    log.info('app finished.')
    
if __name__ == '__main__':
     # execute only if run as a script
     main()