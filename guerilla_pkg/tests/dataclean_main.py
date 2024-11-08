import pandas as pd
import guerilla.core.config as cfg
import os
import argparse
from datetime import datetime
from guerilla.core.utility import logger
from guerilla.core.constants import d_format_long,d_format_short
from guerilla.dataservice.dsUtil import save_to_csv
from guerilla.datamodel.dataclean import tsFillEngine,tsFilterEngine


def main():
    log = logger(__name__,'dataclean.log')
    config_ = cfg.config_parser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--fill', dest='fill', action='store_true')
    parser.add_argument('--filter', dest='filter', action='store_true')
    parser.add_argument("-s", "--source",type = str, choices=['sina', 'rq'], default='sina', help="input data source")
    parser.add_argument("-b", "--begin",type = str, default='2000-01-01', help="input data source")
    parser.add_argument("-e", "--end",type = str, help="input data source")
    
    parser.set_defaults(end = datetime.now().strftime(d_format_short))
    parser.set_defaults(dofill=False)
    parser.set_defaults(dofilter=False)
    args = parser.parse_args()
    src = args.source
    begin_datetime = datetime.strptime(args.begin, d_format_short)
    end_datetime = datetime.strptime(args.end, d_format_short)
    base_path = config_["dataset"]["base_path"]
    data_path = os.path.join(base_path, config_["dataset"]["fut_ct_folder"],src,'1')
    fut_info_path = os.path.join(base_path,config_["dataset"]["fut_info_folder"])
    output_data_path = os.path.join(base_path, config_["dataset"]["fut_ct_folder"],src + '_clean','1')
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
    
    if args.fill:
        process_ts_filling(data_path,output_data_path,fut_info_path,begin_datetime,end_datetime)
    elif args.filter:
        process_ts_filter(data_path,output_data_path,fut_info_path,begin_datetime,end_datetime)
    else:
        process_ts_all(data_path,output_data_path,fut_info_path,begin_datetime,end_datetime)

def process_ts_filling(raw_data_path:str,
                       output_data_path:str,
                       fut_info_base_path:str,
                       s_datetime:datetime,
                       e_datetime:datetime):
    
    pass

def process_ts_filter(raw_data_path:str,
                       output_data_path:str,
                       fut_info_base_path:str,
                       s_datetime:datetime,
                       e_datetime:datetime):
    tsfilter_engine = tsFilterEngine(volFilter=True,timeFilter=True,time_filter_list=['09:01','21:01'])
    ticker = 'AG2204'
    raw_data_df = pd.read_csv(os.path.join(raw_data_path,ticker + '.csv'))
    clean_data_df = tsfilter_engine.process(raw_data_df)
    save_to_csv(os.path.join(output_data_path,ticker + '_test.csv'), clean_data_df)

def process_ts_all(raw_data_path:str,
                       output_data_path:str,
                       fut_info_base_path:str,
                       s_datetime:datetime,
                       e_datetime:datetime):
    tsfill_engine = tsFillEngine(start_time=s_datetime,end_time=e_datetime)
    tsfill_engine.load_trading_time(fut_info_base_path)
    
    dir_list = os.listdir(raw_data_path)
    ticker_list = [f.split('.')[0] for f in dir_list]
    
    output_dir_list = os.listdir(output_data_path)
    ticker_done_list = [f.split('.')[0] for f in output_dir_list]
    ticker_to_do_list = [t for t in ticker_list if t not in ticker_done_list]
    for ticker in ticker_to_do_list:
        raw_data_df = pd.read_csv(os.path.join(raw_data_path,ticker + '.csv'))
        clean_data_df = tsfill_engine.process(raw_data_df, ticker)
        save_to_csv(os.path.join(output_data_path,ticker + '.csv'), clean_data_df)

if __name__ == '__main__':
     # execute only if run as a script
     main()