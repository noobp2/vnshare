import argparse
import config as cfg
from dataservice.dsUtil import FREQ,save_to_vn_sqlite,fetch_ticker_data,read_from_vn_sqlite
from vnpy.trader.constant import Interval
from datetime import datetime,timedelta
from core.utility import logger
from core.constants import d_format_long,d_format_short

def main():
    log = logger(__name__,'vndata_load.log')
    parser = argparse.ArgumentParser()
    freq_choices =[e.value for e in [FREQ.MINUTE1,FREQ.HOURLY,FREQ.DAILY, FREQ.WEEKLY, FREQ.MONTHLY]]
    parser.add_argument("-b", "--begin",type = str, default='2016-01-01', help="input data source")
    parser.add_argument("-f","--freq",type = str, choices=freq_choices,default = '1d', help="input data frequency")
    begin_default = (datetime.now() - timedelta(weeks=15)).strftime(d_format_short)
    parser.set_defaults(begin = begin_default)
    args = parser.parse_args()
    
    config = cfg.config_parser()
    tickers = config["research"]["china_fut_universe"]
    log.info('start loading data for tickers:{0},frequency:{1}'.format(tickers,args.freq))

    save_to_vn_sqlite(tickers,start_date=args.begin,freq=FREQ(args.freq))
    log.info('done loading data.')

if __name__ == '__main__':
    main()