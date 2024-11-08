from guerilla.datamodel.tsgeneric import genfuture
from guerilla.dataservice.dsUtil import FREQ,SOURCE,save_to_vn_sqlite,fetch_ticker_data
from guerilla.core.utility import logger
from guerilla.core.constants import d_format_long,d_format_short
import pandas as pd
from datetime import datetime,timedelta
import os
import guerilla.core.config as cfg
import os
import argparse

def main():
    freq_choices = [e.value for e in FREQ]
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--begin",type = str, default='2016-01-01', help="input data source")
    parser.add_argument("-f","--freq",type = str, choices=freq_choices, help="input data frequency")
    begin_default = (datetime.now() - timedelta(weeks=15)).strftime(d_format_short)
    parser.set_defaults(begin = begin_default)
    frq = 'all'
    args = parser.parse_args()
    if args.freq:
        frq = args.freq
    
    gen_fut = genfuture()
    
    if frq in [FREQ.MINUTE1.value, 'all']:
        gen_fut.build_dominant_allmarket(f=FREQ.MINUTE1,start_date=args.begin)
    
    if frq in [FREQ.MINUTE5.value, 'all']:
        gen_fut.build_dominant_allmarket(f=FREQ.MINUTE5,start_date=args.begin)
    
    if frq in [FREQ.MINUTE15.value, 'all']:
        gen_fut.build_dominant_allmarket(f=FREQ.MINUTE15,start_date=args.begin)
        
    if frq in [FREQ.MINUTE30.value, 'all']:
        gen_fut.build_dominant_allmarket(f=FREQ.MINUTE30,start_date=args.begin)
        
    if frq in [FREQ.HOURLY.value, 'all']:
        gen_fut.build_dominant_allmarket(f=FREQ.HOURLY,start_date=args.begin)
    
    if frq in [FREQ.DAILY.value, 'all']:
        gen_fut.build_dominant_allmarket(f=FREQ.DAILY,start_date=args.begin)
    

if __name__ == '__main__':
    main()