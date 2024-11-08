import os
import multiprocessing
import pandas as pd
from datetime import datetime
from datetime import timedelta as td
from guerilla.dataservice.dsUtil import SOURCE,FREQ,fetch_ticker_data,map_ticker_custom,save_to_csv
from guerilla.core.utility import logger
import guerilla.core.config as cfg
from .dmutil import STITCH_CODE

def calc_dom_start(row,sym_col):
    result = 0
    if row[sym_col] != row[sym_col + '_fwd']:
        result = 1
    return result
def calc_dom_end(row,sym_col):
    result = 0
    if row[sym_col] != row[sym_col + '_bwd']:
        result = 1
    return result
def fetch_dom_start_open(row):
    open_price = 0.0
    date_target = row['dom_start']
    ticker = row['und_contract'].split('.')[1]
    ct_df,_= fetch_ticker_data(ticker,FREQ.DAILY,SOURCE.TQ)
    if len(ct_df) >0:
        ct_df['datekey'] = ct_df['datetime'].apply(lambda d:d.date())
        if date_target not in ct_df['datekey'].values:
            date_target = ct_df['datekey'].min()
        fetchs = ct_df[ct_df['datekey'] ==date_target]['open'].values
        if len(fetchs)>0:
            open_price = fetchs[0]
    return open_price    

def fetch_dom_end_close(row):
    close_price = 0.0
    date_target = row['dom_end']
    ticker = row['und_contract'].split('.')[1]
    ct_df,_= fetch_ticker_data(ticker,FREQ.DAILY,SOURCE.TQ)
    if len(ct_df) >0:
        ct_df['datekey'] = ct_df['datetime'].apply(lambda d:d.date())
        if date_target not in ct_df['datekey'].values:
            date_target = ct_df['datekey'].max()
        fetchs = ct_df[ct_df['datekey'] ==date_target]['close'].values
        if len(fetchs)>0:
            close_price = fetchs[0]
    return close_price


class genfuture:
    def __init__(self) -> None:
        self.logger_ = logger(__name__,"genfuture.log")
        self.config_ = cfg.config_parser()
        self.output_path =os.path.join(self.config_["dataset"]["base_path"], self.config_["dataset"]["fut_gen_folder"])
        self.output_base_info_path = os.path.join(self.config_["dataset"]["base_path"], self.config_["dataset"]["fut_info_folder"])
        self.ct_map_df = self.get_all_contract_history()
    
    def build_dominant_allmarket(self, f:FREQ,stitch_type:STITCH_CODE = STITCH_CODE.VANILLA, start_date:str = '2016-01-01'):
        symbols = [c for c in self.ct_map_df.columns if c not in ['date','datekey']]
        for s in symbols:
            self.build_dominant(s,f,stitch_type, start_date)
    
    def build_dominant_allmarket_multiproc(self, f:FREQ,stitch_type:STITCH_CODE = STITCH_CODE.VANILLA, start_date:str = '2016-01-01'):
        self.logger_.info('generic future build started for freq:{0};stitching method:{1};start_date:{2}'.format(f,stitch_type.value,start_date))
        symbols = [c for c in self.ct_map_df.columns if c not in ['date','datekey']]
        num_workers = multiprocessing.cpu_count()  # Get the number of CPU cores
        chunk_size = len(symbols) // num_workers
        chunks = [symbols[i:i+chunk_size] for i in range(0, len(symbols), chunk_size)]
        self.logger_.info('Total contracts to be processed: {0}'.format(len(symbols)))
        self.logger_.info('Total chunks: {0}'.format(len(chunks)))
        processes = []
        for i in range(num_workers):
            p = multiprocessing.Process(target=self.build_dominant_worker, args=(i,chunks[i],f,stitch_type,start_date))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        self.logger_.info('generic future build successed for freq:{0};stitching method:{1};start_date:{2}'.format(f,stitch_type.value,start_date))
    
    def build_dominant_worker(self,worker_id:int,symbols:list[str], f:FREQ, stitch_type:STITCH_CODE, start_date:str):
        log = logger(__name__,'app-tqwkr' + str(worker_id) + '.log')
        log.info(f"Worker {worker_id} started.")
        log.info(f"contracts: {symbols}")
        log.info(f"frq: {f}")
        for s in symbols:
            try:
                self.build_dominant(s,f,stitch_type, start_date)
                log.info('generic contract {0}, {1} done.'.format(s,f))
            except Exception as e:
                log.error('generic contract {0}, {1} failed.'.format(s,f))
                log.error(e)
                continue
        log.info(f"Worker {worker_id} finished.")
    
    def build_dominant(self, symbol_tq:str, f:FREQ, stitch_type:STITCH_CODE = STITCH_CODE.VANILLA, start_date:str = '2016-01-01'):
        
        output_file_dir = os.path.join(self.output_path,f.value)
        ticker = symbol_tq.split('.')[1]
        file_name = ticker.upper() + '88' + stitch_type.value + '.csv'
        if not os.path.exists(output_file_dir):
            os.makedirs(output_file_dir)
            
        if stitch_type == STITCH_CODE.VANILLA:
            self.build_dominant_vanilla(symbol_tq,f,start_date, os.path.join(output_file_dir, file_name))
        else:
            pass
    
    def build_dominant_vanilla(self, symbol_tq:str, f:FREQ, start_date:str, out_dir:str):
        #we assume symbol_tq has format EXCHANGE.TICKER
        tgt_sym =[c for c in self.ct_map_df.columns if (symbol_tq.lower() == c.lower())]
        
        if len(tgt_sym) <1:
            return False
        sym_col = tgt_sym[0]
        gen_ticker = sym_col.split('.')[1]
        ct_map = self.ct_map_df[self.ct_map_df['datekey']>datetime.strptime(start_date,"%Y-%m-%d").date()].copy()
        ct_map = ct_map[['datekey',sym_col]]
        ct_list = ct_map[sym_col].dropna().drop_duplicates().to_list()
        fut_concat_df = []

        for ct in ct_list:
            exch = ct.split('.')[0]
            ticker = ct.split('.')[1]
            ct_df = self.fetch_und_ticker(ticker,exch, f)
            if len(ct_df) >1:
                ct_df = ct_df.drop_duplicates(subset = ['datetime'],keep = 'last')
                ct_df['datekey']=ct_df['datetime'].apply(lambda d: d.date())
                fut_df = ct_map.merge(ct_df,left_on=['datekey',sym_col],right_on=['datekey','symbol'])
                if len(fut_df):
                    fut_df['trading_date'] = fut_df['datekey']
                    fut_df['und_ticker'] = fut_df['symbol']
                    fut_df['generic_ticker'] = gen_ticker.upper()
                    fut_df = fut_df[['datetime','generic_ticker','open_oi','close_oi',\
                                    'open','close','high','low','volume',\
                                    'trading_date','und_ticker']].dropna()
                    fut_concat_df.append(fut_df)
            else:
                self.logger_.info('missing contract {0}'.format(ct))
        if len(fut_concat_df):
            fut_concat_df = pd.concat(fut_concat_df)
            save_to_csv(out_dir,fut_concat_df)
            self.logger_.info("{0} done".format(sym_col))
        else:
            self.logger_.info("{0} failed".format(sym_col))
        return fut_concat_df
    
    def calc_roll_adjust_allmarket(self,start_date:str = '2016-01-01'):
        
        output_file_dir = os.path.join(self.output_base_info_path,SOURCE.TQ.value)
        file_name = 'fut_roll_master.csv'
        if not os.path.exists(output_file_dir):
            os.makedirs(output_file_dir)
            
        ct_map_df = self.get_all_contract_history()
        roll_master_df = []
        for c in ct_map_df.columns:
            if c != 'date' and c != 'datekey':
                print('processing {0}'.format(c))
                r_df = self.calc_roll_adjust(symbol_tq=c,start_date=start_date)
                roll_master_df.append(r_df)
        roll_master_df = pd.concat(roll_master_df)
        
        roll_master_df.to_csv(os.path.join(output_file_dir, file_name),index = False)
    
    def calc_roll_adjust(self, symbol_tq:str,start_date:str):
        ct_map_df = self.get_all_contract_history()
        ct_map_df = ct_map_df[ct_map_df['datekey']>datetime.strptime(start_date,"%Y-%m-%d").date()]
        ct_map_df = ct_map_df[['datekey',symbol_tq]].copy()
        ct_map_df = ct_map_df.dropna()
        ct_map_df[symbol_tq + '_fwd']= ct_map_df[symbol_tq].shift(1)
        ct_map_df[symbol_tq + '_bwd']= ct_map_df[symbol_tq].shift(-1)
        ct_map_df['dom_start_label'] = ct_map_df.apply(calc_dom_start,args=(symbol_tq,),axis=1)
        ct_map_df['dom_end_label'] = ct_map_df.apply(calc_dom_end,args=(symbol_tq,),axis=1)
        dom_start_df = ct_map_df.loc[ct_map_df['dom_start_label'] == 1,[symbol_tq,'datekey']].copy()
        dom_end_df = ct_map_df.loc[ct_map_df['dom_end_label'] == 1,[symbol_tq,'datekey']].copy()
        adj_master_df = dom_start_df.merge(dom_end_df,left_on=[symbol_tq], right_on = [symbol_tq])
        adj_master_df.columns = ['und_contract','dom_start','dom_end']
        adj_master_df['generic_ticker'] = symbol_tq.split('.')[1]
        adj_master_df['exchange'] = symbol_tq.split('.')[0]
        adj_master_df['open'] = adj_master_df.apply(fetch_dom_start_open, axis=1)
        adj_master_df['close'] = adj_master_df.apply(fetch_dom_end_close, axis=1)
        adj_master_df = adj_master_df.sort_values(by='dom_start',ascending=False)
        adj_master_df['diff'] = adj_master_df['open'].shift(1) - adj_master_df['close']
        adj_master_df['diff'] = adj_master_df['diff'].fillna(0)
        adj_master_df['ratio'] = adj_master_df['open'].shift(1)/adj_master_df['close']
        adj_master_df['ratio'] = adj_master_df['ratio'].fillna(1)
        adj_master_df['diff_adj'] = adj_master_df['diff'].cumsum()
        adj_master_df['ratio_adj'] = adj_master_df['ratio'].cumprod()
        return adj_master_df
    
    def fetch_und_ticker(self,symbol:str, exch:str, f:FREQ):
        #source the underlying from different data provider and consolidate.
        result_df = pd.DataFrame()
        if f == FREQ.MINUTE1:
            #use rq data first
            concat_df=[]
            tb_df,m = fetch_ticker_data(map_ticker_custom(symbol,exch),f,SOURCE.TB)
            if len(tb_df):
                tb_df['symbol'] = exch + '.' + symbol
                tb_df = tb_df[['datetime','open','high','low','close','volume','open_oi','close_oi','symbol']]
                concat_df.append(tb_df)
            #then combine tq data
            tq_df,_ = fetch_ticker_data(map_ticker_custom(symbol,exch),f,SOURCE.TQ)
            if len(tq_df):
                tq_df = tq_df[['datetime','open','high','low','close','volume','open_oi','close_oi','symbol']]
                concat_df.append(tq_df)
            if len(concat_df):
                result_df = pd.concat(concat_df)
                result_df = result_df.drop_duplicates(subset=['datetime'], keep='last')
        else:
            result_df,_ = fetch_ticker_data(map_ticker_custom(symbol,exch),f,SOURCE.TQ)
        
        return result_df
    
    def get_all_contract_history(self):
        ct_map_df = pd.DataFrame()
        contract_hist_tq_path = os.path.join(self.config_["dataset"]["base_path"], self.config_["dataset"]["fut_info_folder"],"tq")
        file_ct_map = os.path.join(contract_hist_tq_path,'dominant_contract_hist.csv')
        if not os.path.exists(os.path.join(contract_hist_tq_path,'dominant_contract_hist.csv')):
            self.logger_.error("exit, can not locate tq contract map")
            return ct_map_df
        
        ct_map_df = pd.read_csv(file_ct_map)
        ct_map_df['datekey'] = ct_map_df['date'].apply(lambda d:datetime.strptime(d,"%Y-%m-%d").date())
        return ct_map_df
    
    