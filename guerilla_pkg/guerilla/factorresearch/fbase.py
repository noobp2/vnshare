from math import log10
from pickle import DICT
import pandas as pd
import numpy as np
import alphalens as al
from typing import List
from datetime import datetime,timedelta
from pytz import timezone
from guerilla.dataservice.dsUtil import FREQ, SOURCE, save_to_csv,fetch_ticker_data
from .futil import *
from guerilla.core.utility import logger
from guerilla.core.objects import FactorData,FTYPE
from guerilla.core.qrdb import SqliteDatabase
from vnpy.trader.utility import virtual
from vnpy.trader.constant import Exchange
from enum import Enum
import guerilla.core.config as cfg
import os

class FCOL(Enum):
    DT = 'datetime'
    PL = 'price_level'
    FL = 'factor_level'
    EP = 'entry_price'
    IND = 'indicator'
    TK = 'ticker'
    GP = 'group'

class FDBCOL(Enum):
    DT = 'datetime'
    SB = 'symbol'
    EXCH = 'exchange'
    FR = 'frequency'
    FT = 'factor_type'
    FN = 'factor_name'
    FV = 'factor_value'
    
def parserFactorData(data_df:pd.DataFrame,cut_off_date = datetime(2000,1,1), tz_str:str = 'Asia/Shanghai') -> List[FactorData]:
    tz = timezone(tz_str)
    factor_list:List[FactorData] = []
    for index, row in data_df.iterrows():
        factor_data = FactorData(symbol=row[FDBCOL.SB.value],
                                 exchange=Exchange(row[FDBCOL.EXCH.value]),
                                 datetime=row[FDBCOL.DT.value].to_pydatetime(),
                                 frequency=FREQ(row[FDBCOL.FR.value]),
                                 factor_type=FTYPE(row[FDBCOL.FT.value]),
                                 factor_name=row[FDBCOL.FN.value],
                                 factor_value=row[FDBCOL.FV.value])
        if factor_data.datetime.date() > cut_off_date.date():
            factor_list.append(factor_data)
    
    return factor_list

class FactorBase:
    def __init__(self,vn_ticker:str,f:FREQ = FREQ.DAILY, s:SOURCE = SOURCE.DEF, group:str = "G1", tz:str = 'Asia/Shanghai') -> None:
        self.factor_name = __name__
        self.freq_:FREQ = f
        self.ticker = vn_ticker
        self.exchange = 'unknown'
        
        if vn_ticker.find('.') > 0:
            self.exchange = vn_ticker.split('.')[1]
            self.ticker = vn_ticker.split('.')[0]
            
        self.tz_ = tz
        self.group = group
        self.input_df = pd.DataFrame()
        self.output_df = pd.DataFrame()
        self.factordata_df = pd.DataFrame()
        self.data_engine:FactorDataEngine = FactorDataEngine(f,s)
        self.logger_ = logger(__name__)
    
    def init_data(self):
        #init data
        self.input_df = self.data_engine.get_input_data(self.ticker)
        if 'date' in self.input_df.columns:
            self.input_df.rename(columns = {'date':'datetime'}, inplace = True)
        
        exch = self.exchange
        if 'und_ticker' in self.input_df.columns:
            exch = self.input_df['und_ticker'].iloc[0].split('.')[0]
        elif 'symbol' in self.input_df.columns:
            exch = self.input_df['symbol'].iloc[0].split('.')[0]
        else:
            pass
        
        self.input_df[FDBCOL.EXCH.value] = exch
        self.input_df[FDBCOL.SB.value] = self.ticker
        self.input_df[FDBCOL.FR.value] = self.freq_
        self.output_df = pd.DataFrame()
    
    @virtual
    def setup_params(self, params:DICT) -> None:  # type: ignore
        pass
    
    @virtual
    def calc_factor(self) -> None:
        pass
    
    @virtual
    def calc_indicator(self) -> None:
        pass
    
    def save_factor(self):
        self.data_engine.save_factor_data(self.output_df,self.ticker, self.factor_name)
    
    def save_factor_to_db(self,window_days:int = 3650):
        self.data_engine.save_factor_data_db(self.factordata_df, self.tz_, window_days)
    
    def get_factor(self):
        self.output_df = self.data_engine.get_factor_data(self.factor_name, self.ticker)
    
    def get_factor_from_db(self,factor_names:List[str],
                           start_date:datetime = datetime(2016,1,1),
                           end_date:datetime = datetime.now()) -> pd.DataFrame:
        factor_df = pd.DataFrame()
        factor_df = self.data_engine.get_factor_data_db(self.ticker,
                                                                 factor_names,
                                                                 start_date,
                                                                 end_date)
        factor_df = factor_df.pivot(index=FDBCOL.DT.value, columns=FDBCOL.FN.value, values=FDBCOL.FV.value)
        factor_df[FDBCOL.SB.value] = self.ticker
        factor_df[FDBCOL.EXCH.value] = self.exchange
        factor_df = factor_df.dropna()
        return factor_df
    
    def get_visual_data(self, start_date:datetime = datetime(2010,1,1), end_date:datetime = datetime.now()):
        self.output_df[FCOL.TK.value] = self.ticker
        self.output_df[FCOL.GP.value] = self.group
        if self.freq_ == FREQ.DAILY:
            self.output_df[FCOL.DT.value] = self.output_df[FCOL.DT.value].apply(lambda x:datetime(x.year,x.month, x.day))

        return self.output_df[(self.output_df[FCOL.DT.value] >= start_date) & (self.output_df[FCOL.DT.value] <= end_date)]

class XsectorFactorsBase:
    def __init__(self) -> None:
        pass    
        
class FactorDataEngine:
    def __init__(self, f:FREQ, s:SOURCE) -> None:
        self.freq_:FREQ = f
        self.source_:SOURCE = s
        self.logger_ = logger(__name__)
        self.config_ = cfg.config_parser()
        
    def get_input_data(self, ticker):
        data_df,m =  fetch_ticker_data(ticker, self.freq_, self.source_)
        
        if 'success' not in m:
            self.logger_.warning(m)
        
        return data_df
    
    def get_factor_data(self, factor_name, ticker):
        base_path = self.config_["research"]["base_path"]
        filepath = os.path.join(base_path,factor_name,self.freq_)       # type: ignore
        filename = os.path.join(filepath, ticker + '.csv')
        
        result = pd.DataFrame()
        if os.path.exists(filename):
            result = pd.read_csv(filename)
        else:
            self.logger_.warning("file not found for factor {0}, ticker {1}".format(factor_name, ticker))
        return result
    
    def save_factor_data(self, data_df:pd.DataFrame, ticker:str,factor_name:str):
        base_path = self.config_["research"]["base_path"]
        filepath = os.path.join(base_path,factor_name,self.freq_)  # type: ignore
        
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        filename = os.path.join(filepath, ticker + '.csv')
        isdaily = False
        if self.freq_ == FREQ.DAILY or self.freq_ == FREQ.WEEKLY:
            isdaily = True
        save_to_csv(fileName=filename, data_df=data_df,is_daily= isdaily)
    
    def save_factor_data_db(self, data_df:pd.DataFrame, tz:str = 'Asia/Shanghai',window_days:int = 3650):
        db_engine = SqliteDatabase()
        cut_off_date = datetime.now(timezone(tz)) - timedelta(days = window_days)
        fdata = parserFactorData(data_df,cut_off_date=cut_off_date,tz_str= tz)
        db_engine.save_factor_data(fdata)
        db_engine.close()
        return
    
    def get_factor_data_db(self, ticker: str,
                     factor_names:List[str],
                     start_date:datetime,
                     end_date:datetime)-> pd.DataFrame:
        db_engine = SqliteDatabase()
        result_df = db_engine.load_factor_data_multi_df(symbols= [ticker],
                                                        factor_names=factor_names,
                                                        frequency=self.freq_,
                                                        start= start_date,
                                                        end= end_date)
        db_engine.close()
        return result_df

class FactorVisualEngine:
    def __init__(self,
                 num_q:int = 5,
                 horizon = (1,5,10),
                 multi_ticker:bool = False,
                 bin_type:str = "quantile",
                 static_bin = [],
                 f:FREQ = FREQ.DAILY,
                 dmean:bool = False) -> None:
        self.num_quantiles = num_q
        self.bin_type = bin_type
        self.horizon = horizon
        self.factor_data = pd.DataFrame()
        self.factor_data_path = pd.DataFrame()
        self.is_multi_tickers = multi_ticker
        self.freq_ = f
        self.dmean_ = dmean
        self.static_bin = static_bin
    
    def init_data(self,input_data:pd.DataFrame):
        cols = input_data.columns
        if (FCOL.EP.value not in cols) or (FCOL.FL.value not in cols) or (FCOL.TK.value not in cols) or (FCOL.GP.value not in cols):
            print("input data columns incomplete, need to have {0},{1},{2},{3}"
                  .format(FCOL.EP.value,FCOL.FL.value,FCOL.TK.value,FCOL.GP.value))
            return
        
        if len(input_data[FCOL.TK.value].drop_duplicates()) > 1:
            self.is_multi_tickers = True
        
        process_df = input_data[[FCOL.DT.value,FCOL.EP.value,FCOL.FL.value,FCOL.TK.value]].copy()
        process_df.columns = ['date','entry_price','factor_level','asset']
        price_df = pd.pivot_table(process_df,values = 'entry_price', index=['date'], columns=['asset'])
        process_df = process_df.set_index(['date','asset'])
        factor_series = process_df['factor_level']
        
        #get asset - sector mapping
        glist = input_data['group'].drop_duplicates().tolist()
        idx = [i for i in range(len(glist))]
        group_idx_map = dict(zip(glist,idx))
        idx_sector_map = dict(zip(idx,glist))
        ticker_idx_list = [group_idx_map[g] for g in input_data['group']]
        ticker_sector_map = dict(zip(input_data['ticker'],ticker_idx_list))
        
        #cacluate bins
        b = self.generate_bin(factor_series)
        # check max horizon and generate horizon path
        max_horizon = max(self.horizon)
        h_path = tuple(np.arange(max_horizon) + 1)
        #generate factordata
        if self.freq_ is FREQ.DAILY:
            self.factor_data = al.utils.get_clean_factor_and_forward_returns(factor_series, 
                                                                        price_df, 
                                                                        quantiles=None,
                                                                        bins=b,
                                                                        periods=self.horizon,
                                                                        groupby=ticker_sector_map,
                                                                        groupby_labels=idx_sector_map)
            
            self.factor_data_path = al.utils.get_clean_factor_and_forward_returns(factor_series, 
                                                                        price_df, 
                                                                        quantiles=None,
                                                                        bins=b,
                                                                        periods=h_path,
                                                                        groupby=ticker_sector_map,
                                                                        groupby_labels=idx_sector_map)
        else:
            self.factor_data = get_intraday_clean_factor_and_forward_returns(factor_series, 
                                                                        price_df,
                                                                        self.freq_.value, 
                                                                        quantiles=None,
                                                                        bins=b,
                                                                        periods=self.horizon,
                                                                        groupby=ticker_sector_map,
                                                                        groupby_labels=idx_sector_map)
            
            self.factor_data_path = get_intraday_clean_factor_and_forward_returns(factor_series, 
                                                                        price_df,
                                                                        self.freq_.value, 
                                                                        quantiles=None,
                                                                        bins=b,
                                                                        periods=h_path,
                                                                        groupby=ticker_sector_map,
                                                                        groupby_labels=idx_sector_map)
        return factor_series, price_df
        
    def generate_bin(self,f_series:pd.Series):
        b =[round(f_series.quantile(i),2) for i in np.linspace(0,1,self.num_quantiles + 1)]
        
        if len(set(b)) < len(b) or 'lin' in self.bin_type:
            #we cannot use quantiles as boundary for bins here, use linear spaceing instead
            print("using linear spacing for bins.")
            b = [round(i,2) for i in np.linspace(f_series.min(),f_series.max(),self.num_quantiles + 1)]
            
        if 'sig' in self.bin_type:
            print("using sigma multiple spacing for bins.")
            m = f_series.mean()
            std = f_series.std()
            b = [f_series.min(), m - 3*std, m - 2*std, m - std, m + std, m + 2*std, m + 3*std, f_series.max()]
        
        if len(self.static_bin) >1 and 'static' in self.bin_type:
            print("static bin detected, using static bins.")
            b = []
            if f_series.min() < self.static_bin[0]:
                b.append(f_series.min())
            b.extend(self.static_bin)
            if f_series.max() > self.static_bin[-1]:
                b.append(f_series.max())
        
        b[0] = round(b[0] - np.abs(b[0]) * 0.05,2)
        b[-1] = round(b[-1] + np.abs(b[-1]) * 0.05,2)
        
        #show bin
        q_names = [i +1 for i in range(len(b))]
        f_ranges = tuple(zip(b[:-1],b[1:]))
        dict(zip(q_names,f_ranges))
        q_info_df = pd.DataFrame(dict(zip(q_names,f_ranges)))
        q_info_df = q_info_df.T
        q_info_df.columns = ['start','end']
        q_info_df.index = q_info_df.index.set_names('quantile')
        print("bin calculated:")
        print(q_info_df)
        return b
    
    def plotfactor(self, with_indicator:bool = False):
        pass
    
    def plotheatmap(self):
        pass

    def calc_factor_stats(self):
        result_dict = {}
        #calculate IC
        ic = factor_ic_by_asset(self.factor_data)
        for c in ic.columns:
            result_dict["IC_" + c] = ic[c][0]

        return result_dict

    def plot_ic_summary_table(self):
        if self.is_multi_tickers:
            ic = al.performance.factor_information_coefficient(self.factor_data)
            al.plotting.plot_information_table(ic)
        else:
            ic = factor_ic_by_asset(self.factor_data)
            print("Information Analysis")
            al.utils.print_table(ic.apply(lambda x: x.round(3)))

    def plot_quantile_return_bar(self):
        if self.is_multi_tickers:
            mean_return_by_q,std_dev = al.performance.mean_return_by_quantile(self.factor_data, by_date=False, demeaned= self.dmean_)
        else:
            mean_return_by_q,std_dev = al.performance.mean_return_by_quantile(self.factor_data, demeaned=self.dmean_)
        al.plotting.plot_quantile_returns_bar(mean_return_by_q)
        
    def plot_ret_violin(self):
        if self.is_multi_tickers:
            mean_return_by_q_daily, std_err = al.performance.mean_return_by_quantile(self.factor_data, by_date=True,demeaned= self.dmean_)
        else:
            mean_return_by_q_daily = self.factor_data.copy()
            mean_return_by_q_daily = mean_return_by_q_daily.reset_index()
            cols_filter = [c for c in mean_return_by_q_daily.columns if c not in ['group','asset','factor']]
            mean_return_by_q_daily = mean_return_by_q_daily[cols_filter]
            mean_return_by_q_daily = mean_return_by_q_daily.set_index(['factor_quantile','date'])
        al.plotting.plot_quantile_returns_violin(mean_return_by_q_daily)
        
    def plot_tear_sheet_c1(self):
        """
        this is tear sheet view customized for time series signals.
        demean = False (there is no systematic return underlying time series factor)
        """
        al.plotting.plot_quantile_statistics_table(self.factor_data)
        self.plot_ic_summary_table()
        
        if self.is_multi_tickers:
            self.plot_monthly_ic_heatmap()
        #al.tears.create_returns_tear_sheet(self.factor_data)

        self.plot_quantile_return_bar()
        self.plot_ret_violin()
        
        self.plot_quantile_bbi_bar()
        self.plot_rr_skew_violin()
        self.plot_rr_ratio_violin()
    
    def plot_quantile_bbi_bar(self):
        """
        this is to plot the customized bull/bear index(BBI) by quantile.
        The BBI is defined as (number of returns > 0)/(Total number of returns) * 100 - 50.
        A positive value indicate changes of positive return > 50% vs
        A negative value indicate changes of negative return > 50%.
        """
        bull_bear_index_by_q = bull_bear_prob_by_quantile(self.factor_data)
        plot_q_bb_prob_bar(bull_bear_index_by_q)
    
    def plot_rr_skew_violin(self):
        
        rr_skew_data = self.calc_rr_skew_df()
        
        if self.is_multi_tickers:
            rr_skew_by_q_daily, std_err = al.performance.mean_return_by_quantile(rr_skew_data, by_date=True,demeaned= self.dmean_)
        else:
            rr_skew_by_q_daily = rr_skew_data.copy()
            rr_skew_by_q_daily = rr_skew_by_q_daily.reset_index()
            cols_filter = [c for c in rr_skew_by_q_daily.columns if c not in ['group','asset','factor']]
            rr_skew_by_q_daily = rr_skew_by_q_daily[cols_filter]
            rr_skew_by_q_daily = rr_skew_by_q_daily.set_index(['factor_quantile','date'])
        plot_quantile_rr_skew_violin(rr_skew_by_q_daily)
    
    def plot_rr_ratio_violin(self):
        
        rr_ratio_data = self.calc_rr_ratio_df()
        
        if self.is_multi_tickers:
            rr_ratio_by_q_daily, std_err = al.performance.mean_return_by_quantile(rr_ratio_data, by_date=True,demeaned= self.dmean_)
        else:
            rr_ratio_by_q_daily = rr_ratio_data.copy()
            rr_ratio_by_q_daily = rr_ratio_by_q_daily.reset_index()
            cols_filter = [c for c in rr_ratio_by_q_daily.columns if c not in ['group','asset','factor']]
            rr_ratio_by_q_daily = rr_ratio_by_q_daily[cols_filter]
            rr_ratio_by_q_daily = rr_ratio_by_q_daily.set_index(['factor_quantile','date'])
        plot_quantile_rr_ratio_violin(rr_ratio_by_q_daily)
        
    def calc_rr_skew_df(self):
        """
        This methos calculate the skew of risk/reward of the holding period.
        RR_Skew := (max of returns during target holding period + min of returns in holding period)/2.
        The skew can be interporated as a quick look at risk/rewards. For example, a positive skew means
        max downside returns < max upside returns, risk/rewards favors long.
        """
        cols_full = al.utils.get_forward_returns_columns(self.factor_data_path.columns)
        cols_target = al.utils.get_forward_returns_columns(self.factor_data.columns)
        group_stats_max = self.factor_data_path[cols_full].cummax(axis = 1)
        group_stats_min = self.factor_data_path[cols_full].cummin(axis = 1)
        skew_df = (group_stats_max + group_stats_min)/2
        skew_df = skew_df[cols_target].join(self.factor_data_path[['factor','group','factor_quantile']])
        return skew_df
    
    def calc_rr_ratio_df(self):
        """
        This method calculates the rewards/risk ratio. The rewards is defined as maximum gains acheived during the
        holding period. The risk is defined as maximum loss happened during the holding period. If there is no loss
        occurred, we hardcode the ratio as 2.
        
        Parameters
        ----------
        filter_zscore : int or float, optional
        Sets rr_ratio greater than X standard deviations
        from the the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.
        
        Returns:
            rr_ratio_df: risk/rewards for each holding periods.
        """
        cols_full = al.utils.get_forward_returns_columns(self.factor_data_path.columns)
        cols_target = al.utils.get_forward_returns_columns(self.factor_data.columns)
        
        factor_series = self.factor_data['factor']
        group_stats_max = self.factor_data_path[cols_full].cummax(axis = 1)
        group_stats_min = self.factor_data_path[cols_full].cummin(axis = 1)
        rr_ratio_df = self.factor_data[['factor','group','factor_quantile']].copy()
        
        for col in cols_target:
            rewards = (factor_series > 0) * np.maximum(group_stats_max[col],0) + (factor_series < 0) * np.maximum(-group_stats_min[col],0)
            risk = (factor_series > 0) *np.maximum(-group_stats_min[col],0) + (factor_series < 0) * np.maximum(group_stats_max[col],0)
            rr_ratio = rewards/risk
            rr_ratio =rr_ratio.replace([-np.inf,np.inf],2)
            
            #cap the ratio at 5, we are more interested in the distribution between 0-5
            mask = abs(rr_ratio) > 5
            rr_ratio[mask] = 5
            
            rr_ratio =rr_ratio.fillna(0)
            rr_ratio_df[col] = rr_ratio.values
        
        return rr_ratio_df
    
    def plot_mean_quantile_return_sprd(self):
        pass

    def plot_long_short_cum_returns(self):
        pass

    def plot_monthly_ic_heatmap(self):
        mean_monthly_ic = al.performance.mean_information_coefficient(self.factor_data, by_time='M')
        al.plotting.plot_monthly_ic_heatmap(mean_monthly_ic)