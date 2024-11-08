# this is portfolio construction tool in dimension of 
# 1. risk factors: C(risk factors) -> single synthetic risk factor
# 2. across assets: C(assets) -> rank(asset)
from typing import List
from .futil import *
from .fbase import FactorBase,FCOL,FDBCOL,FactorDataEngine
from guerilla.dataservice.dsUtil import FREQ,SOURCE,read_from_vn_sqlite
from guerilla.core.constants import d_format_long,d_format_short,d_format_ms,pattern_number_re,pattern_letters_re
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster

class ConstructionEngine:
    def __init__(self, ticker_universe:List[str],
                 f:FREQ,
                 factor_name:str = 'NA',
                 mask:List[float] = [-75,75]) -> None:
        self.universe = ticker_universe
        self.f_name = factor_name
        self.freq_= f
        self.fdata_engine = FactorDataEngine(f = self.freq_,s=SOURCE.DEF)
        self.f_data_df = pd.DataFrame()
        self.p_data_df = pd.DataFrame()
        self.mask_ = mask
    
    def init_data(self, start_date:datetime = datetime(2016,1,1),
                         end_date:datetime = datetime.now()) -> None:
        self.f_data_df = self.load_factor_data(start_date=start_date,end_date=end_date)
        self.p_data_df = self.load_tickers_history(start_date=start_date,end_date=end_date)
    
    #public methods
    def factor_ranks_map(self, last:int = 10):
        f_pivot_df = self.f_data_df.pivot(index='datetime',columns='symbol',values='factor_value')\
            .sort_index(ascending=False).head(last)
        def highlight_cell(val):
            color = ''
            if val > self.mask_[1]:
                color = 'background-color: red'
            elif val < self.mask_[0]:
                color = 'background-color: green'
            else:
                pass
            return color

        def format_with_unit(x):
            return '{:.0f}'.format(x)

        f_pivot_v_df= f_pivot_df.round(0)
        # Apply the highlighting function to specific columns
        highlight_columns = f_pivot_df.columns
        styled_df = f_pivot_v_df.style.applymap(highlight_cell, subset=pd.IndexSlice[:, highlight_columns])\
            .format(formatter=format_with_unit)
        pd.set_option('display.float_format', '{:.2f}'.format)
        pd.set_option('display.max_columns', None)
        # Display the styled DataFrame with the full column list
        return styled_df
    
    def correlation_map(self,last:int=90) -> pd.DataFrame:
        return_df = self.p_data_df.pivot(index='datetime',columns='symbol',values='close_price').pct_change().rolling(3).mean()
        missing_percentage = return_df.isna().mean() * 100
        columns_to_drop = missing_percentage[missing_percentage >= 50].index
        return_clean_df = return_df.drop(columns=columns_to_drop).dropna().tail(last)
        correlation_matrix = return_clean_df.corr()
        correlation_matrix = self.correlation_cluster(correlation_matrix)
        sns.heatmap(correlation_matrix, cmap='coolwarm')
        return correlation_matrix
    
    def volatility_map(self) -> None:
        pass
    
    #data loading
    def load_factor_data(self,start_date:datetime = datetime(2016,1,1),
                         end_date:datetime = datetime.now()) -> pd.DataFrame:
        result_df = []
        for vn_ticker in self.universe:
            ticker = vn_ticker.split('.')[0]
            temp_df = self.fdata_engine.get_factor_data_db(ticker,[self.f_name],start_date,end_date)
            temp_df['symbol'] = ticker
            result_df.append(temp_df)
        result_df = pd.concat(result_df)
        return result_df
    
    def load_tickers_history(self,start_date:datetime = datetime(2016,1,1),
                         end_date:datetime = datetime.now()) -> pd.DataFrame:
        start_str = start_date.strftime(d_format_short)
        end_str = end_date.strftime(d_format_short)
        result_df = read_from_vn_sqlite(self.universe,self.freq_,start_str,end_str)
        return result_df
    
    #utility functions
    def calc_corr(self) -> None:
        pass
    
    def calc_vol(self) -> None:
        pass
    
    def calc_ranks(self) -> None:
        pass
    
    def correlation_cluster(self,corr_matrix:pd.DataFrame,iter:int = 2) -> pd.DataFrame:
        
        for i in range(iter):
            distance_matrix = corr_matrix
            linkage_matrix = linkage(distance_matrix, method='complete')
            threshold = 1
            # Perform clustering using the threshold
            clusters = fcluster(linkage_matrix, threshold, criterion='distance')
            # Assign the cluster labels to the DataFrame
            corr_matrix['cluster'] = clusters
            corr_matrix.sort_values(by='cluster',axis=0,inplace=True)
            row_list = corr_matrix.index.to_list()
            corr_matrix = corr_matrix[row_list]
        
        return corr_matrix