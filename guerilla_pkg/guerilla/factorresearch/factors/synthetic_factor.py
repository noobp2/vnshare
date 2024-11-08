from typing import List
import pandas as pd
from datetime import datetime,timedelta
from ..fbase import FactorBase, FCOL,FDBCOL
from guerilla.dataservice.dsUtil import FREQ,SOURCE
from guerilla.core.objects import FTYPE
from guerilla.core.qrdb import SqliteDatabase
from pickle import DICT
from ta import trend as td
from ta import momentum as mo
from ta import volatility as vo

class QtlRumiFactor(FactorBase):
    
    def __init__(self, vn_ticker: str, f: FREQ = FREQ.DAILY, s: SOURCE = SOURCE.DEF, group: str = "G1", tz: str = 'Asia/Shanghai') -> None:
        super().__init__(vn_ticker, f, s, group, tz)
        self.ma_win_short = 5
        self.ma_win_long = 60
        self.rumi_win = 20
        self.qtl_win = 200
        self.factor_name = "QTL_200_RUMI_20"
        
    def setup_params(self, params = {"QTL_WIN": 200, 
                                           "RUMI_WIN": 20,
                                           "MA_WIN_SHORT":5,
                                           "MA_WIN_LONG":60}) -> None:
        self.qtl_win = params["QTL_WIN"]
        self.rumi_win = params["RUMI_WIN"]
        self.ma_win_short = params["MA_WIN_SHORT"]
        self.ma_win_long = params["MA_WIN_LONG"]
        self.qtl_name = "QTL_" + str(self.qtl_win)
        self.rumi_name = "RUMI_5_" + str(self.rumi_win) + "_5"
        self.factor_name = "QTL_" + str(self.qtl_win) + "_RUMI_" + str(self.rumi_win)
    
    def calc_factor(self) -> pd.DataFrame:
        # load factor data
        calc_df = self.get_factor_from_db([self.qtl_name, self.rumi_name])
        calc_df.reset_index(inplace=True)
        calc_df['rumi_pre'] = calc_df[self.rumi_name].shift(1)
        calc_df['cross_down'] = -1 * (calc_df[self.rumi_name] < 0) * (calc_df['rumi_pre'] > 0) * calc_df[self.qtl_name]
        calc_df['cross_up'] = (calc_df[self.rumi_name] > 0) * (calc_df['rumi_pre'] < 0) * (100 - calc_df[self.qtl_name])
        calc_df['qtl_rumi_factor'] = (calc_df['cross_down'] + calc_df['cross_up'])
        calc_df[FDBCOL.FR.value] = self.freq_
        
        factor_df = calc_df[[FDBCOL.DT.value, FDBCOL.SB.value, FDBCOL.EXCH.value, FDBCOL.FR.value, 'qtl_rumi_factor']].copy().fillna(0)  
        factor_df.columns = [FDBCOL.DT.value, FDBCOL.SB.value, FDBCOL.EXCH.value, FDBCOL.FR.value, FDBCOL.FV.value]
        factor_df[FDBCOL.FT.value] = FTYPE.MOM
        factor_df[FDBCOL.FN.value] = 'QTL_RUMI_' + str(self.qtl_win) + '_' + str(self.rumi_win)
        self.factordata_df = factor_df
        
        return calc_df