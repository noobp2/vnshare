import pandas as pd
from ..fbase import FactorBase,FCOL,FDBCOL
from guerilla.dataservice.dsUtil import FREQ,SOURCE
from guerilla.core.objects import FTYPE
from pickle import DICT
from ta import trend as td
from ta import momentum as mo
from ta import volatility as vo

class MultiFactor(FactorBase):
    def __init__(self, ticker: str, f: FREQ = FREQ.DAILY, s: SOURCE = SOURCE.DEF,g:str = "G1", tz:str = 'Asia/Shanghai') -> None:
        super().__init__(ticker, f, s, g, tz)
        self.ma_short_window = 5
        self.ma_long_window = 60
        self.vol_short_window = 5
        self.vol_long_window = 10
        self.rsi_short_window = 7
        self.rsi_long_window = 14
        self.qtl_window = 10
        self.smooth_window = 5
        
    def setup_params(self, params = {"ma_win_short": 5,
                                           "ma_win_long": 60,
                                           "vol_win_short":5,
                                           "vol_win_long":10,
                                           "rsi_win_short":7,
                                           "rsi_win_long":14,
                                           "qtl_win":10,
                                           "smooth_win":5}) -> None:
        self.ma_short_window = params["ma_win_short"]
        self.ma_long_window = params["ma_win_long"]
        self.vol_short_window = params["vol_win_short"]
        self.vol_long_window = params["vol_win_long"]
        self.rsi_long_window = params["rsi_win_long"]
        self.rsi_short_window = params["rsi_win_short"]
        self.qtl_window = params["qtl_win"]
        self.smooth_window = params["smooth_win"]
        self.factor_name = __class__.__name__
    
    def calc_factor(self) -> None:
        calc_df = self.input_df
        calc_df['ma_short'] = td.sma_indicator(close=calc_df['close'], window= self.ma_short_window, fillna=True)
        calc_df['ma_long'] = td.sma_indicator(close=calc_df['close'], window= self.ma_long_window, fillna=True)
        calc_df['rumi'] = (calc_df['ma_short'] - calc_df['ma_long'])/calc_df['ma_long'] * 100
        calc_df['rumi_sma'] = td.sma_indicator(close=calc_df['rumi'], window= self.smooth_window, fillna=True)
        calc_df['rsi_long'] = mo.rsi(close=calc_df['close'],window= self.rsi_long_window,fillna=True)
        calc_df['rsi_short'] = mo.rsi(close=calc_df['close'],window= self.rsi_short_window,fillna=True)
        calc_df['rsi_diff'] = calc_df['rsi_short'] - calc_df['rsi_long']
        calc_df['rsi_diff_sma'] = td.sma_indicator(close=calc_df['rsi_diff'], window= self.smooth_window, fillna=True)
        calc_df['vol_short'] = td.sma_indicator(close=calc_df['volume'], window= self.vol_short_window, fillna=True)
        calc_df['vol_long'] = td.sma_indicator(close=calc_df['volume'], window= self.vol_long_window, fillna=True)
        calc_df['vol_diff'] = (calc_df['vol_short'] - calc_df['vol_long'])/calc_df['vol_long'] * 100
        calc_df['h'] = vo.donchian_channel_hband(high = calc_df['high'],low = calc_df['low'], close=calc_df['close'], window= self.qtl_window, fillna=True)
        calc_df['l'] = vo.donchian_channel_lband(high = calc_df['high'],low = calc_df['low'], close=calc_df['close'], window= self.qtl_window, fillna=True)
        calc_df['qtl'] = (calc_df['close'] - calc_df['l'])/(calc_df['h'] - calc_df['l']) * 100


        #rumi is similar to macd
        rumi_df = calc_df[[FDBCOL.DT.value, FDBCOL.SB.value, FDBCOL.EXCH.value, FDBCOL.FR.value, 'rumi']].copy().fillna(0)
        rumi_df.columns = [FDBCOL.DT.value, FDBCOL.SB.value, FDBCOL.EXCH.value, FDBCOL.FR.value, FDBCOL.FV.value]
        rumi_df[FDBCOL.FT.value] = FTYPE.MOM
        rumi_df[FDBCOL.FN.value] = 'RUMI_' + str(self.ma_short_window) + '_' + str(self.ma_long_window) + '_' + str(self.smooth_window)
        
        #rsi
        rsi_df = calc_df[[FDBCOL.DT.value, FDBCOL.SB.value, FDBCOL.EXCH.value, FDBCOL.FR.value, 'rsi_long']].copy()
        rsi_df.columns = [FDBCOL.DT.value, FDBCOL.SB.value, FDBCOL.EXCH.value, FDBCOL.FR.value, FDBCOL.FV.value]
        rsi_df[FDBCOL.FT.value] = FTYPE.MOM
        rsi_df[FDBCOL.FN.value] = 'RSI_' + str(self.rsi_long_window)
        
        # rsi macd
        rsi_mom_df = calc_df[[FDBCOL.DT.value, FDBCOL.SB.value, FDBCOL.EXCH.value, FDBCOL.FR.value, 'rsi_diff_sma']].copy()
        rsi_mom_df.columns = [FDBCOL.DT.value, FDBCOL.SB.value, FDBCOL.EXCH.value, FDBCOL.FR.value, FDBCOL.FV.value]
        rsi_mom_df[FDBCOL.FT.value] = FTYPE.MOM
        rsi_mom_df[FDBCOL.FN.value] = 'RSI_MOM_' +str(self.rsi_short_window) +'_' + str(self.rsi_long_window) + '_' + str(self.smooth_window)
        
        # volume mom
        vol_diff_df = calc_df[[FDBCOL.DT.value, FDBCOL.SB.value, FDBCOL.EXCH.value, FDBCOL.FR.value, 'vol_diff']].copy().fillna(0)
        vol_diff_df.columns = [FDBCOL.DT.value, FDBCOL.SB.value, FDBCOL.EXCH.value, FDBCOL.FR.value, FDBCOL.FV.value]
        vol_diff_df[FDBCOL.FT.value] = FTYPE.VLM
        vol_diff_df[FDBCOL.FN.value] = 'VOL_DIFF_' +str(self.vol_short_window) +'_' + str(self.vol_long_window)
        # Quantile level ST
        qtl_df = calc_df[[FDBCOL.DT.value, FDBCOL.SB.value, FDBCOL.EXCH.value, FDBCOL.FR.value, 'qtl']].copy().fillna(50)  
        qtl_df.columns = [FDBCOL.DT.value, FDBCOL.SB.value, FDBCOL.EXCH.value, FDBCOL.FR.value, FDBCOL.FV.value]
        qtl_df[FDBCOL.FT.value] = FTYPE.LEV
        qtl_df[FDBCOL.FN.value] = 'QTL_' +str(self.qtl_window)
        
        self.factordata_df = pd.concat([rumi_df,rsi_df,rsi_mom_df,vol_diff_df, qtl_df],ignore_index=True)
        
        