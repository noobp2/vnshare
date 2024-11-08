from ..fbase import FactorBase,FCOL
from guerilla.dataservice.dsUtil import FREQ,SOURCE
from pickle import DICT
from ta import trend as td
from ta import momentum as mo
from ta import volatility as vo
import numpy as np

class HmFactor(FactorBase):
    def __init__(self, ticker: str, f: FREQ = FREQ.DAILY, s: SOURCE = SOURCE.DEF,g:str = "G1",  tz:str = 'Asia/Shanghai') -> None:
        super().__init__(ticker, f, s, g, tz)
        self.short_window = 5
        self.long_window = 60
        
    def setup_params(self, params = {"window_short": 5, "window_long": 60}) -> None:
        self.short_window = params["window_short"]
        self.long_window = params["window_long"]
        self.factor_name = __class__.__name__ + 'S' + str(self.short_window) + 'L' + str(self.long_window)
    
    def calc_factor(self) -> None:
        calc_df = self.input_df.copy()
        calc_df['tr'] = vo.average_true_range(high = calc_df['high'],low = calc_df['low'],close=calc_df['close'],window=1,fillna=True)
        calc_df['atr'] = vo.average_true_range(high = calc_df['high'],low = calc_df['low'],close=calc_df['close'],window=self.short_window,fillna=True)
        calc_df['tr_multi'] = calc_df['tr']/calc_df['atr'] * np.sign(calc_df['close'] - calc_df['open'])
        calc_df['entry_price'] = calc_df['open'].shift(-1)
        self.output_df = calc_df[['datetime', 'close', 'tr_multi','entry_price']].copy()
        self.output_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.output_df = self.output_df.dropna()
        self.output_df.columns = [FCOL.DT.value, FCOL.PL.value, FCOL.FL.value,FCOL.EP.value]