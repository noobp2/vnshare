from ..fbase import FactorBase,FCOL
from guerilla.dataservice.dsUtil import FREQ,SOURCE
from pickle import DICT
from ta import trend as td
from ta import momentum as mo
from ta import volatility as vo

class MaFactor(FactorBase):
    def __init__(self, ticker: str, f: FREQ = FREQ.DAILY, s: SOURCE = SOURCE.DEF,g:str = "G1",  tz:str = 'Asia/Shanghai') -> None:
        super().__init__(ticker, f, s, g, tz)
        self.short_window = 5
        self.long_window = 60
        
    def setup_params(self, params = {"window_short": 5, "window_long": 60}) -> None:
        self.short_window = params["window_short"]
        self.long_window = params["window_long"]
        self.factor_name = __class__.__name__ + 'S' + str(self.short_window) + 'L' + str(self.long_window)
    
    def calc_factor(self) -> None:
        calc_df = self.input_df
        calc_df['ma_short'] = td.sma_indicator(close=calc_df['close'], window= self.short_window, fillna=True)
        calc_df['ma_long'] = td.sma_indicator(close=calc_df['close'], window= self.long_window, fillna=True)
        #calc_df['rsi'] = mo.rsi(close=calc_df['close'],window= self.short_window *2)
        calc_df['ma_short_pre'] = calc_df['ma_short'].shift(1)
        calc_df['ma_long_pre'] = calc_df['ma_long'].shift(1)
        calc_df['cross_up'] = (calc_df['ma_short'] > calc_df['ma_long']) * (calc_df['ma_short_pre'] < calc_df['ma_long_pre'])
        calc_df['cross_down'] = -1 * (calc_df['ma_short'] < calc_df['ma_long']) * (calc_df['ma_short_pre'] > calc_df['ma_long_pre'])
        calc_df['tr'] = vo.average_true_range(high = calc_df['high'],low = calc_df['low'],close=calc_df['close'],window=1)
        calc_df['tr_pct'] = calc_df['tr']/calc_df['ma_short'] * 100
        calc_df['cross_factor'] = (calc_df['cross_up'] + calc_df['cross_down']) * calc_df['tr_pct'] + 0.01
        calc_df['entry_price'] = calc_df['open'].shift(-1)
        self.output_df = calc_df[['datetime', 'close', 'cross_factor','entry_price']]
        self.output_df.columns = [FCOL.DT.value, FCOL.PL.value, FCOL.FL.value,FCOL.EP.value]
        
        