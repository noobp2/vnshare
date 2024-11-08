#this is a position management module
from guerilla.core.tradedb import SqliteDatabase as tradeDB
from vnpy.trader.object import TradeData
from vnpy.trader.constant import Exchange,Direction,Offset
import pandas as pd
from datetime import datetime

class PosEngine(object):
    
    strategy_name:str = None
    
    def __init__(self, strategy_name:str, vt_symbol:str) -> None:
        self.strategy_name = strategy_name
        self.vt_symbol = vt_symbol
        self.trade_db = tradeDB()
        self.pos_net = 0
        self.pos_long = 0
        self.pos_short = 0
        self.calc_pos()
    
    def get_pos(self, netting:bool = True):
        if netting:
            return self.pos_net
        else:
            return self.pos_long, self.pos_short
    
    def update_trade(self, trade:TradeData):
        self.trade_db.save_trade_data(trade, self.strategy_name)
        self.calc_pos()
    
    def update_position(self):
        self.calc_pos()
        timenow = datetime.now()
        position = {}
        position['strategy_name'] = self.strategy_name
        position['symbol'] = self.vt_symbol.split('.')[0]
        position['exchange'] = self.vt_symbol.split('.')[1]
        position['units'] = self.pos_net
        position['date_time'] = datetime(timenow.year, timenow.month, timenow.day)
        #we don't save position data if there is no position(net position = 0)
        if self.pos_net != 0:
            self.trade_db.save_position_data(position)
        
    
    def calc_pos(self):
        trades_df = self.trade_db.read_trade_data(self.strategy_name, self.vt_symbol)
        if len(trades_df) == 0:
            self.pos_long = 0
            self.pos_short = 0
        else:
            #calculate long position
            long_open = trades_df.loc[(trades_df['direction'] == Direction.LONG.value) 
                                      & (trades_df['offset'] == Offset.OPEN.value), 'volume'].sum()
            short_close = trades_df.loc[(trades_df['direction'] == Direction.SHORT.value) 
                                        & ((trades_df['offset'] == Offset.CLOSE.value)
                                           | (trades_df['offset'] == Offset.CLOSETODAY.value)
                                           |(trades_df['offset'] == Offset.CLOSEYESTERDAY.value)), 'volume'].sum()
            self.pos_long = long_open - short_close
            #calculate short position
            short_open = trades_df.loc[(trades_df['direction'] == Direction.SHORT.value) 
                                       & (trades_df['offset'] == Offset.OPEN.value), 'volume'].sum()
            long_close = trades_df.loc[(trades_df['direction'] == Direction.LONG.value) 
                                        & ((trades_df['offset'] == Offset.CLOSE.value)
                                           | (trades_df['offset'] == Offset.CLOSETODAY.value)
                                           |(trades_df['offset'] == Offset.CLOSEYESTERDAY.value)), 'volume'].sum()
            self.pos_short = long_close - short_open
        
        self.pos_net = self.pos_long + self.pos_short
    
    def first_trading_date(self):
        trades_df = self.trade_db.read_trade_data(self.strategy_name, self.vt_symbol)
        
        if len(trades_df) == 0:
            return None
        
        trades_open_df = trades_df[trades_df['offset'] == Offset.OPEN.value].sort_values(by=['datetime'], ascending=False)
        
        if len(trades_open_df) > 0:
            return datetime.strptime(trades_open_df.iloc[0]['datetime'], "%Y-%m-%d %H:%M:%S%z")
        else:
            return None