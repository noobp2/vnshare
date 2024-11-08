import pandas as pd
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import guerilla.core.config as cfg
from peewee import (
    AutoField,
    CharField,
    DateTimeField,
    FloatField, IntegerField,
    Model,
    SqliteDatabase as PeeweeSqliteDatabase,
    ModelSelect,
    ModelDelete,
    CompositeKey,
    chunked,
    fn
)
from guerilla.core.objects import FREQ, FTYPE, FactorData, FactorOverview
from vnpy.trader.object import TradeData
from vnpy.trader.constant import Exchange
from vnpy.trader.utility import get_file_path
from vnpy.trader.database import (
    BaseDatabase,
    DB_TZ,
    convert_tz
)

config_ = cfg.config_parser()
path: str = str(get_file_path(config_["database"]["trade_db"]))
db: PeeweeSqliteDatabase = PeeweeSqliteDatabase(path)

class DbTradeData(Model):
    
    gateway_name: str = CharField()
    stratergy_name: str = CharField()
    symbol: str = CharField()
    exchange:str = CharField()
    orderid: int = CharField()
    tradeid: str = CharField()
    direction: str = CharField()
    offset: str = CharField()
    price: float = FloatField()
    volume: float = FloatField()
    datetime: datetime = DateTimeField()
    
    class Meta:
        database: PeeweeSqliteDatabase = db
        #primary_key = CompositeKey("gateway_name","stratergy_name", "symbol", "exchange", "datetime", "tradeid", "orderid")
        indexes: tuple = ((("stratergy_name", "symbol", "exchange", "datetime", "tradeid", "orderid"), True),)
        primary_key = False

class DbPositionData(Model):
    
    strategy_name: str = CharField()
    symbol: str = CharField()
    exchange:str = CharField()
    units: float = FloatField()
    date_time: datetime = DateTimeField()
        
    class Meta:
        database: PeeweeSqliteDatabase = db
        indexes: tuple = ((("strategy_name", "symbol", "exchange", "date_time"), True),)
        primary_key = False

class SqliteDatabase:
    """SQLite数据库接口"""

    def __init__(self) -> None:
        """"""
        self.db: PeeweeSqliteDatabase = db
        self.open()
        self.db.create_tables([DbTradeData,DbPositionData], safe=True)
        self.close()
    
    def close(self) -> None:
        """"""
        if not self.db.is_closed():
            self.db.close()
    
    def open(self) -> None:
        """"""
        if self.db.is_closed():
            self.db.connect()
    
    def save_trade_data(self,trade:TradeData, strategy_name:str) -> bool:
        """"""
        self.open()
        data :dict = {}
        data['gateway_name'] = trade.gateway_name
        data['stratergy_name'] = strategy_name
        data['symbol'] = trade.symbol
        data['orderid'] = trade.orderid
        data['tradeid'] = trade.tradeid
        data['price'] = trade.price
        data['volume'] = trade.volume
        data['datetime'] = trade.datetime
        data['exchange'] = trade.exchange.value
        data['direction'] = trade.direction.value
        data['offset'] = trade.offset.value
        
        with self.db.atomic():
            DbTradeData.insert(data).on_conflict_replace().execute()
        
        self.close()
        return True
    
    def save_position_data(self,position:Dict[str, float]) -> bool:
        """"""
        self.open()
        data :dict = {}
        data['strategy_name'] = position['strategy_name']
        data['symbol'] = position['symbol']
        data['exchange'] = position['exchange']
        data['units'] = position['units']
        data['date_time'] = position['date_time']
        
        with self.db.atomic():
            DbPositionData.insert(data).on_conflict_replace().execute()
        
        self.close()
        return True
    
    def read_trade_data(self, strategy_name:str, vt_symbol:str = 'NONE') -> pd.DataFrame:
        """"""
        
        self.open()
        if vt_symbol == 'NONE':
            query = DbTradeData.select().where(DbTradeData.stratergy_name == strategy_name)
        else:
            exch = vt_symbol.split('.')[1]
            symbol = vt_symbol.split('.')[0]
            query = DbTradeData.select().where(DbTradeData.stratergy_name == strategy_name,
                                               DbTradeData.symbol == symbol,
                                               DbTradeData.exchange == exch)
        query_data = list(query.dicts())
        df = pd.DataFrame(query_data)
        self.close()
        return df
    
    def read_position_data(self, date: datetime) -> pd.DataFrame:
        """"""
        self.open()
        query = DbPositionData.select().where(DbPositionData.date_time == date)
        position_data = list(query.dicts())
        self.close()
        
        return pd.DataFrame(position_data)
    
    def read_position_data_by_strategy(self, strategy_name: str) -> pd.DataFrame:
        """"""
        self.open()
        query = DbPositionData.select().where(DbPositionData.strategy_name == strategy_name)
        position_data = list(query.dicts())
        self.close()
        
        return pd.DataFrame(position_data)