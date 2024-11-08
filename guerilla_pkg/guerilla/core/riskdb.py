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
from guerilla.core.objects import FREQ,RiskData
from vnpy.trader.object import TradeData
from vnpy.trader.constant import Exchange
from vnpy.trader.utility import get_file_path
import pandas as pd
from vnpy.trader.database import (
    BaseDatabase,
    DB_TZ,
    convert_tz
)

config_ = cfg.config_parser()
path: str = str(get_file_path(config_["database"]["risk_db"]))
db: PeeweeSqliteDatabase = PeeweeSqliteDatabase(path)    

class DbRiskData(Model):
    
    date_time: datetime = DateTimeField()
    group: str = CharField()
    frequency: str = CharField()
    horizon: int = IntegerField()
    risk_type: str = CharField()
    risk_name: str = CharField()
    risk_value: float = FloatField()
    
    
    class Meta:
        database: PeeweeSqliteDatabase = db
        indexes: tuple = ((( "date_time", "group", "frequency", "horizon","risk_type","risk_name"), True),)
        primary_key = False

class SqliteDatabase:
    """SQLite数据库接口"""

    def __init__(self) -> None:
        """"""
        self.db: PeeweeSqliteDatabase = db
        self.open()
        self.db.create_tables([DbRiskData], safe=True)
        self.close()
    
    def close(self) -> None:
        """"""
        if not self.db.is_closed():
            self.db.close()
    
    def open(self) -> None:
        """"""
        if self.db.is_closed():
            self.db.connect()
    
    def save_risk_data(self,risk:List[RiskData]) -> bool:
        """"""
        self.open()
        data = []
        for r in risk:
            data.append({'date_time': r.date_time,
                         'group': r.group,
                         'frequency': r.frequency.value,
                         'horizon': r.horizon,
                         'risk_type': r.risk_type,
                         'risk_name': r.risk_name,
                         'risk_value': r.risk_value})
        
        with self.db.atomic():
            DbRiskData.insert(data).on_conflict_replace().execute()
        
        self.close()
        return True
    
    def read_risk_data(self, date: datetime) -> pd.DataFrame:
        """"""
        self.open()
        query = DbRiskData.select().where(DbRiskData.date_time == date)
        risk_data = list(query.dicts())
        self.close()
        
        return pd.DataFrame(risk_data)
    
    def read_risk_data_ts_all(self, group_name:str, frequency: str, start_date:str) -> pd.DataFrame:
        """"""
        self.open()
        query = DbRiskData.select().where(DbRiskData.group == group_name,
                                           DbRiskData.frequency == frequency,
                                           DbRiskData.date_time >= start_date)
        risk_data = list(query.dicts())
        self.close()
        
        return pd.DataFrame(risk_data)
    
    def read_risk_data_ts(self, group_name:str, 
                          frequency: str,
                          horizon:int,
                          risk_type:str,
                          risk_name:str,
                          start_date:str) -> pd.DataFrame:
        """"""
        self.open()
        query = DbRiskData.select().where(DbRiskData.group == group_name,
                                           DbRiskData.frequency == frequency,
                                           DbRiskData.horizon == horizon,
                                           DbRiskData.risk_type == risk_type,
                                           DbRiskData.risk_name == risk_name,
                                           DbRiskData.date_time >= start_date)
        risk_data = list(query.dicts())
        self.close()
        
        return pd.DataFrame(risk_data)
    
    def read_risk_data_ts_bytype(self, group_name:str, 
                          frequency: str,
                          horizon:int,
                          risk_type:str,
                          start_date:str) -> pd.DataFrame:
        """"""
        self.open()
        query = DbRiskData.select().where(DbRiskData.group == group_name,
                                           DbRiskData.frequency == frequency,
                                           DbRiskData.horizon == horizon,
                                           DbRiskData.risk_type == risk_type,
                                           DbRiskData.date_time >= start_date)
        risk_data = list(query.dicts())
        self.close()
        
        return pd.DataFrame(risk_data)