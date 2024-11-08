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
from vnpy.trader.constant import Exchange
from vnpy.trader.utility import get_file_path
from vnpy.trader.database import (
    BaseDatabase,
    DB_TZ,
    convert_tz
)

config_ = cfg.config_parser()
path: str = str(get_file_path(config_["database"]["qr_db"]))
db: PeeweeSqliteDatabase = PeeweeSqliteDatabase(path)


class DbFactorData(Model):
    """K线数据表映射对象"""

    id: AutoField = AutoField()

    symbol: str = CharField()
    exchange: str = CharField()
    datetime: datetime = DateTimeField()
    frequency: str = CharField()
    factor_type: str = CharField()
    factor_name: str = CharField()
    factor_value: float = FloatField()

    class Meta:
        database: PeeweeSqliteDatabase = db
        #primary_key = CompositeKey("symbol", "exchange","datetime", "frequency", "factor_type", "factor_name")
        indexes: tuple = ((("symbol", "exchange", "datetime", "frequency", "factor_type", "factor_name"), True),)

class DbFactorOverview(Model):
    """K线汇总数据表映射对象"""

    symbol: str = CharField()
    exchange: str = CharField()
    frequency: str = CharField()
    factor_type: str = CharField()
    factor_name: str = CharField()
    count: int = IntegerField()
    start: datetime = DateTimeField()
    end: datetime = DateTimeField()

    class Meta:
        database: PeeweeSqliteDatabase = db
        indexes: tuple = ((("symbol", "exchange", "frequency","factor_type", "factor_name"), True),)
        primary_key = False

class SqliteDatabase:
    """SQLite数据库接口"""

    def __init__(self) -> None:
        """"""
        self.db: PeeweeSqliteDatabase = db
        self.open()
        self.db.create_tables([DbFactorData, DbFactorOverview])
    
    def close(self) -> None:
        """"""
        if not self.db.is_closed():
            self.db.close()
    
    def open(self) -> None:
        """"""
        if self.db.is_closed():
            self.db.connect()
        
    def save_factor_data(self, factors: List[FactorData], update_view: bool = False) -> bool:
        """保存K线数据"""
        self.open()
        # 将BarData数据转换为字典，并调整时区
        data: list = []

        for f in factors:
            f.datetime = convert_tz(f.datetime)
            d: dict = f.__dict__
            d['exchange'] = d['exchange'].value
            d['frequency'] = d['frequency'].value
            d['factor_type'] = d['factor_type'].value
            data.append(d)
            
        # 使用upsert操作将数据更新到数据库中
        with self.db.atomic():
            for c in chunked(data, 50):
                DbFactorData.insert_many(c).on_conflict_replace().execute()

        # 更新汇总数据
        if update_view:
            self.init_factor_overview()
        
        self.close()
        return True
   
    def load_factor_data(
        self,
        symbol: str,
        exchange: Exchange,
        frequency: FREQ,
        factor_name: str,
        start: datetime,
        end: datetime
    ) -> List[FactorData]:
        """读取K线数据"""
        self.open()
        
        s: ModelSelect = (
            DbFactorData.select().where(
                (DbFactorData.symbol == symbol)
                & (DbFactorData.exchange == exchange.value)
                & (DbFactorData.frequency == frequency.value)
                & (DbFactorData.factor_name == factor_name)
                & (DbFactorData.datetime >= start)
                & (DbFactorData.datetime <= end)
            ).order_by(DbFactorData.datetime)
        )

        factors: List[FactorData] = []
        for db_f in s:
            f: FactorData = FactorData(
                symbol=db_f.symbol,
                exchange=Exchange(db_f.exchange),
                datetime=datetime.fromtimestamp(db_f.datetime.timestamp(), DB_TZ),
                frequency=FREQ(db_f.frequency),
                factor_type=FTYPE(db_f.factor_type),
                factor_name=db_f.factor_name,
                factor_value=db_f.factor_value
            )
            factors.append(f)
        self.close()
        return factors
    
    def load_factor_data_df(
        self,
        symbol: str,
        exchange: Exchange,
        frequency: FREQ,
        factor_name: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """读取因子数据"""
        
        factors: List[FactorData] = []
        factors = self.load_factor_data(symbol, exchange, frequency, factor_name, start, end)
        result_df: pd.DataFrame = pd.DataFrame([f.__dict__ for f in factors])
        return result_df
    
    def load_factor_data_multi(
        self,
        symbols: List[str],
        frequency: FREQ,
        factor_names: List[str],
        start: datetime,
        end: datetime
    ) -> List[FactorData]:
        """读取K线数据"""
        self.open()
        s: ModelSelect = (
            DbFactorData.select().where(
                (DbFactorData.symbol.in_(symbols))
                & (DbFactorData.frequency == frequency.value)
                & (DbFactorData.factor_name.in_(factor_names))
                & (DbFactorData.datetime >= start)
                & (DbFactorData.datetime <= end)
            ).order_by(DbFactorData.datetime)
        )

        factors: List[FactorData] = []
        for db_f in s:
            f: FactorData = FactorData(
                symbol=db_f.symbol,
                exchange=Exchange(db_f.exchange),
                datetime=datetime.fromtimestamp(db_f.datetime.timestamp(), DB_TZ),
                frequency=FREQ(db_f.frequency),
                factor_type=FTYPE(db_f.factor_type),
                factor_name=db_f.factor_name,
                factor_value=db_f.factor_value
            )
            factors.append(f)
        self.close()
        return factors
    
    def load_factor_data_multi_df(
        self,
        symbols: List[str],
        frequency: FREQ,
        factor_names: List[str],
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """读取因子数据"""
        factors: List[FactorData] = []
        factors = self.load_factor_data_multi(symbols, frequency, factor_names, start, end)
        result_df: pd.DataFrame = pd.DataFrame([f.__dict__ for f in factors])
        return result_df
    
    def delete_factor_data(
        self,
        symbol: str,
        exchange: Exchange,
        frequency: FREQ,
        factor_type: FTYPE,
        factor_name: str
    ) -> int:
        """删除K线数据"""
        self.open()
        d: ModelDelete = DbFactorData.delete().where(
            (DbFactorData.symbol == symbol)
            & (DbFactorData.exchange == exchange.value)
            & (DbFactorData.frequency == frequency.value)
            & (DbFactorData.factor_type == factor_type.value)
            & (DbFactorData.factor_name == factor_name)
        )
        count: int = d.execute()

        # 删除K线汇总数据
        d2: ModelDelete = DbFactorOverview.delete().where(
            (DbFactorOverview.symbol == symbol)
            & (DbFactorOverview.exchange == exchange.value)
            & (DbFactorOverview.frequency == frequency.value)
            & (DbFactorOverview.factor_type == factor_type.value)
            & (DbFactorOverview.factor_name == factor_name)
        )
        d2.execute()
        self.close()
        return count

    def get_factor_overview(self) -> List[FactorOverview]:
        """查询数据库中的K线汇总信息"""
        # 如果已有K线，但缺失汇总信息，则执行初始化
        self.open()
        data_count: int = DbFactorData.select().count()
        overview_count: int = DbFactorOverview.select(fn.SUM(DbFactorOverview.count).alias('total_count')).scalar()

        if data_count != overview_count:
            self.init_factor_overview()

        s: ModelSelect = DbFactorOverview.select()
        overviews: List[FactorOverview] = []
        for overview in s:
            fview = FactorOverview(
                symbol=overview.symbol,
                exchange=Exchange(overview.exchange),
                frequency=FREQ(overview.frequency),
                factor_type=FTYPE(overview.factor_type),
                factor_name=overview.factor_name,
                count=overview.count,
                start=overview.start,
                end=overview.end
            )
            overviews.append(fview)
        self.close()
        overview_df = pd.DataFrame([o.__dict__ for o in overviews])
        return overview_df

    def init_factor_overview(self) -> bool:
        """初始化数据库中的K线汇总信息"""
        
        query = (DbFactorData
         .select(
             DbFactorData.symbol,
             DbFactorData.exchange,
             DbFactorData.frequency,
             DbFactorData.factor_type,
             DbFactorData.factor_name,
             fn.COUNT(DbFactorData.factor_value).alias('count'),
             fn.MIN(DbFactorData.datetime).alias('start'),
             fn.MAX(DbFactorData.datetime).alias('end')).group_by(
             DbFactorData.symbol,
             DbFactorData.exchange,
             DbFactorData.frequency,
             DbFactorData.factor_type,
             DbFactorData.factor_name))
        
        results = query.execute()
        
        data: list = []
        
        for r in results:
            d: dict = {}
            d['symbol'] = r.symbol
            d['exchange'] = r.exchange
            d['frequency'] = r.frequency
            d['factor_type'] = r.factor_type
            d['factor_name'] = r.factor_name
            d['count'] = r.count
            d['start'] = r.start
            d['end'] = r.end
            data.append(d)
        
        # 使用upsert操作将数据更新到数据库中
        with self.db.atomic():
            for c in chunked(data, 50):
                DbFactorOverview.insert_many(c).on_conflict_replace().execute()
        
        return True

    def query_generic(self, qry_str:str) -> pd.DataFrame:
        """查询因子数据"""
        self.open()
        # Execute the query using Peewee's raw() method
        query = self.db.execute_sql(qry_str)
        # Retrieve the results
        results = query.fetchall()
        # Create a DataFrame from the query results
        result_df = pd.DataFrame(results, columns=[desc[0] for desc in query.description])
        self.close()
        return result_df
    
    def update_generic(self, upd_str:str) -> None:
        """更新因子数据"""
        self.open()
        # Execute the query using Peewee's raw() method
        query = self.db.execute_sql(upd_str)
        self.close()
        return None
