import numpy as np
from datetime import datetime
import pandas as pd
from guerilla.core.tradedb import SqliteDatabase as tradeDB
from guerilla.core.riskdb import SqliteDatabase as riskDB
from guerilla.dataservice.dsUtil import FREQ,SOURCE,read_from_vn_sqlite
from guerilla.trader.tutils import lookup_price_tick
from guerilla.core.objects import RiskData
from ta import volatility as vo
from enum import Enum
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class RiskType(Enum):
    VOL = 'vol'
    ATR = 'atr'
    MAXLOSS = 'maxloss'

class riskEngine(object):
    def __init__(self,
                 riskdate:datetime,
                 totalEquity:float,
                 horizion_st:int = 14,
                 horizion_lt:int = 90,
                 f:FREQ = FREQ.DAILY,
                 position_df:pd.DataFrame = None,
                 market_info_df:pd.DataFrame= None,
                 risklimit:float = 0.3,
                 portfolio:str = 'CTA'):
        self.rdate = riskdate
        self.totalEquity = totalEquity
        self.risklimit = risklimit
        self.pos_df = position_df
        self.freq = f
        self.horizion_st = horizion_st
        self.horizion_lt = horizion_lt
        self.market_info_df = market_info_df
        self.port_grp = portfolio
        #interim result cache
        self.correlation_df = None
        self.volatility_df = None
        self.atr_df = None
        self.exp_max_loss_df = None
        self.p_volatility_df = None
        self.p_atr_df = None
        self.p_exp_max_loss_df = None
        self.calc_df = None
        self.lev = None
        #final risk metrics
        self.risk_metrics = {}
    
    # load position info into risk engine. this is the prequisite for risk calculation.
    def load_position_info(self):
        trade_db = tradeDB()
        self.pos_df = trade_db.read_position_data(date=self.rdate)
    
    # load market info into risk engine. this is the prequisite for risk calculation.
    def load_market_info(self):
        if self.pos_df is None:
            self.load_position_info()
            
        symbol_map = dict(zip(self.pos_df['symbol'], self.pos_df['exchange']))
        vn_symbols = [k + '.' + v for k,v in symbol_map.items()]
        mdata_df = read_from_vn_sqlite(vn_tickers=vn_symbols)
        
        if len(mdata_df) < 1:
            raise ValueError('No market data found for the date: {}'.format(self.rdate.date()))
        
        mdata_df['date']= mdata_df['datetime'].apply(lambda x:x.date())
        self.market_info_df = mdata_df[mdata_df['date'] <= self.rdate.date()]
        
        #prepare for the calculation_df table
        pos_agg_df = self.pos_df.groupby('symbol')['units'].agg(['sum'])
        pos_agg_df.columns =['units']
        self.calc_df = self.market_info_df[self.market_info_df['datetime'].dt.date == max(self.market_info_df['date'])][['symbol','close_price']]
        self.calc_df = pd.merge(self.calc_df, pos_agg_df, on='symbol')
        self.calc_df = self.calc_df.set_index('symbol')
        price_tick_list = [lookup_price_tick(s) for s in self.calc_df.index]
        unit_price = [u for (p,u) in price_tick_list]
        self.calc_df['unit_price'] = unit_price
        self.calc_df['notional'] = self.calc_df['unit_price'] * self.calc_df['close_price'] * self.calc_df['units']
    
    def calculate_risk(self):
        pass
    
    def calculate_leverage(self):
        leverage = np.sum(np.abs(self.calc_df['notional']))/self.totalEquity
        self.lev = leverage
        return leverage
        
    def calculate_correlation(self, smoothing_factor:int = 3):
        last = self.horizion_lt
        if self.correlation_df is not None:
            return self.correlation_df
        
        return_df = self.market_info_df.pivot(index='date',columns='symbol',values='close_price').pct_change().rolling(smoothing_factor).mean()
        missing_percentage = return_df.isna().mean() * 100
        columns_to_drop = missing_percentage[missing_percentage >= 50].index
        return_clean_df = return_df.drop(columns=columns_to_drop).dropna().tail(last)
        self.correlation_df = return_clean_df.corr()
        return self.correlation_df
    
    def calculate_volatility(self):
        last = self.horizion_lt
        if self.volatility_df is not None:
            return self.volatility_df
        
        return_df = self.market_info_df.pivot(index='date',columns='symbol',values='close_price').pct_change()
        missing_percentage = return_df.isna().mean() * 100
        columns_to_drop = missing_percentage[missing_percentage >= 50].index
        return_clean_df = return_df.drop(columns=columns_to_drop).dropna().tail(last)
        self.volatility_df = return_clean_df.std()
        return self.volatility_df
    
    def calculate_atr(self):
        win = self.horizion_st
        if self.atr_df is not None:
            return self.atr_df
        atr =[]
        #anchor to the volatility_df for order consistency.
        for s in self.volatility_df.index:
            price_df = self.market_info_df[self.market_info_df['symbol']==s][['datetime','symbol','high_price','low_price','close_price']]\
                .sort_values(by='datetime')
            atr_df = vo.average_true_range(high=price_df['high_price'],low=price_df['low_price'],close=price_df['close_price'],window=win)
            atr.append(atr_df.iloc[-1])
        average_true_range = pd.Series(dict(zip(self.volatility_df.index,atr)))
        atr_pct = average_true_range/np.abs(self.calc_df['close_price'])
        self.atr_df = atr_pct
        return atr_pct
    
    #calculate the expected max loss based on the stop loss specified in individual strategy.[tbd]
        """_summary_
        stops: dict of stop loss price for each strategy.
        """
    def calculate_expected_max_loss(self, stops:dict = {}):
        
        if self.exp_max_loss_df is not None:
            return self.exp_max_loss_df
        
        if stops == {}:
            print('No stop loss defined, please define stop loss first.')
            return None
        else:
            self.pos_df['stop_price'] = [stops[s] for s in self.pos_df['strategy_name']]
            self.pos_df['stop_price_x_unit'] = self.pos_df['stop_price']*self.pos_df['units']
            sl_x_units = self.pos_df.groupby('symbol')['stop_price_x_unit'].sum()
            self.calc_df['avg_sl_price'] = sl_x_units/self.calc_df['units']
            self.calc_df['exp_dd'] = np.abs(self.calc_df['close_price'] - self.calc_df['avg_sl_price'])
            exp_loss = []
            for s in self.volatility_df.index:
                exp_loss.append(self.calc_df.loc[s]['exp_dd'])
            expected_max_loss = pd.Series(dict(zip(self.volatility_df.index,exp_loss)))
            exp_max_loss = expected_max_loss/np.abs(self.calc_df['close_price'])
            self.exp_max_loss_df = exp_max_loss
            return exp_max_loss
    
    def calculate_portfolio_risk(self,risk_type:str = 'vol'):
        if RiskType(risk_type) == RiskType.VOL:
            vol = self.calculate_volatility().to_numpy()
        elif RiskType(risk_type) == RiskType.ATR:
            vol = self.calculate_atr().to_numpy()
        elif RiskType(risk_type) == RiskType.MAXLOSS:
            vol = self.calculate_expected_max_loss().to_numpy()
        else:
            vol = self.calculate_volatility().to_numpy() #default to volatility
        
        w = [self.calc_df.loc[s]['notional'] for s in self.volatility_df.index]
        w = np.array(w)
        corr = self.correlation_df.to_numpy()
        # Calculate covariance matrix
        covariance_matrix = np.outer(vol, vol) * corr
        # Calculate portfolio volatility
        portfolio_volatility = np.sqrt(w @ covariance_matrix @ w.T)
        # Calculate the risk contributions
        rc = w * (covariance_matrix @ w.T) / portfolio_volatility
        rc_df = pd.DataFrame({
            'Asset': self.volatility_df.index,
            'Risk Contribution': rc
        })
        pvol_dict = {'Asset': 'TP', 'Risk Contribution': portfolio_volatility}
        pvol_df = pd.DataFrame(pvol_dict, index=[0])
        rc_df =pd.concat([rc_df,pvol_df])
        rc_df = rc_df.set_index('Asset')
        #populate to interim cache
        if RiskType(risk_type) == RiskType.VOL:
            self.p_volatility_df = rc_df['Risk Contribution'].copy()
        elif RiskType(risk_type) == RiskType.ATR:
            self.p_atr_df = rc_df['Risk Contribution'].copy()
        elif RiskType(risk_type) == RiskType.MAXLOSS:
            self.p_exp_max_loss_df = rc_df['Risk Contribution'].copy()
        else:
            pass
        
        return rc_df
    
    def populate_risk_metrics(self):
        
        if self.totalEquity:
            self.risk_metrics['totalEquity'] = self.totalEquity
            
        if self.lev:
            self.risk_metrics['leverage'] = self.lev
        
        if self.volatility_df is not None:
            self.risk_metrics[RiskType.VOL.value] = self.volatility_df.to_dict()
        
        if self.atr_df is not None:
            self.risk_metrics[RiskType.ATR.value] = self.atr_df.to_dict()
        
        if self.exp_max_loss_df is not None:
            self.risk_metrics[RiskType.MAXLOSS.value] = self.exp_max_loss_df.to_dict()
            
        if self.p_volatility_df is not None:
            self.risk_metrics['portfolio_volatility'] = self.p_volatility_df.to_dict()
            
        if self.p_atr_df is not None:    
            self.risk_metrics['portfolio_atr'] = self.p_atr_df.to_dict()
        
        if self.p_exp_max_loss_df is not None:
            self.risk_metrics['portfolio_maxloss'] = self.p_exp_max_loss_df.to_dict()
        
    def update_risk_db(self):
        rd = []
        if self.totalEquity:
            rd.append(RiskData(date_time=self.rdate,
                               group=self.port_grp,
                               frequency=self.freq,
                               horizon=self.horizion_st,
                               risk_type='ACCT',
                               risk_name='totalEquity',
                               risk_value=self.totalEquity))
            
        if self.lev:
            self.risk_metrics['leverage'] = self.lev
            rd.append(RiskData(date_time=self.rdate,
                               group=self.port_grp,
                               frequency=self.freq,
                               horizon=self.horizion_st,
                               risk_type='ACCT',
                               risk_name='leverage',
                               risk_value=self.lev))
        
        if self.volatility_df is not None:
            for s in self.volatility_df.index:
                rd.append(RiskData(date_time=self.rdate,
                                   group=self.port_grp,
                                   frequency=self.freq,
                                   horizon=self.horizion_lt,
                                   risk_type=RiskType.VOL.value,
                                   risk_name=s,
                                   risk_value=self.volatility_df[s]))
        
        if self.atr_df is not None:
            for s in self.atr_df.index:
                rd.append(RiskData(date_time=self.rdate,
                                   group=self.port_grp,
                                   frequency=self.freq,
                                   horizon=self.horizion_st,
                                   risk_type=RiskType.ATR.value,
                                   risk_name=s,
                                   risk_value=self.atr_df[s]))
        
        if self.exp_max_loss_df is not None:
            for s in self.exp_max_loss_df.index:
                rd.append(RiskData(date_time=self.rdate,
                                   group=self.port_grp,
                                   frequency=self.freq,
                                   horizon=self.horizion_st,
                                   risk_type=RiskType.MAXLOSS.value,
                                   risk_name=s,
                                   risk_value=self.exp_max_loss_df[s]))
            
        if self.p_volatility_df is not None:
            for s in self.p_volatility_df.index:
                rd.append(RiskData(date_time=self.rdate,
                                   group=self.port_grp,
                                   frequency=self.freq,
                                   horizon=self.horizion_lt,
                                   risk_type="Port_" + RiskType.VOL.value,
                                   risk_name=s,
                                   risk_value=self.p_volatility_df[s]))
            
        if self.p_atr_df is not None:    
            for s in self.p_atr_df.index:
                rd.append(RiskData(date_time=self.rdate,
                                   group=self.port_grp,
                                   frequency=self.freq,
                                   horizon=self.horizion_st,
                                   risk_type="Port_" + RiskType.ATR.value,
                                   risk_name=s,
                                   risk_value=self.p_atr_df[s]))
        
        if self.p_exp_max_loss_df is not None:
            for s in self.p_exp_max_loss_df.index:
                rd.append(RiskData(date_time=self.rdate,
                                   group=self.port_grp,
                                   frequency=self.freq,
                                   horizon=self.horizion_st,
                                   risk_type="Port_" + RiskType.MAXLOSS.value,
                                   risk_name=s,
                                   risk_value=self.p_exp_max_loss_df[s]))
        
        #insert db
        rdb_eng = riskDB()
        rdb_eng.save_risk_data(rd)
        return True
    
    #update the latest position info to position table.
    def update_position_snap(self):
        pass
    
    #cross check local(db) position with exchange position.
    def xcheck_position(self):
        pass
    
    def plot_risk_history(self,window = 30, show = True) -> object:
        start_date_str = (self.rdate - pd.Timedelta(days=window)).strftime('%Y-%m-%d')
        risk_db = riskDB()
        total_equity_df = risk_db.read_risk_data_ts(group_name='CTA',frequency = '1d',horizon=14,risk_type = 'ACCT', risk_name = 'totalEquity', start_date = start_date_str)
        lev_ratio_df = risk_db.read_risk_data_ts(group_name='CTA',frequency = '1d',horizon=14,risk_type = 'ACCT', risk_name = 'leverage', start_date = start_date_str)
        port_vol = risk_db.read_risk_data_ts_bytype(group_name='CTA',frequency = '1d',horizon=90,risk_type = 'Port_vol',start_date = start_date_str)
        port_atr = risk_db.read_risk_data_ts_bytype(group_name='CTA',frequency = '1d',horizon=14,risk_type = 'Port_atr',start_date = start_date_str)
        port_max_loss = risk_db.read_risk_data_ts_bytype(group_name='CTA',frequency = '1d',horizon=14,risk_type = 'Port_maxloss',start_date = start_date_str)
        
        #generate figs
        # Step 1: Create subplots using Matplotlib
        fig, axes = plt.subplots(5, 1, figsize=(10, 20))
        plt.subplots_adjust(hspace=0.5)
        # Subplot 1
        x1 = total_equity_df['date_time'].apply(lambda x: x.strftime('%Y-%m-%d'))
        y1 = total_equity_df['risk_value']
        axes[0].plot(x1, y1)
        axes[0].set_xlabel('date')
        axes[0].set_ylabel('total equity')
        axes[0].set_title('total equity')
        axes[0].grid(True)

        # Subplot 2
        x2 = lev_ratio_df['date_time'].apply(lambda x: x.strftime('%Y-%m-%d'))
        y2 = lev_ratio_df['risk_value']
        axes[1].plot(x2, y2)
        axes[1].set_xlabel('date')
        axes[1].set_ylabel('leverage ratio')
        axes[1].set_title('leverage ratio')
        axes[1].grid(True)

        #subplot 3
        port_vol_pivot = port_vol.pivot(index='date_time', columns='risk_name', values='risk_value').replace(np.nan, 0)
        port_vol_pivot.index = port_vol_pivot.index.strftime('%Y-%m-%d')
        columns =[c for c in port_vol_pivot.columns if c != 'TP']
        port_contr_vol_pivot = port_vol_pivot[columns]
        ax = port_contr_vol_pivot.plot.bar(stacked=True, ax=axes[2],rot = 0)

        # Adding labels and title
        ax.legend(fontsize='small')
        ax.set_xlabel('date')
        ax.set_ylabel('contribution volatility')
        ax.set_title('90d portfolio volatility')

        # Subplot 4
        port_atr_pivot = port_atr.pivot(index='date_time', columns='risk_name', values='risk_value').replace(np.nan, 0)
        port_atr_pivot.index = port_atr_pivot.index.strftime('%Y-%m-%d')
        columns =[c for c in port_atr_pivot.columns if c != 'TP']
        port_contr_atr_pivot = port_atr_pivot[columns]
        ax = port_contr_atr_pivot.plot.bar(stacked=True, ax=axes[3],rot = 0)
        # Adding labels and title
        ax.legend(fontsize='small')
        ax.set_xlabel('date')
        ax.set_ylabel('contribution atr')
        ax.set_title('14d portfolio atr')

        #subplot 5
        port_max_loss_pivot = port_max_loss.pivot(index='date_time', columns='risk_name', values='risk_value').replace(np.nan, 0)
        port_max_loss_pivot.index = port_max_loss_pivot.index.strftime('%Y-%m-%d')
        columns =[c for c in port_max_loss_pivot.columns if c != 'TP']
        port_contr_max_loss_pivot = port_max_loss_pivot[columns]
        ax = port_contr_max_loss_pivot.plot.bar(stacked=True, ax=axes[4],rot = 0)
        # Adding labels and title
        ax.legend(fontsize='small')
        ax.set_xlabel('date')
        ax.set_ylabel('contribution max loss')
        ax.set_title('14d portfolio max loss')
        
        image_data = BytesIO()
        fig.savefig(image_data, format='png')
        image_data.seek(0)  # Move the buffer position to the beginning
        image_data_base64 = base64.b64encode(image_data.read()).decode('utf-8')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return image_data_base64