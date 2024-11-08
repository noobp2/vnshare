from re import T
from typing import Dict,List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from pytz import timezone
from vnpy.trader.constant import Interval
from vnpy.trader.optimize import OptimizationSetting
from .backtest import BacktestingEngine
from .tutils import load_json_file,get_next_key
import os
import traceback
from pathlib import Path
from copy import copy

d_format_short = "%Y-%m-%d"


class PortfolioBacktestEngine:
    
    def __init__(self, strategy_class: type, setting_base_path:str = Path.cwd()):
        #strategies
        self.strategy_A = strategy_class
        self.strategy_A_name = self.strategy_A.__name__
        self.strategy_dev = None
        self.strategy_dev_name = ''
        
        #path set up
        self.setting_path = os.path.join(setting_base_path,'.vntrader')
        self.bt_setting_A_path = os.path.join(self.setting_path,self.strategy_A_name + '_bt_setting.json')
        
        self.optm_setting_path = os.path.join(self.setting_path,'cta_strategy_optm_setting.json')
        self.optm_schedule_path = os.path.join(self.setting_path,'cta_param_optm_schedule.json')
        self.grid_setting_path = os.path.join(self.setting_path,'cta_strategy_grid_setting.json')
        
        self.baseline_param_file = '_base_params.csv'
        #settings
        self.bt_A_setting = {}
        self.opt_setting = {}
        self.opt_schedule = {}
        self.grid_setting = {}
        
        self.load_settings()
    
    def add_dev_strategy(self, strategy_class_dev:type):
        """
        add the strategy we want to compare(using same set of params as strategy A)
        """
        self.strategy_dev = strategy_class_dev
        self.strategy_dev_name = self.strategy_dev.__name__
    
    def load_settings(self):
        self.bt_A_setting = load_json_file(self.bt_setting_A_path)
        self.opt_setting = load_json_file(self.optm_setting_path)
        self.opt_schedule = load_json_file(self.optm_schedule_path)
        self.grid_setting = load_json_file(self.grid_setting_path)
    
    def run_strategy_compare(self, ticker:str, target:str = 'annual_return'):
        """_summary_

        Args:
            ticker (str): commodity ticker
            target (str, optional): dummy input. Defaults to 'annual_return'.
        return:
            None, just show histgram          
        """
        cols = ['max_ddpercent','annual_return','return_std','sharpe_ratio']
        
        s,base_df = self.load_baseline_params(ticker=ticker, target=target)
        
        if not s:
            return
        
        dev_df = self.run_dev_strategy(ticker=ticker, target=target)
        delta_df = dev_df[cols] - base_df[cols]
        
        #plot
        fig, axs = plt.subplots(2,2, figsize=(15, 10), constrained_layout=True)
        fd={'color': 'silver','fontsize': 14}
        for i in [0,1]:
            for j in [0,1]:
                ax = axs[i,j]
                col_ind = i*2 + j
                x = delta_df[cols[col_ind]]
                pos_prob = round(np.sum([e > 0 for e in x]) / len(x),2)
                ax.hist(x, bins = 30, color='c', edgecolor='k', alpha=0.65)
                ax.axvline(0, color='r', linestyle='dashed', linewidth=3)
                ax.set_title('delta of ' + cols[col_ind] + ' (dev - base)',fontdict=fd)
                ax.legend(['0',cols[col_ind]])
                ax.set_ylabel('Freq',fontdict=fd)
                if cols[col_ind] == 'return_std':
                    ax.set_xlabel('Prob. Improvement {0}'.format(1 - pos_prob),fontdict=fd)
                else:
                    ax.set_xlabel('Prob. Improvement {0}'.format(pos_prob),fontdict=fd)
        
    def run_dev_strategy(self, ticker:str, target:str = 'annual_return',
                         bp_df:pd.DataFrame = pd.DataFrame()):
        """for the baseline params, perform backtesting using dev strategy.

        Args:
            ticker (str): the ticker we want to do backtest.
            target (str, optional): used to filter specific set of params. Defaults to 'annual_return'.

        Returns:
            _type_: dataframe of result stats for all base params.
        """
        if len(bp_df) < 1:
            s, base_df = self.load_baseline_params(ticker=ticker,target=target)
        else:
            base_df = bp_df
        
        engine = BacktestingEngine()
        if not s:
            engine.output("no base result found for ticker: {0}, strategy: {1}"
                          .format(ticker, self.strategy_A_name))
            return pd.DataFrame()
        
        s_dev = self.bt_A_setting[ticker]
        strat_setting = s_dev['s_setting']
        bt_setting = s_dev['bt_setting']
        
        #read params and udpate.
        s_date = datetime.strptime(base_df['input_start_date'].values[0],d_format_short)
        e_date = datetime.strptime(base_df['input_end_date'].values[0],d_format_short)
        engine.set_parameters(
            vt_symbol=s_dev['vt_symbol'],
            interval=bt_setting['interval'],
            start=s_date,
            end=e_date,
            rate=bt_setting['rate'],
            slippage=bt_setting['slippage'],
            size=bt_setting['size'],
            pricetick=bt_setting['pricetick'],
            capital=bt_setting['capital']
        )
        engine.add_strategy(self.strategy_dev, strat_setting)
        
        settings = self.generate_group_settings(base_df, strat_setting)
        engine.output("Dev strategy batch run for {0} started:{1} - {2}"\
            .format(s_dev['vt_symbol'],\
                    datetime.strftime(s_date,d_format_short),\
                    datetime.strftime(e_date,d_format_short)))
        result_df = engine.run_group_backtesting(settings)
        engine.output("Dev strategy batch run succeeded.")
        return result_df
        
    def update_baseline_params(self,ticker:str = 'IC',
                                target:str = 'annual_return',
                                start:str = '',
                                end:str = ''):
        baseline_file_name = os.path.join(self.setting_path, self.strategy_A_name + self.baseline_param_file)
        r_df = self.run_params_optimization(ticker=ticker,start=start,end=end,target=target)
        r_df['ticker'] = ticker
        r_df['input_start_date'] = start
        r_df['input_end_date'] = end
        r_df = self.params_filter(r_df, quality='mq')
        if(os.path.exists(baseline_file_name)):
            hist_df = pd.read_csv(baseline_file_name)
            hist_df = hist_df[~((hist_df['ticker'] == ticker) &
                                (hist_df['target'] == target))]
            concat_df=pd.concat([hist_df,r_df])
        else:
            concat_df=r_df
        concat_df.to_csv(baseline_file_name,index=False)
        return concat_df
    
    def run_bt_rolling_optm_allmarket(self, schedule_group:str = 'schedule1', lb_weeks:int = 27, apply_start_shift:int = 10):
        stats_df_lst = []
        daily_df_dict = {}
        for t in self.bt_A_setting.keys():
            s_df, d_df = self.run_backtest_rolling_optm(ticker=t,
                                                        sched_group=schedule_group,
                                                        lb_weeks= lb_weeks,
                                                        apply_start_shift= apply_start_shift)
            s_df['ticker'] = t
            stats_df_lst.append(s_df)
            daily_df_dict[t] = d_df
            #save result on every iteration
            stats_df = pd.concat(stats_df_lst)
            self.write_bt_results_to_excel(stats_df=stats_df,daily_result_dfs=daily_df_dict, filename=self.strategy_A_name + '_roptm_')

    def run_optm_allmarket(self, tickers:List[str] = ['ALL'],target:str = 'annual_return', start:str ='', end:str = ''):
        result = {}
        result_stats = []
        fname =self.strategy_A_name + "_" +  "optm" + '_'.join(tickers) + start + '_' + end
        if tickers[0].lower() == "all":
            #run all
            for t in self.bt_A_setting.keys():
                r_df = self.run_params_optimization(ticker=t,start=start,end=end,target=target)
                r_stats = self.calculate_optimization_stats(r_df)
                r_stats["ticker"] = t
                result_stats.append(r_stats)
                self.write_tab_to_excel(t,r_df,fname)
                result[t] = r_df
        else:
            for t in tickers:
                r_df = self.run_params_optimization(ticker=t,start=start,end=end,target=target)
                r_stats = self.calculate_optimization_stats(r_df)
                r_stats["ticker"] = t
                result_stats.append(r_stats)
                self.write_tab_to_excel(t,r_df,fname)
                result[t] = r_df
        
        self.write_tab_to_excel("sum",pd.DataFrame(result_stats),fname)

        return result, result_stats
        #dump to excel
        #self.write_bt_results_to_excel(sum_df=pd.DataFrame(),result_dfs=result, filename=fname)
                     
    
    def run_backtest_rolling_optm(self, ticker:str = 'IC',
                                  optm_freq:str = '6m',
                                  sched_group:str = 'schedule1',
                                  lb_weeks:int = 27,
                                  apply_start_shift:int = 10):
        """_summary_

        Args:
            ticker (str, optional): commodity ticker. Defaults to 'IC'.
            optm_freq (str, optional): optimization frequency, 6m or 3m. Defaults to '6m'.
            sched_group (str, optional): specify the schedule sub group. Defaults to 'schedule1'.
            lb_weeks (int, optional): specify the look back period for optimization. Defaults to 27.
            apply_start_shift (int, optional): shift the apply_start backward to account for data burning period(strategy depenedent). Defaults to 10.
        Returns:
            _type_: _description_
        """
        s_summary = []
        r_summary = []
        schedule = self.opt_schedule[optm_freq][sched_group]
        
        for d,v in schedule.items():
            if v < 1:
                continue
            train_start = datetime.strptime(d,d_format_short) - timedelta(weeks = lb_weeks)
            train_start = datetime.strftime(train_start,d_format_short)
            train_end = d
            
            apply_start = datetime.strptime(d,d_format_short) - timedelta(days = apply_start_shift)
            apply_start = datetime.strftime(apply_start,d_format_short)
            apply_end = get_next_key(d,schedule)
            if apply_end is None:
                apply_end = datetime.strptime(d,d_format_short) + timedelta(weeks = lb_weeks)
                apply_end = datetime.strftime(apply_end,d_format_short)
                
            try:
                r_df = self.run_params_optimization(ticker=ticker,
                                            start=train_start,
                                            end=train_end)
                opt_params_setting = self.get_optimal_strat_params(r_df,self.bt_A_setting[ticker]['s_setting'])
                
                s,daily_df = self.run_backtest(ticker=ticker,
                                strategy_setting=opt_params_setting,
                                start=apply_start,
                                end=apply_end)
                s['start_date'] = apply_start
                s['end_date'] = apply_end
                s['params_date'] = d
                s.update(opt_params_setting)
                s_summary.append(s)
                r_summary.append(daily_df)
            except Exception as e:
                print(e)
                print('exception for schedule:{0}'.format(d))
        
        #consolidate results
        stats_df = pd.DataFrame(data=s_summary)
        return_df = pd.concat(r_summary)
        return_df = return_df[~return_df.index.duplicated(keep = 'last')] #drop the duplicates if any
        return_df['balance_adj'] = return_df['net_pnl'].cumsum() + self.bt_A_setting[ticker]['bt_setting']['capital']
        
        return stats_df,return_df
    
    def run_backtest(self,ticker:str = 'IC', strategy_setting:dict = {}, start:str = '',end:str = '', show_chart:bool = False):
        """_summary_

        Args:
            ticker (str, optional): commodity ticker. Defaults to 'IC'.
            strategy_setting (dict, optional): strategy setting to overwrite base settting. Defaults to {}.
            start (str, optional): start date in string. Defaults to ''.
            end (str, optional): end date in string. Defaults to ''.

        Returns:
            _type_: stats in dataframe and daily result in dataframe
        """
        s_base = self.bt_A_setting[ticker]
        
        if start == '' or end == '':
            s_date = datetime.strptime(s_base['bt_setting']['start'],d_format_short)
            e_date = datetime.strptime(s_base['bt_setting']['end'],d_format_short)
        else:
            s_date = datetime.strptime(start,d_format_short)
            e_date = datetime.strptime(end,d_format_short)
        if not strategy_setting == {}:
            s_setting = strategy_setting
        else:
            s_setting = s_base['s_setting']
        
        engine = BacktestingEngine()
        engine.output("backtest for {0} started:{1} - {2}".format(s_base['vt_symbol'],\
                            datetime.strftime(s_date,d_format_short),\
                            datetime.strftime(e_date,d_format_short)))
        engine.set_parameters(
            vt_symbol=s_base['vt_symbol'],
            interval=s_base['bt_setting']['interval'],
            start=s_date,
            end=e_date,
            rate=s_base['bt_setting']['rate'],
            slippage=s_base['bt_setting']['slippage'],
            size=s_base['bt_setting']['size'],
            pricetick=s_base['bt_setting']['pricetick'],
            capital=s_base['bt_setting']['capital']
        )
        engine.add_strategy(self.strategy_A, s_setting)
        engine.load_data()
        engine.run_backtesting()
        engine.calculate_result()
        engine.output("backtest ended.")
        
        if show_chart:
            engine.calculate_statistics()
            engine.show_chart()
        
        stats_dict = engine.calculate_statistics(output=False)
        stats_dict['vt_symbol'] = s_base['vt_symbol']
        r_df = engine.daily_df
        r_df= r_df[['close_price','pre_close','trade_count','turnover','slippage',\
                    'total_pnl','net_pnl','balance','return','ddpercent']]
        
        return stats_dict, r_df
    
    def run_params_optimization(self, ticker:str = 'IC',
                                target:str = 'annual_return',
                                start:str = '',
                                end:str = ''):
        """_summary_

        Args:
            ticker (str, optional): commodity simbol. Defaults to 'IC'.
            target (str, optional): target for optmization. Defaults to 'annual_return'.
            start (str, optional): start date in string. Defaults to ''.
            end (str, optional): end date in string. Defaults to ''.

        Returns:
            _type_: optimization result in dataframe
        """
        s_base = self.bt_A_setting[ticker]
        
        if start == '' or end == '':
            s_date = datetime.strptime(s_base['bt_setting']['start'],d_format_short)
            e_date = datetime.strptime(s_base['bt_setting']['end'],d_format_short)
        else:
            s_date = datetime.strptime(start,d_format_short)
            e_date = datetime.strptime(end,d_format_short)
            
        engine = BacktestingEngine()
        engine.set_parameters(
            vt_symbol=s_base['vt_symbol'],
            interval=s_base['bt_setting']['interval'],
            start=s_date,
            end=e_date,
            rate=s_base['bt_setting']['rate'],
            slippage=s_base['bt_setting']['slippage'],
            size=s_base['bt_setting']['size'],
            pricetick=s_base['bt_setting']['pricetick'],
            capital=s_base['bt_setting']['capital']
        )
        engine.add_strategy(self.strategy_A, s_base['s_setting'])
        
        opt_setting = self.generate_optm_settings(target=target,s_base=s_base['s_setting'])
        
        engine.output("optimization for {0} started:{1} - {2}"\
            .format(s_base['vt_symbol'],\
                    datetime.strftime(s_date,d_format_short),\
                    datetime.strftime(e_date,d_format_short)))
        result_df = engine.run_ga_optm_stats(opt_setting)
        engine.output("optimization succeeded.")
        
        return result_df
    
    def run_sensitivity_analysis(self, ticker:str = 'IC',
                                 s_setting_base = {},
                                 start:str = '',
                                 end:str = '',
                                 target:str = 'sharpe_ratio'):
        result_df = self.run_backtest_grid(ticker=ticker,
                                           s_setting_base=s_setting_base,
                                           start= start,
                                           end=end)
        #show plot
        y_name = target
        x_list = result_df['x_param'].unique()
        fig_nrow = int(len(x_list)/2) + len(x_list)%2
        fd={'color': 'silver','fontsize': 14}
        fig, axs = plt.subplots(fig_nrow,2, figsize=(15, 10), constrained_layout=True)
        for i in range(fig_nrow):
            for j in range(2):
                ax = axs[i,j]
                x_ind = i*2 + j
                x_name = x_list[x_ind]
                x = result_df[result_df['x_param'] == x_name][x_name]
                y = result_df[result_df['x_param'] == x_name][y_name]
                z = np.polyfit(x, y, deg=3)
                p = np.poly1d(z)
                y_est = p(x)
                y_err = x.std()
                ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
                ax.plot(x,y,'o',color='tab:red')
                ax.plot(x, y_est, '-')
                ax.set_ylabel(y_name,fontdict=fd)
                ax.set_xlabel(x_name,fontdict = fd)
    
    def run_backtest_grid(self, ticker:str = 'IC',
                                 s_setting_base = {},
                                 start:str = '',
                                 end:str = ''):
        
        s_base = self.bt_A_setting[ticker]
        strat_setting = s_base['s_setting']
        bt_setting = s_base['bt_setting']
        
        if not s_setting_base == {}:
            s_base = s_setting_base
            
        p_domain = self.grid_setting[self.strategy_A_name]['params']
        
        if start == '' or end == '':
            s_date = datetime.strptime(s_base['bt_setting']['start'],d_format_short)
            e_date = datetime.strptime(s_base['bt_setting']['end'],d_format_short)
        else:
            s_date = datetime.strptime(start,d_format_short)
            e_date = datetime.strptime(end,d_format_short)
        
        engine = BacktestingEngine()
        engine.set_parameters(
            vt_symbol=s_base['vt_symbol'],
            interval=bt_setting['interval'],
            start=s_date,
            end=e_date,
            rate=bt_setting['rate'],
            slippage=bt_setting['slippage'],
            size=bt_setting['size'],
            pricetick=bt_setting['pricetick'],
            capital=bt_setting['capital']
        )
        engine.add_strategy(self.strategy_A, strat_setting)
        r_lst = []
        for param, vals in p_domain.items():
            engine.output("grid backtest for param:{0}".format(param))
            settings = self.generate_grid_settings(p_name = param,
                                                   p_range = vals,
                                                   s_base = s_base['s_setting'])
            r_df = engine.run_group_backtesting(settings)
            r_df['x_param'] = param
            r_lst.append(r_df)

        return pd.concat(r_lst)
        
    
    #local utility func
    def calculate_optimization_stats(self, df: pd.DataFrame = None):
        """"""
        # Check for init DataFrame
        if df is None:
            # Set all statistics to 0 if no trade.
            start_date = ""
            end_date = ""
            ar_1_p = 0
            ar_5_p = 0
            ar_10_p = 0
            sr_50_p = 0
            sr_100_p = 0
            sr_150_p = 0
        else:
            # Calculate statistics value
            total_count = len(df)
            start_date = df["start_date"][0]
            end_date = df["end_date"][0]
            ar_1_p = np.sum(df["annual_return"] > 1.0)/total_count
            ar_5_p = np.sum(df["annual_return"] > 5.0)/total_count
            ar_10_p = np.sum(df["annual_return"] > 10.0)/total_count
            sr_50_p = np.sum(df["sharpe_ratio"] > 0.5)/total_count
            sr_100_p = np.sum(df["sharpe_ratio"] > 1.0)/total_count
            sr_150_p = np.sum(df["sharpe_ratio"] > 1.5)/total_count

        statistics = {
            "start_date": start_date,
            "end_date": end_date,
            "total_counts": total_count,
            "annual_return_exceed_1": ar_1_p,
            "annual_return_exceed_5": ar_5_p,
            "annual_return_exceed_10": ar_10_p,
            "sharpe_exceed_0p5": sr_50_p,
            "sharpe_exceed_1": sr_100_p,
            "sharpe_exceed_1p5": sr_150_p,
        }

        # Filter potential error infinite value
        for key, value in statistics.items():
            if value in (np.inf, -np.inf):
                value = 0
            statistics[key] = np.nan_to_num(value)
            
        return statistics

    def get_optimal_strat_params(self, params_df, s_setting_base:dict = {}):
        s_setting_opt = copy(s_setting_base)
        f_df = self.params_filter(params_df,quality='hq')
        if len(f_df) < 10:
            f_df = self.params_filter(params_df,quality='mq')
        if len(f_df) < 10:
            f_df = self.params_filter(params_df,quality='lq')
        if len(f_df) < 5:
            f_df = self.params_filter(params_df)
        q_series = f_df.quantile(q=0.5,interpolation='nearest')
        col_types = f_df.dtypes
        for p in s_setting_opt.keys():
            if p in q_series.keys():
                s_setting_opt[p] = q_series[p].astype(col_types[p])
        return s_setting_opt

    
    def write_bt_results_to_excel(self, sum_df, result_dfs:dict = {}, filename:str = 'default'):
        #write to excel
        output_base_path = 'output'
        if not os.path.exists(output_base_path):
            os.makedirs(output_base_path)
        
        fname = filename
        if filename == 'default':
            fname = self.strategy_A_name
            
        with pd.ExcelWriter(os.path.join(output_base_path, fname + '_bt_output.xlsx')) as writer:  
            sum_df.to_excel(writer, sheet_name='summary')
            for k,v_df in result_dfs.items():
                v_df.to_excel(writer, sheet_name=k)
    
    def write_tab_to_excel(self, tab_name:str, data_df:pd.DataFrame, filename:str = 'default'):
        #write to excel
        output_base_path = 'output'
        if not os.path.exists(output_base_path):
            os.makedirs(output_base_path)
        
        fname = filename
        if filename == 'default':
            fname = self.strategy_A_name
        file_name_full = os.path.join(output_base_path, fname + '_bt_output.xlsx')

        if not os.path.exists(file_name_full):
             with pd.ExcelWriter(file_name_full) as writer:
                data_df.to_excel(writer, sheet_name=tab_name)
        else:
            with pd.ExcelWriter(file_name_full, mode = 'a', if_sheet_exists = "replace") as writer:  
                data_df.to_excel(writer, sheet_name=tab_name)
    
    def parser_param(self, p:str = ''):
        results = []
        is_float = False
        if '.' in p:
            is_float = True
        p_arry = p.split(':')
        if is_float:
            results = [float(p) for p in p_arry]
        else:
            results = [int(p) for p in p_arry]
        return results
    
    def params_filter(self, data_df, quality:str = '') -> pd.DataFrame:
        filter_df = pd.DataFrame()
        if quality.lower() == 'hq':
            filter_df = data_df.loc[(data_df['annual_return']>5) & (data_df['sharpe_ratio']>1.5)]
        elif quality.lower() == 'mq':
            filter_df = data_df.loc[(data_df['annual_return']>3) & (data_df['sharpe_ratio']>1)]
        elif quality.lower() == 'lq':
            filter_df = data_df.loc[(data_df['annual_return']>1) & (data_df['sharpe_ratio']>0.5)]
        else:
            filter_df = data_df.iloc[:10]
        return filter_df
    
    def generate_group_settings(self, base_df:pd.DataFrame, strat_setting = {}):
        settings = []
        for index, row in base_df.iterrows():
            s = copy(strat_setting)
            for k,v in s.items():
                if k in base_df.columns:
                    s[k] = row[k]
            settings.append(s)
        return settings
    
    def generate_optm_settings(self, target:str, s_base:Dict):
        p_domain = self.opt_setting[self.strategy_A_name]['params']
        opt_setting = OptimizationSetting()
        opt_setting.set_target(target)
        # for p_name,p_val in p_domain.items():
        #     p_arry = self.parser_param(p_val)
        #     if len(p_arry) == 3:
        #         opt_setting.add_parameter(p_name, p_arry[0],p_arry[1],p_arry[2])
        #     elif len(p_arry) ==1:
        #         opt_setting.add_parameter(p_name, p_arry[0])
        #     else:
        #         print("please check strategy optimization param:{0}".format(p_name))
        
        #corrected
        for k,v in s_base.items():
            if k in p_domain.keys():
                p_arry = self.parser_param(p_domain[k])
                if len(p_arry) == 3:
                    opt_setting.add_parameter(k, p_arry[0],p_arry[1],p_arry[2])
                elif len(p_arry) ==1:
                    opt_setting.add_parameter(k, p_arry[0])
                else:
                    print("please check strategy optimization param:{0}".format(k))
            else:
                opt_setting.add_parameter(k,v)
        
        return opt_setting
    
    def generate_grid_settings(self,
                               p_name:str,
                               p_range:str,
                               s_base:Dict):
        opt_setting = OptimizationSetting()
        opt_setting.set_target('annual_return') #dummy target.

        for k,v in s_base.items():
            if k == p_name:
                p_arry = self.parser_param(p_range)
                if len(p_arry) == 3:
                    opt_setting.add_parameter(k, p_arry[0],p_arry[1],p_arry[2])
                else:
                    print("please check grid setting for param: {0}".format(k))
            else:
                opt_setting.add_parameter(k,v)
        
        return opt_setting.generate_settings()
                
    def load_baseline_params(self,ticker:str, target:str):
        
        result = pd.DataFrame()
        baseline_file_name = os.path.join(self.setting_path, self.strategy_A_name + self.baseline_param_file)
        if(not os.path.exists(baseline_file_name)):
            print('baseline param file not found.')
            return False,result
        
        base_df = pd.read_csv(baseline_file_name)
        base_df = base_df[(base_df['ticker'] == ticker) &
                                (base_df['target'] == target)].reset_index()
        
        if(len(base_df)< 1):
            print("no base result found for ticker: {0}, strategy: {1}"
                          .format(ticker, self.strategy_A_name))
            return False,result
        
        return True,base_df