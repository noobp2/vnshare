from pathlib import Path
from datetime import datetime
from sys import path
from vnpy.trader.constant import Interval
from vnpy.trader.optimize import OptimizationSetting
from strategies.statistical_arb_strategy import SprdAtrMaStrategyBT
from strategies.backtest import SpreadBacktestEngine

def main():
    setting = {"atr_window": 2, "atr_dev": 0.4, "max_pos": 2, "pay_up": 5, "interval": 5}
    spread_setting ={
        "name": "IC_1st_2nd",
        "leg_settings": [
            {
                "variable":"A",
                "vt_symbol":"IC88.CFFEX",
                "trading_direction":1,
                "trading_multiplier":1
            },
            {
                "variable":"B",
                "vt_symbol":"IC88A2.CFFEX",
                "trading_direction":-1,
                "trading_multiplier":-1
            }
        ],
        "price_formula": "A-B",
        "active_symbol":"IC88A2.CFFEX",
        "min_volume":1.0
    }

    engine = SpreadBacktestEngine()
    engine.clear_data()
    engine.set_parameters(
        interval = Interval.MINUTE,
        start = datetime(2021,11,2),
        end = datetime(2021,12,24),
        rate = 0.26/10000,
        slippage = 0.2,
        size = 200,
        pricetick = 0.2,
        capital = 600000
    )
    engine.add_spread(spread_setting)
    engine.add_strategy(SprdAtrMaStrategyBT,setting)
    engine.load_data()
    engine.run_backtesting()
    result_df = engine.calculate_result()
    stats_dict = engine.calculate_statistics()
    engine.show_chart()
    
    #run optimization
    # opt_setting = OptimizationSetting()
    # opt_setting.set_target('total_return')
    # opt_setting.add_parameter('atr_window', 2, 9, 1)
    # opt_setting.add_parameter('atr_dev', 0.4, 0.5, 0.1)
    # opt_setting.add_parameter('max_pos', 2)
    # opt_setting.add_parameter('payup', 5)
    # opt_setting.add_parameter('interval', 5)
    # engine.clear_data()
    # engine.run_ga_optimization(opt_setting)

if __name__ == "__main__":
    print("current working directory is: {0}".format(Path.cwd()))
    main()