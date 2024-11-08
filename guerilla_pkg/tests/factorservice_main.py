from guerilla.factorresearch.factors.multi_factor import MultiFactor
from guerilla.factorresearch.factors.synthetic_factor import QtlRumiFactor
from guerilla.dataservice.dsUtil import FREQ,SOURCE
import guerilla.core.config as cfg
import numpy as np
import pandas as pd
import warnings
from guerilla.core.utility import logger
warnings.filterwarnings('ignore')

params_st = {"ma_win_short": 5,
            "ma_win_long": 20,
            "vol_win_short":5,
            "vol_win_long":10,
            "rsi_win_short":7,
            "rsi_win_long":14,
            "qtl_win":10,
            "smooth_win":5}

params_mt = {"ma_win_short": 5,
            "ma_win_long": 60,
            "vol_win_short":5,
            "vol_win_long":10,
            "rsi_win_short":7,
            "rsi_win_long":14,
            "qtl_win":50,
            "smooth_win":5}

params_lt = {"ma_win_short": 5,
            "ma_win_long": 100,
            "vol_win_short":5,
            "vol_win_long":10,
            "rsi_win_short":7,
            "rsi_win_long":14,
            "qtl_win":100,
            "smooth_win":5}

params_slt = {"ma_win_short": 5,
            "ma_win_long": 200,
            "vol_win_short":5,
            "vol_win_long":10,
            "rsi_win_short":7,
            "rsi_win_long":14,
            "qtl_win":200,
            "smooth_win":5}

params_qlt_rumi = {"QTL_WIN": 200, 
                      "RUMI_WIN": 20,
                      "MA_WIN_SHORT":5,
                      "MA_WIN_LONG":60}


def main():
    multifactor_load()
    qtlrumi_load()

def multifactor_load():
    log = logger(__name__,'multi_factor_calc.log')
    config = cfg.config_parser()
    tickers = config["research"]["china_fut_universe"]
    params_lst = [params_st,params_mt,params_lt,params_slt]
    #load factors daily
    log.info('multi factor calc started.')
    for ticker in tickers:
        log.info('multi factor calc daily for ticker {0}.'.format(ticker))
        mf_obj = MultiFactor(ticker=ticker,f=FREQ.DAILY)
        mf_obj.init_data()
        for p in params_lst:
            mf_obj.setup_params(p)
            mf_obj.calc_factor()
            mf_obj.save_factor_to_db()
    
    
    #load factors hourly
    for ticker in tickers:
        log.info('multi factor calc hourly for ticker {0}.'.format(ticker))
        mf_obj = MultiFactor(ticker=ticker,f=FREQ.HOURLY)
        mf_obj.init_data()
        for p in params_lst:
            mf_obj.setup_params(p)
            mf_obj.calc_factor()
            mf_obj.save_factor_to_db()
    
    log.info('multi factor calc ended.')

def qtlrumi_load():
    log = logger(__name__,'qtlrumi_factor_calc.log')
    config = cfg.config_parser()
    tickers = config["research"]["china_fut_universe"]
    #load qlt_rumi factor
    log.info('qtlrumi factor calc started.')
    for ticker in tickers:
        log.info('qtlrumi factor calc for ticker {0}.'.format(ticker))
        f_obj = QtlRumiFactor(vn_ticker=ticker,f=FREQ.DAILY)
        f_obj.setup_params(params_qlt_rumi)
        f_obj.init_data()
        f_obj.calc_factor()
        f_obj.save_factor_to_db()
    log.info('qtlrumi factor calc ended.')
    
if __name__ == '__main__':
     # execute only if run as a script
     main()