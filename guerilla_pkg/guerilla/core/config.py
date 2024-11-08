import json
import os
DEFAULT_CONFIG = {
    "dataset": {
        "base_path": "E:\\dataset\\",
        "rel_base_path": ".\\dataset\\",
        "fut_info_folder": "future_base_info",
        "fut_gen_folder": "future_generic_hist",
        "fut_ct_folder": "future_contract_hist",
        "ticker_gen_folder":"ticker_generic_hist",
        "yh_ticker_list_file":"yahoo_tickers.csv"
    },
    "database":{
        "default_db":"database.db",
        "qr_db" : "quantresearch.db",
        "trade_db" : "trades.db",
        "risk_db" : "risk.db"
    },
    "dataprovider":{
        "tq":{
            "user":"xxxx",
            "pwd":"xxxx"
            }
    },
    "ems":{
        "sender":"xxxx@xxxx",
        "pwd":"XXXXX"
    },
    "research": {
        "base_path": "E:\\qr\\",
        "rel_base_path": ".\\qr\\",
        "factor_output": "factor_output",
        "china_fut_universe": ["IC88A.CFFEX","MA88A.CZCE","SA88A.CZCE","FG88A.CZCE",
        "SR88A.CZCE","CF88A.CZCE","ZN88A.SHFE","SP88A.SHFE",
        "AL88A.SHFE","OI88A.CZCE","RM88A.CZCE","SM88A.CZCE",
        "RU88A.SHFE","HC88A.SHFE","FU88A.SHFE","TA88A.CZCE",
       "CU88A.SHFE","BU88A.SHFE","V88A.DCE","RB88A.SHFE",
       "Y88A.DCE","C88A.DCE","FB88A.DCE","I88A.DCE",
        "JM88A.DCE","L88A.DCE","M88A.DCE","PP88A.DCE",
        "EG88A.DCE","EB88A.DCE","AP88A.CZCE","SF88A.CZCE",
        "AG88A.SHFE","AU88A.SHFE","A88A.DCE","B88A.DCE",
        "JD88A.DCE","IM88A.CFFEX","IH88A.CFFEX","IF88A.CFFEX",
         "PF88A.CZCE","CS88A.DCE","PK88A.CZCE","P88A.DCE",
         "UR88A.CZCE","EC88A.INE"],
         "china_fut_pricktick":{
            "IF":["0.2","300"],
            "IC":["0.2","200"],
            "IH":["0.2","300"],
            "IM":["0.2","200"],
            "T":["0.005","10000"],
            "TF":["0.005","10000"],
            "TS":["0.005","20000"],
            "TL":["0.01","10000"],
            "AU":["0.02","1000"],
            "AG":["1","15"],
            "CU":["10","5"],
            "AL":["5","5"],
            "ZN":["5","5"],
            "PB":["5","5"],
            "NI":["10","1"],
            "SN":["10","1"],
            "RB":["1","10"],
            "WR":["1","10"],
            "I":["0.5","100"],
            "HC":["1","10"],
            "SS":["5","5"],
            "SF":["2","5"],
            "SM":["2","5"],
            "JM":["0.5","60"],
            "J":["0.5","100"],
            "ZC":["0.2","100"],
            "FG":["1","20"],
            "SP":["2","10"],
            "FU":["1","10"],
            "LU":["1","10"],
            "SC":["0.1","1000"],
            "BC":["10","5"],
            "EC":["0.1","50"],
            "BU":["1","10"],
            "PG":["1","20"],
            "RU":["5","10"],
            "NR":["5","10"],
            "L":["1","5"],
            "TA":["2","5"],
            "V":["1","5"],
            "EG":["1","10"],
            "MA":["1","10"],
            "PP":["1","5"],
            "EB":["1","5"],
            "UR":["1","20"],
            "SA":["1","20"],
            "C":["1","10"],
            "A":["1","10"],
            "CS":["1","10"],
            "B":["1","10"],
            "M":["1","10"],
            "Y":["2","10"],
            "RM":["1","10"],
            "OI":["1","10"],
            "P":["2","10"],
            "CF":["5","5"],
            "SR":["1","10"],
            "JD":["1","10"],
            "AP":["1","10"],
            "CJ":["5","5"],
            "PF":["2","5"],
            "PK":["2","5"],
            "AO":["1","20"],
            "LC":["50","1"],
            "SI":["5","5"],
            "BR":["5","5"],
            "RR":["1","10"],
            "LH":["5","16"],
            "PX":["2","5"],
            "SH":["1","30"]
         }
    },
    "trading": {
        "ch_fut_u10":["CZCE.SR","DCE.jd","DCE.pg","CZCE.AP",
                      "CZCE.UR","DCE.eg","CZCE.SA","CZCE.TA",
                      "CZCE.SF","DCE.pp","CZCE.SM","CZCE.RM",
                      "INE.ec","DCE.c","SHFE.hc","DCE.m",
                      "SHFE.rb","CZCE.FG","SHFE.fu","CZCE.MA",
                      "SHFE.sp","SHFE.ag","SHFE.bu","DCE.v",
                      "DCE.l"],
        "ch_fut_u10_st":["DCE.v"]
    }
}

def config_parser():
    #config_path = __file__.replace('config.py','configuration.json')
    config_path = os.path.join(os.getcwd(), 'configuration.json')
    if not os.path.exists(config_path):
        with open(config_path,"w") as json_data_file:
            json.dump(DEFAULT_CONFIG, json_data_file,ensure_ascii=True, indent=4)
            
    with open(config_path) as json_data_file:
        data = json.load(json_data_file)
    return data
