from multiprocessing import Process
from TRADE_HIGH_NXT import FOREX_TRADING

import logging


def setup_logger(symbol):

    logger = logging.getLogger(symbol)

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        formatter = logging.Formatter(f'%(asctime)s | [{symbol}] | %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger


def run_symbol(symbol):

    logger = setup_logger(symbol)

    logger.info(f'CURRENTLY RUNNING THIS SYMBOL [[ {symbol} ]]')

    FOREX_TRADING(SYMBOL=symbol,logger=logger)

# "EURUSD", "USDJPY", "GBPUSD", "XAUUSD", 'USDCAD', 'AUDCAD', 'NZDUSD', 'AUDUSD','USDCHF'

if __name__ == "__main__":

    broker = 'NairaProp22'

    symbols = ["AUDCAD","AUDJPY","AUDUSD","CADJPY","EURJPY","EURUSD",'GBPUSD',"NZDUSD","USDJPY","USDSEK"]

    if broker =='NairaProp':
        
        symbols = [x+'m' for x in symbols]

    processes = []

    for symbol in symbols:

        p = Process(target=run_symbol,args=(symbol,))

        p.start()

        processes.append(p)

    for p in processes:
        p.join()


