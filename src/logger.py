import os
import sys
import logging as logging
import datetime
import time
from src.config import Config



def get_logger(logger_name: str= "default", save_file: bool= True):
    '''
    Usage:
    log = get_logger("{Script name}")
    log.info("{message}")
    '''
    date = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y%m%d')
    log_path = Config['log_dir']
    handlers = []
    if save_file:
        os.makedirs(log_path, exist_ok=True)
        file_handler = logging.FileHandler(
                            filename= f'{log_path}/{date}.log',
                            mode= 'a',
                            encoding= 'utf-8'
                        )
        handlers.append(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    handlers.append(stream_handler)

    logging.basicConfig(
        level= logging.INFO,
        format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers= handlers
        )
    logging.getLogger("kafka.conn").disabled = True
    logging.getLogger("kafka.client").disabled = True
    logging.getLogger("kafka.consumer.subscription_state").disabled = True
    logging.getLogger("kafka.coordinator.consumer").disabled = True
    logging.getLogger("kafka.consumer.fetcher").disabled = True

    return logging.getLogger(logger_name)
