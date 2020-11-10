import logging

def get_logger(logger_name, level=logging.INFO, stream_handler=True):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if stream_handler:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.info('Starting logger ...')

    return logger