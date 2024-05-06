import logging

import pandas as pd
import tqdm

pd.options.mode.copy_on_write = True


# Configure Tqdm with default logging module
# Taken from: https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit/38739634#38739634
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


# Configure the root formatter
logging.basicConfig(
    format="{asctime}|{levelname}|{module}.{funcName}: {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    level=logging.INFO,
    handlers=[TqdmLoggingHandler()],
)
