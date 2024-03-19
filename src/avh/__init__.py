import logging

# configure the root formatter
logging.basicConfig(
    format='{asctime}|{levelname}|{name}: {message}',
    datefmt = "%Y-%m-%d %H:%M:%S",
    style="{", level=logging.INFO
    )