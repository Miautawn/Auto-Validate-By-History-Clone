import logging
import time
from collections.abc import Iterable
from numbers import Number
from typing import Any, Union

import numpy as np


def debug_timeit(local_logger_name=None):
    def decorator(function):
        def wrapper(*args, **kwargs):
            local_logger = logging.getLogger(local_logger_name)

            start_time = time.time()
            result = function(*args, **kwargs)
            end_time = time.time()
            elapsed_time_ms = (end_time - start_time) * 1000

            local_logger.debug(
                f"{function.__module__}.{function.__name__} Took {elapsed_time_ms:.4f} ms to execute."
            )
            return result

        return wrapper

    return decorator


def diff(data: Iterable, period: int) -> np.ndarray:
    """
    Perform difference of the elements separated by the period.
    (Useful for time series differencing)

    The resulting array is prepended with the np.nan
        to keep the shap of the original array
    """
    data = np.array(data)
    if period == 0:
        return data
    return np.concatenate((np.full(period, np.nan), data[period:] - data[:-period]))


def function_repr(repr):
    """
    Decorator function to set the '__function_repr__' field
        for the targetted function.

    Used in scikit-learn representations.
    """

    def wrapper(func):
        setattr(func, "__function_repr__", repr)
        return func

    return wrapper


@function_repr("identity")
def identity(x: Any) -> Any:
    """
    Identity function, returns the input
    """
    return x


@function_repr("log")
def safe_log(x: Union[Number, Iterable]) -> Union[Number, np.ndarray]:
    """
    Perform log(x), but with safeguards
        against cases where x == 0
    """
    data = np.array(x)
    log_data = np.log(data, where=(data != 0))

    if log_data.shape:
        return log_data
    return log_data.item()
