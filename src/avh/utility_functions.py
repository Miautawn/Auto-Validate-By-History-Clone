from typing import Tuple, List, Union, Any, Optional, Callable, Dict, Set
from collections.abc import Iterable
from numbers import Number
import time

import numpy as np

def timeit_decorator(func):
    """
    Meassures elapsed time of the function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time} seconds to execute.")
        return result

    return wrapper

def diff(data: Iterable, period: int) -> np.array:
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
def safe_log(x: Union[Number, Iterable]) -> np.array:
    """
    Perform log(x), but with safeguards
        against cases where x == 0
    """
    data = np.array(x)
    log_data = np.log(
        data, out=np.zeros_like(data, dtype=np.float32), where=(data != 0)
    )

    if log_data.shape:
        return log_data
    return log_data.item()