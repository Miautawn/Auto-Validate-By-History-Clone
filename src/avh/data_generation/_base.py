from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from avh.aliases import Seed


class DataColumn(ABC):
    """
    Abstract column class.
    Blueprint for generating data of specified type & behavior.

    Parameters
    ----------
    name: str
        The name of the column in the final output

    Attributes
    ----------
    name: str
        The name of the column in the final output
    dtype: Any
        The dtype of the column in the final output
    """

    def __init__(self, name: str, random_state: Seed = None):
        """
        If subclassed, the child should call the parent constructor
        """
        self.name = name
        self.random_state = random_state

    @property
    @abstractmethod
    def dtype(self) -> Any: ...

    @abstractmethod
    def generate(self, n: int, i: int = 0) -> pd.Series:
        """
        Output the generated Series of length n

        Parameters
        ----------
        n: int
            The length of the column to output
        i: int
            The i'th call of the column generation.
            Can be useful for modifying the genration parameters based on "time"
        """
        ...


class NumericColumn(DataColumn):
    """
    Abstract numeric column class.
    Blueprint for generating data for specifically numeric columns.

    Parameters
    ----------
    name: str
        The name of the column in the final output
    minimum: Optional[float]
        The minimum value this column should output
    maximum: Optional[float]
        The maximum value this column should ouptput
    dtype: Union[np.float32, np.int32]
        The dtype of this numeric column.
        Currently only accepts np.float32 or np.int32
    parameter_function: Optional[Callable]
        A function which accepts and returns the data generation parameters.
        Can be used to create "moving" columns, where column parameters are
            changed each call. If none, this function won't be applied
        The function must have the following definition:
            ```
            def func(n, i, *args):
                return *args
            ```
            Where:
            - 'n' is the column size during this generation call
            - 'i' is the iteration count (default is 0)
            - '*args' are the generaton parameters
    """

    def __init__(
        self,
        name: str,
        minimum: float = -np.inf,
        maximum: float = np.inf,
        scale: float = 1.0,
        shift: float = 0.0,
        dtype: type = np.float32,
        parameter_function: Optional[Callable] = None,
        **kwargs
    ):
        """
        If subclassed, the child should call the parent constructor
        """
        super().__init__(name, **kwargs)
        assert (
            dtype == np.float32 or dtype == np.int32
        ), "Numeric column currently can only be of type np.float32 or np.int32"

        self._minimum = minimum
        self._maximum = maximum
        self._scale = scale
        self._shift = shift
        self._dtype = dtype
        self._parameter_function = parameter_function

    @property
    def dtype(self):
        return self._dtype

    @abstractmethod
    def _update_parameters(self, n: int, i: int) -> None:
        """
        Template method for applying the `parameter_function`
            to data generation parameters.

        Parameters
        ----------
        n: int
            The length of the column
        i: int
            The i'th call of the column generation.
        """
        ...

    @abstractmethod
    def _generate(self, n: int) -> pd.Series:
        """
        Template method for generating data.
        """
        ...

    def generate(self, n: int, i: int = 0) -> pd.Series:
        data = self._generate(n) * self._scale + self._shift
        data = np.clip(data, self._minimum, self._maximum)

        if self._parameter_function is not None:
            self._update_parameters(n, i)

        return pd.Series(
            data.astype(self.dtype),
            name=self.name,
            dtype=self.dtype,
        )


class CategoricalColumn(DataColumn):
    """
    Abstract categorical/string column class.
    Blueprint for generating data for specifically string based columns.

    Note: this class uses the dtype of "object", since pandas Categorical dtype
        is specifically designed to protect against DQ issues, thus isn't flexible
        for our use case.

    Parameters
    ----------
    name: str
        The name of the column in the final output
    """

    def __init__(self, name: str):
        super().__init__(name)

    @property
    def dtype(self):
        return "string[pyarrow]"

    @abstractmethod
    def _generate(self, n: int) -> pd.Series:
        """
        Template method for generating data.
        """
        ...

    def generate(self, n: int, i: int = 0) -> pd.Series:
        data = self._generate(n)
        return pd.Series(
            data,
            name=self.name,
            dtype=self.dtype,
        )
