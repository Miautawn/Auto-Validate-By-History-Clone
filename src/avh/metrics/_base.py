from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import pandas as pd


class Metric(ABC):
    """
    Metric is a static utility class
    """

    @classmethod
    def is_column_compatable(self, dtype: Any) -> bool:
        return True

    @classmethod
    @abstractmethod
    def calculate(self, *args) -> Union[float, List[float]]:
        """
        Method for calculating the target metric from given data
        """
        ...


# Metric dtype mixins
class NumericMetricMixin:
    @classmethod
    def is_column_compatable(self, dtype: Any) -> bool:
        return pd.api.types.is_numeric_dtype(dtype)


class CategoricalMetricMixin:
    @classmethod
    def is_column_compatable(self, dtype: Any) -> bool:
        return not pd.api.types.is_numeric_dtype(dtype)


# Different input type subclasses
class SingleDistributionMetric(Metric):
    @classmethod
    def calculate(self, data: Union[pd.Series, List[pd.Series]]) -> Union[float, List[float]]:
        """
        Method for calculating the target metric from given data
        """
        if isinstance(data, list):
            return list(map(self._calculate, data))
        return self._calculate(data)

    @classmethod
    @abstractmethod
    def _calculate(self, data: pd.Series) -> float: ...

    @classmethod
    def _is_empty(self, data: pd.Series) -> bool:
        if data.count() == 0:
            return True
        return False


class TwoDistributionMetric(Metric):
    @classmethod
    def calculate(
        self, data: Union[pd.Series, List[pd.Series]], referene_data: Optional[pd.Series] = None
    ) -> Union[float, List[float]]:
        """
        Funny magic
        """
        if referene_data is not None:
            return self._calculate(data, referene_data)
        return [self._calculate(data[i], data[i - 1]) for i in range(1, len(data))]

    @classmethod
    @abstractmethod
    def _calculate(self, new_sample: pd.Series, old_sample: pd.Series) -> float: ...

    @classmethod
    def _is_empty(self, data: pd.Series, reference_data: pd.Series) -> bool:
        if data.count() == 0 or reference_data.count() == 0:
            return True
        return False
