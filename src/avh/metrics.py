from abc import ABC, abstractmethod
from typing import List, Union, Any
import time

import pandas as pd
import numpy as np

class Metric(ABC):
    """
    Metric is a static utility class
    """

    @classmethod
    def is_column_compatable(self, dtype: Any) -> bool:
        return True

    @classmethod
    def calculate(
        self, data: Union[pd.Series, List[pd.Series]]
    ) -> Union[float, List[float]]:
        """
        Method for calculating the target metric from given data
        """
        if isinstance(data, list):
            return list(map(self._calculate, data))
        return self._calculate(data)

    @classmethod
    @abstractmethod
    def _calculate(self, data: pd.Series) -> float:
        ...

    @classmethod
    def _is_empty(self, data: pd.Series) -> bool:
        if data.count() == 0:
            return True
        return False

class NumericMetric(Metric):
    @classmethod
    def is_column_compatable(self, dtype: Any) -> bool:
        return pd.api.types.is_numeric_dtype(dtype)

class CategoricalMetric(Metric):
    @classmethod
    def is_column_compatable(self, dtype: Any) -> bool:
        return not pd.api.types.is_numeric_dtype(dtype)

class RowCount(Metric):
    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        return len(column)

class DistinctRatio(Metric):
    """
    I don't like that this is also a numeric metric!
    Since it's almost always treated as a statistical invariate
    because yeah, floating point numbers will mostly be unique all the time!!!
    """

    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        return column.nunique(dropna=False) / len(column)

class CompleteRatio(Metric):
    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        if column.empty:
            return 0.0
        return column.count() / column.size

class Min(NumericMetric):
    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        if self._is_empty(column):
            return 0.0
        return column.min()

class Max(NumericMetric):
    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        if self._is_empty(column):
            return 0.0
        return column.max()


class Mean(NumericMetric):
    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        if self._is_empty(column):
            return 0.0
        return column.mean()

class Median(NumericMetric):
    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        if self._is_empty(column):
            return 0.0
        return np.nanmedian(column)


class Sum(NumericMetric):
    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        if self._is_empty(column):
            return 0.0
        return column.sum()

class Range(NumericMetric):
    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        if self._is_empty(column):
            return 0.0
        return column.max() - column.min()


class DistinctCount(CategoricalMetric):
    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        return column.nunique(dropna=False)


class MeanStringLength(CategoricalMetric):
    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        if self._is_empty(column):
            return 0.0
        return np.nanmean(column.str.len())

class MeanDigitLength(CategoricalMetric):
    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        if self._is_empty(column):
            return 0.0
        return np.nanmean(column.str.count(r"\d"))


class MeanPunctuationLength(CategoricalMetric):
    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        if self._is_empty(column):
            return 0.0
        return np.nanmean(column.str.count(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]"))