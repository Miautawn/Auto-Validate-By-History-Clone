import math
from typing import List

import pandas as pd
import numpy as np
from sklearn.base import check_is_fitted
from scipy import integrate

import avh.metrics as metrics
from avh.constraints._base import Constraint

class ConstantConstraint(Constraint):
    """
    Concrete Constraint subclass,
        which operates on manually provided threshold values
    """

    def __init__(self, metric: metrics.Metric, u_lower: float, u_upper: float, expected_fpr):
        super().__init__(metric)

        # technically not following the sklearn style guide :(
        self.u_upper_ = u_upper
        self.u_lower_ = u_lower
        self.expected_fpr_ = expected_fpr

    def _fit(self, metric_history: List[float], **kwargs):
        return self


class ChebyshevConstraint(Constraint):
    """
    Chebyshev!
    """

    def _fit(self, metric_history: List[float], raw_history: List[pd.Series], beta: float, strategy: str = "raw", **kwargs):
        assert strategy in ["raw", "std"], "Strategy can only be 'raw' or 'std'"
        self.last_reference_sample_ = raw_history[-1]

        mean = np.nanmean(metric_history)
        var = np.nanvar(metric_history)

        beta = beta if strategy == "raw" else np.sqrt(var) * beta

        self.u_upper_ = mean + beta
        self.u_lower_ = mean - beta

        if var == 0:
            self.expected_fpr_ = 0.0
        else:
            self.expected_fpr_ = var / beta**2

    def predict(self, column: pd.Series, **kwargs) -> bool:
        check_is_fitted(self)

        if issubclass(self.metric, metrics.SingleDistributionMetric):
            m = self.metric.calculate(column)
        else:
            m = self.metric.calculate(column, self.last_reference_sample_)

        return self._predict(m, **kwargs)

class CantelliConstraint(Constraint):
    """
    Cantelli!
    """

    compatable_metrics = (
        metrics.EMD,
        metrics.KsDist,
        metrics.CohenD,
        metrics.KlDivergence,
        metrics.JsDivergence
    )

    def _fit(self, metric_history: List[float], raw_history: List[pd.Series], beta: float, strategy: str = "raw", **kwargs):
        assert strategy in ["raw", "std"], "Strategy can only be 'raw' or 'std'"
        self.last_reference_sample_ = raw_history[-1]

        mean = np.nanmean(metric_history)
        var = np.nanvar(metric_history)

        beta = beta if strategy == "raw" else np.sqrt(var) * beta

        self.u_upper_ = mean + beta
        self.u_lower_ = 0

        if var == 0:
            self.expected_fpr_ = 0.0
        else:
            self.expected_fpr_ = var / (var + beta**2)

    def predict(self, column: pd.Series, **kwargs) -> bool:
        check_is_fitted(self)

        m = self.metric.calculate(column, self.last_reference_sample_)
        prediction = self._predict(m, **kwargs)

        return prediction


class CLTConstraint(Constraint):
    compatable_metrics = (
        metrics.RowCount,
        metrics.Mean,
        metrics.MeanStringLength,
        metrics.MeanDigitLength,
        metrics.MeanPunctuationLength,
        metrics.CompleteRatio,
    )

    def _bell_function(sefl, x):
        return math.pow(math.e, -(x**2))

    def _fit(self, metric_history: List[float], beta: float, strategy: str = "raw", **kwargs):
        assert strategy in ["raw", "std"], "Strategy can only be 'raw' or 'std'"

        mean = np.nanmean(metric_history)
        std = np.nanstd(metric_history)

        beta = beta if strategy == "raw" else std * beta

        self.u_upper_ = mean + beta
        self.u_lower_ = mean - beta

        if std == 0:
            self.expected_fpr_ = 0.0
        else:
            satisfaction_p = (2 / np.sqrt(math.pi)) * (
                integrate.quad(self._bell_function, 0, beta / (np.sqrt(2) * std))[0]
            )
            self.expected_fpr_ = 1 - satisfaction_p
