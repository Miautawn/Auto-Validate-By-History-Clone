from typing import Callable, List, Optional, Tuple, cast

import pandas as pd
from sklearn.base import BaseEstimator, check_is_fitted

import avh.metrics as metrics
import avh.utility_functions as utility_functions


class Constraint(BaseEstimator):
    """
    Constraint Predictor entity class.
    It acts as a general abtraction for doing inference with Metric.

    The Constraint entity needs to have the following attributes:
        * compatable_metrics - a tuple of compatable metric classes.
            By default, all (sub)classes of type Metric are compatable.
        * u_upper_ - threshold for triggering the constraint if Metric goes above it
        * u_lower_ - threshold for triggering the constraint if Metric goes below it
        * expected_fpr - expected false positive rate once constraint is fitted.
        * metric_history_ - H(C) = {M(C1), M(C2), ..., M(C3)}

    The Constraint entity needs to have the following methods:
        * fit - prepare the constraint for inference.
        * predict - given a value, check if it violates the constraint.
    """

    compatable_metrics: Tuple[metrics.MetricType, ...] = (metrics.Metric,)

    @classmethod
    def is_metric_compatable(self, metric: metrics.MetricType) -> bool:
        return issubclass(metric, self.compatable_metrics)

    def __init__(
        self,
        metric: metrics.MetricType,
        time_differencing_window: int = 0,
        metric_preprocessing_function: Callable = utility_functions.identity,
    ):
        self.metric = metric
        self.time_differencing_window = time_differencing_window
        self.metric_preprocessing_function = metric_preprocessing_function

    def __repr__(self):
        if hasattr(self, "_is_fitted") and self._is_fitted:
            metric_repr = self._get_metric_repr()
            return "{name}({u_lower:0.4f} <= {metric} <= {u_upper:0.4f}, FPR = {fpr:0.4f})".format(
                name=self.__class__.__name__,
                u_lower=self.u_lower_,
                metric=metric_repr,
                u_upper=self.u_upper_,
                fpr=self.expected_fpr_,
            )
        else:
            return super().__repr__()

    def _get_metric_repr(self):
        metric_repr = self.metric.__name__
        preprocessng_func_repr = self.metric_preprocessing_function.__function_repr__
        if preprocessng_func_repr != "identity":
            metric_repr = "{}({})".format(preprocessng_func_repr, metric_repr)
        if self.time_differencing_window != 0:
            metric_repr = "{}.diff({})".format(metric_repr, self.time_differencing_window)
        return metric_repr

    def fit(
        self,
        X: List[pd.Series],
        y=None,
        metric_history: Optional[List[float]] = None,
        hotload_metric_history: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        X - raw data
        metric_history - calculated metric history without any pre/post-processing
        hotload_metric_history - final metric history including pre/post-processing
            and timeseries differencing. Constraint interval will be estimated directly on this.
        """

        assert self.is_metric_compatable(self.metric), (
            f"The {self.metric.__name__} is not compatible with " f"{self.__class__.__name__}"
        )

        self.raw_metric_history_ = (
            metric_history
            if metric_history is not None
            else cast(List[float], self.metric.calculate(X))
        )
        self.metric_history_ = (
            hotload_metric_history
            if hotload_metric_history is not None
            else self._preprocess_metric_history(self.raw_metric_history_)
        )

        self._fit(self.metric_history_, raw_history=X, **kwargs)

        self._is_fitted = True
        return self

    def _fit(self, *args, **kwargs):
        self.u_lower_ = 0.0
        self.u_upper_ = 1.0
        self.expected_fpr_ = 1.0
        return self

    def predict(self, column: pd.Series, **kwargs) -> bool:
        check_is_fitted(self)

        if issubclass(self.metric, metrics.SingleDistributionMetric):
            m = self.metric.calculate(column)
        else:
            m = self.metric.calculate(column, self.last_reference_sample_)

        m = self._preprocess_metric(cast(float, m))
        prediction = self._predict(m, **kwargs)

        return prediction

    def _predict(self, m: float, **kwargs) -> bool:
        return self.u_lower_ <= m <= self.u_upper_

    def _preprocess_metric(self, metric: float) -> float:
        m = self.metric_preprocessing_function(metric)
        if self.time_differencing_window > 0:
            m = m - self.metric_preprocessing_function(
                self.raw_metric_history_[-self.time_differencing_window]
            )

        return m

    def _preprocess_metric_history(self, metric_history: List[float]) -> List[float]:
        preprocessed_metric_history = self.metric_preprocessing_function(metric_history)
        if self.time_differencing_window > 0:
            time_differenced_history = (
                pd.Series(preprocessed_metric_history).diff(self.time_differencing_window).tolist()
            )
            return time_differenced_history[self.time_differencing_window :]
        return preprocessed_metric_history
