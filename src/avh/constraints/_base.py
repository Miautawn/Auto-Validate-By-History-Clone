from typing import List, Optional, Tuple, cast

import pandas as pd
from sklearn.base import BaseEstimator, check_is_fitted

import avh.metrics as metrics


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
    ):
        self.metric = metric

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
        return metric_repr

    def fit(
        self,
        X: List[pd.Series],
        y=None,
        hotload_history: Optional[List[float]] = None,
        **kwargs,
    ):

        assert self.is_metric_compatable(self.metric), (
            f"The {self.metric.__name__} is not compatible with " f"{self.__class__.__name__}"
        )

        self.metric_history_ = cast(
            List[float],
            hotload_history if hotload_history is not None else self.metric.calculate(X),
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

        m = cast(float, self.metric.calculate(column))
        prediction = self._predict(m, **kwargs)

        return prediction

    def _predict(self, m: float, **kwargs) -> bool:
        return self.u_lower_ <= m <= self.u_upper_
