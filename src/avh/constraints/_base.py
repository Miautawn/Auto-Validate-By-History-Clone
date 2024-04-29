import math
from typing import List, Union, Callable, Optional, Set

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, check_is_fitted
from scipy import integrate

import avh.utility_functions as utils
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

    # TODO: find a better way to do this without hardcoding solid class names
    compatable_metrics = (metrics.Metric,)

    def __init__(
        self,
        metric: metrics.Metric,
        # differencing_lag: int = 0,
        # preprocessing_func: Callable = utils.identity,
    ):
        self.metric = metric
        # self.differencing_lag = differencing_lag
        # self.preprocessing_func = preprocessing_func

        self.differencing_lag = 0
        self.preprocessing_func = utils.identity

    @classmethod
    def is_metric_compatable(self, metric: metrics.Metric):
        return issubclass(metric, self.compatable_metrics)

    def _get_metric_repr(self):
        metric_repr = self.metric.__name__
        preprocessng_func_repr = self.preprocessing_func.__function_repr__
        if preprocessng_func_repr != "identity":
            metric_repr = "{}({})".format(preprocessng_func_repr, metric_repr)
        if self.differencing_lag != 0:
            metric_repr = "{}.diff({})".format(metric_repr, self.differencing_lag)
        return metric_repr

    def __repr__(self):
        metric_repr = self._get_metric_repr()
        return "{name}({u_lower:0.4f} <= {metric} <= {u_upper:0.4f}, FPR = {fpr:0.4f})".format(
            name=self.__class__.__name__,
            u_lower=self.u_lower_,
            metric=metric_repr,
            u_upper=self.u_upper_,
            fpr=self.expected_fpr_,
        )

    def fit(
        self,
        history: List[pd.Series],
        y=None,
        hotload_history: Optional[List[float]] = None,
        # preprocessed_metric_history: np.array = None,
        **kwargs,
    ) -> None:
        assert self.is_metric_compatable(self.metric), (
            f"The {self.metric.__name__} is not compatible with "
            f"{self.__class__.__name__}"
        )

        self.metric_history_ = hotload_history if hotload_history is not None else self.metric.calculate(history)
        self._fit(self.metric_history_, **kwargs)

        # self.metric_history_raw_ = (
        #     column_history if hotload_history else self.metric.calculate(column_history)
        # )
        # self.metric_history_post_ = (
        #     preprocessed_metric_history
        #     if preprocessed_metric_history
        #     else self._preprocess(self.metric_history_raw_, inference=False)
        # )

        # self._fit(self.metric_history_post_, **kwargs)
        return self
    
    def _fit(self, metric_history: List[float], **kwargs):
        self.u_lower_ = 0.0
        self.u_upper_ = 1.0
        self.expected_fpr_ = 1.0
        return self

    def predict(self, column: pd.Series, **kwargs) -> bool:
        check_is_fitted(self)

        prediction = self._predict(self.metric.calculate(column), **kwargs)

        # metric_raw = self.metric.calculate(column)
        # metric_post = self._preprocess(metric_raw)

        # prediction = self._predict(metric_post, **kwargs)
        # self.metric_history_raw_.append(metric_raw)
        # self.metric_history_post_.append(metric_post)
        return prediction

    def _predict(self, m: float, **kwargs) -> bool:
        return self.u_lower_ <= m <= self.u_upper_

    # def _preprocess(
    #     self, data: Union[List[float], float], inference=True
    # ) -> Union[List[float], float]:
    #     data = self.preprocessing_func(data)

    #     if self.differencing_lag != 0:
    #         if inference:
    #             data = data - self.preprocessing_func(
    #                 self.metric_history_raw_[-self.differencing_lag]
    #             )
    #         else:
    #             data = diff(data, self.differencing_lag)

    #     return data