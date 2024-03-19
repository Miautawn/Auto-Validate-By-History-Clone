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
        column_history,
        y=None,
        # hotload_history: bool = False,
        # preprocessed_metric_history: np.array = None,
        **kwargs,
    ) -> None:
        assert self.is_metric_compatable(self.metric), (
            f"The {self.metric.__name__} is not compatible with "
            f"{self.__class__.__name__}"
        )

        self._fit(column_history, **kwargs)

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

    def predict(self, column: pd.Series, **kwargs) -> bool:
        check_is_fitted(self)

        prediction = self._predict(self.metric.calculate(column), **kwargs)

        # metric_raw = self.metric.calculate(column)
        # metric_post = self._preprocess(metric_raw)

        # prediction = self._predict(metric_post, **kwargs)
        # self.metric_history_raw_.append(metric_raw)
        # self.metric_history_post_.append(metric_post)
        return prediction

    def _fit(self, metric_history: List[float], **kwargs):
        self.u_lower_ = 0.0
        self.u_upper_ = 1.0
        self.expected_fpr_ = 1.0
        return self

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


class ChebyshevConstraint(Constraint):
    """
    Chebyshev!
    """

    def _fit(self, metric_history: List[float], beta: float):
        mean = np.nanmean(metric_history)
        var = np.nanvar(metric_history)

        self.u_upper_ = mean + beta
        self.u_lower_ = mean - beta

        if var == 0:
            self.expected_fpr_ = 0.0
        else:
            self.expected_fpr_ = var / (beta**2)


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

    def _fit(self, metric_history: List[float], beta: float):
        mean = np.nanmean(metric_history)
        std = np.nanstd(metric_history)

        self.u_upper_ = mean + beta
        self.u_lower_ = mean - beta

        if std == 0:
            self.expected_fpr_ = 0.0
        else:
            satisfaction_p = (2 / np.sqrt(math.pi)) * (
                integrate.quad(self._bell_function, 0, beta / (np.sqrt(2) * std))[0]
            )
            self.expected_fpr_ = 1 - satisfaction_p


class ConjuctivDQProgram:
    def __init__(
        self,
        constraints: Optional[List[Constraint]] = None,
        recall: Optional[Set[str]] = None,
        contributions: Optional[List[Set[str]]] = None,
    ):
        self.constraints = constraints if constraints else []
        self.recall = recall if recall else set({})
        self.contributions = contributions if contributions else []

    def __repr__(self):
        return "{constraints}, FPR = {fpr:4f}".format(
            constraints="\n".join([repr(q) for q in self.constraints]),
            fpr=self.expected_fpr,
        )

    @property
    def expected_fpr(self):
        return sum([q.expected_fpr_ for q in self.constraints])

    # def fit()

    def predict(self, column: pd.Series, **kwargs):
        for constraint in self.constraints:
            if not constraint.predict(column, **kwargs):
                return False
        return True