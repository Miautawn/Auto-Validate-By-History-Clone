import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, ks_2samp, wasserstein_distance

from avh.metrics._base import NumericMetricMixin, SingleDistributionMetric, TwoDistributionMetric

#### Single distribution metrics


class Min(NumericMetricMixin, SingleDistributionMetric):
    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        if self._is_empty(column):
            return 0.0
        return column.min()


class Max(NumericMetricMixin, SingleDistributionMetric):
    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        if self._is_empty(column):
            return 0.0
        return column.max()


class Mean(NumericMetricMixin, SingleDistributionMetric):
    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        if self._is_empty(column):
            return 0.0
        return column.mean()


class Median(NumericMetricMixin, SingleDistributionMetric):
    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        if self._is_empty(column):
            return 0.0
        return np.nanmedian(column)


class Sum(NumericMetricMixin, SingleDistributionMetric):
    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        if self._is_empty(column):
            return 0.0
        return column.sum()


class Range(NumericMetricMixin, SingleDistributionMetric):
    @classmethod
    def _calculate(self, column: pd.Series) -> float:
        if self._is_empty(column):
            return 0.0
        return column.max() - column.min()


#### Two distribution metrics


class EMD(NumericMetricMixin, TwoDistributionMetric):
    @classmethod
    def _calculate(self, new_sample: pd.Series, old_sample: pd.Series) -> float:
        if self._is_empty(new_sample, old_sample):
            return np.inf

        # Have to drop na, since if there is at least 1 null value,
        #   the scipy.wasserstein_distance() will return null
        return wasserstein_distance(new_sample.dropna(), old_sample.dropna())


class KsDist(NumericMetricMixin, TwoDistributionMetric):
    @classmethod
    def _calculate(self, new_sample: pd.Series, old_sample: pd.Series) -> float:
        if self._is_empty(new_sample, old_sample):
            return np.inf

        _, ks_p_val = ks_2samp(new_sample, old_sample, nan_policy="omit", method="asymp")
        return 1 - ks_p_val


class CohenD(NumericMetricMixin, TwoDistributionMetric):
    @classmethod
    def _calculate(self, new_sample: pd.Series, old_sample: pd.Series) -> float:
        """
        We use null-ignoring operations
        """
        n_new, n_old = new_sample.count(), old_sample.count()

        # Return np.inf if there isn't enough proper data for calculations
        if self._is_empty(new_sample, old_sample) or n_new + n_old <= 2:
            return np.inf

        degrees_of_freedom = n_new + n_old - 2

        mu_new = np.nanmean(new_sample)
        mu_old = np.nanmean(old_sample)

        var_new = np.nanvar(new_sample, ddof=1)
        var_old = np.nanvar(old_sample, ddof=1)

        sp = np.sqrt(((n_new - 1) * var_new + (n_old - 1) * var_old) / degrees_of_freedom)

        return abs((mu_new - mu_old) / (sp + 1e-10))


class KlDivergence(NumericMetricMixin, TwoDistributionMetric):
    @classmethod
    def _calculate(self, new_sample: pd.Series, old_sample: pd.Series) -> float:
        if self._is_empty(new_sample, old_sample):
            return np.inf

        # Don't have to normalize it, as scipy.stats.entropy() will do it for us
        p, e = np.histogram(new_sample.dropna(), 10)
        q, _ = np.histogram(old_sample.dropna(), e)

        # If the new sample is that widely different such that reference sample
        #   doesn't have any values across it's histogram edges,
        #   don't bother computing and just return np.inf
        if sum(q) == 0:
            return np.inf

        # Finding support intersections where there are no 0's
        p, q = zip(*[(x, y) for x, y in zip(p, q) if x != 0 and y != 0])
        return entropy(p, q)


class JsDivergence(NumericMetricMixin, TwoDistributionMetric):
    @classmethod
    def _calculate(self, new_sample: pd.Series, old_sample: pd.Series) -> float:
        if self._is_empty(new_sample, old_sample):
            return np.inf

        # Don't have to normalize it, as scipy.distance.jensenshannon() will do it for us
        p, e = np.histogram(new_sample.dropna(), 10)
        q, _ = np.histogram(old_sample.dropna(), e)

        # If the new sample is that widely different such that reference sample
        #   doesn't have any values across it's histogram edges,
        #   don't bother computing and just return np.inf
        if sum(q) == 0:
            return np.inf

        # Finding support intersections where there are no 0's
        p, q = zip(*[(x, y) for x, y in zip(p, q) if x != 0 and y != 0])
        return jensenshannon(p, q)
