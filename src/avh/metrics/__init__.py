from typing import Type, TypeAlias

from avh.metrics._base import (
    CategoricalMetricMixin,
    Metric,
    NumericMetricMixin,
    SingleDistributionMetric,
    TwoDistributionMetric,
)
from avh.metrics._categorical import (
    DistinctCount,
    MeanDigitLength,
    MeanPunctuationLength,
    MeanStringLength,
)
from avh.metrics._metrics import CompleteRatio, DistinctRatio, RowCount
from avh.metrics._numeric import (
    EMD,
    CohenD,
    JsDivergence,
    KlDivergence,
    KsDist,
    Max,
    Mean,
    Median,
    Min,
    Range,
    Sum,
)

MetricType: TypeAlias = Type[Metric]

__all__ = [
    "Metric",
    "MetricType",
    "SingleDistributionMetric",
    "TwoDistributionMetric",
    "NumericMetricMixin",
    "CategoricalMetricMixin",
    "RowCount",
    "DistinctRatio",
    "CompleteRatio",
    "EMD",
    "KsDist",
    "CohenD",
    "KlDivergence",
    "JsDivergence",
    "Min",
    "Max",
    "Mean",
    "Median",
    "Sum",
    "Range",
    "DistinctCount",
    "MeanStringLength",
    "MeanDigitLength",
    "MeanPunctuationLength",
]
