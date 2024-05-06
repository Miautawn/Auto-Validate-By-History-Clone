from typing import Type, TypeAlias

from avh.data_issues._base import (
    CategoricalIssueTransformer,
    IssueTransfomer,
    NumericIssueTransformer,
)
from avh.data_issues._categorical import CasingChange
from avh.data_issues._dataset import DQIssueDatasetGenerator
from avh.data_issues._issues import (
    DistributionChange,
    IncreasedNulls,
    SchemaChange,
    VolumeChangeDownsample,
    VolumeChangeUpsample,
)
from avh.data_issues._numeric import NumericPerturbation, UnitChange

"""
Provides the 'issue transformers', which can be used to simulate data quality issues
    for the AVH algorithm.

Note: not all issue transformers are identical in their functionality to the original paper's:
    https://github.com/microsoft/Auto-Validate-by-History/blob/main/gene_sample.py

    Here are the main differences:
        * IncreasedNulls - In the author's code, a bunch of nulls are appended
            to the original column. In our implementation we replace values with nulls instead.
            This way we try to isolate the effect of null increase, rather than a combination of
            null and row count increase.
        * DistributionChange - In the author's code, only the last/first p% of rows are taken
            to simulate the distribution shift.
            In our implementation we tile the last/first p% of values across all rows.
            Additionally, we drop null values before sorting as to avoid taking all-null slice
                and preserve them from the original df after the operation.
            This way we try to isolate the effect of distribution shift.
"""

IssueType: TypeAlias = Type[IssueTransfomer]

__all__ = [
    "IssueTransfomer",
    "IssueType",
    "SchemaChange",
    "IncreasedNulls",
    "VolumeChangeUpsample",
    "VolumeChangeDownsample",
    "DistributionChange",
    "NumericIssueTransformer",
    "UnitChange",
    "NumericPerturbation",
    "CategoricalIssueTransformer",
    "CasingChange",
    "DQIssueDatasetGenerator",
]
