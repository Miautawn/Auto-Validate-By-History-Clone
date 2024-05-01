import math
from typing import Tuple, List, Union, Any, Dict, Iterable
import multiprocessing as mp
from itertools import product
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from avh.aliases import Seed, FloatRange, IntRange
from avh.data_issues._base import IssueTransfomer, NumericIssueTransformer, CategoricalIssueTransformer

pd.options.mode.copy_on_write = True

    
class SchemaChange(IssueTransfomer):
    def __init__(self, p: FloatRange = 0.5, random_state: Seed = None, randomize: bool = True):
        self.p = p
        self.random_state = random_state
        self.randomize = randomize

    def _fit(self, df: pd.DataFrame, **kwargs):
        column_index = df.columns

        # Dictionary of {dtype: [column indexes of that type]}
        self.dtype_metadata_ = {
            dtype: column_index.get_indexer_for(
                df.select_dtypes(dtype).columns
            ) for dtype in df.dtypes.unique()
        }

        for dtype, column_indexes in self.dtype_metadata_.items():
            assert (
                len(column_indexes) >= 2
            ), f"Column of dtype {dtype} does not have enough neighboars of the same type"

        return self
    
    def _get_prob(self) -> float:
        if isinstance(self.p, Iterable):
            rng = np.random.default_rng(self.random_state)
            return rng.uniform(self.p[0], self.p[1])
        return self.p

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        new_df = df.copy()

        n = len(new_df)
        sample_n = max(int(n * self._get_prob()), 1)

        if self.randomize:
            rng = np.random.default_rng(self.random_state)
            random_idx = rng.choice(range(n), size=sample_n, replace=False)

        for column_indexes in self.dtype_metadata_.values():
            for idx, column_idx in enumerate(column_indexes):
                next_column_idx = column_indexes[(idx + 1) % len(column_indexes)]

                if self.randomize:
                    new_df.iloc[random_idx, column_idx] = df.iloc[random_idx, next_column_idx]
                else:
                    new_df.iloc[:sample_n, column_idx] = df.iloc[:sample_n, next_column_idx]

        return new_df


class IncreasedNulls(IssueTransfomer):
    def __init__(self, p: FloatRange = 0.5, random_state: Seed = None, randomize: bool = True):
        self.p = p
        self.random_state = random_state
        self.randomize = randomize

    def _get_prob(self) -> float:
        if isinstance(self.p, Iterable):
            rng = np.random.default_rng(self.random_state)
            return rng.uniform(self.p[0], self.p[1])
        return self.p

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()

        n = len(new_df)
        sample_n = max(int(n * self._get_prob()), 1)

        if self.randomize:
            rng = np.random.default_rng(self.random_state)
            indexes = rng.choice(range(n), size=sample_n, replace=False)
            new_df.iloc[indexes] = np.nan
        else:
            new_df.iloc[:sample_n] = np.nan

        return new_df

class VolumeChangeUpsample(IssueTransfomer):
    def __init__(self, f: IntRange = 2, random_state: Seed = None, randomize: bool = True):
        """
        Performs upsampling.
        f >= 1
        """
        self.f = f
        self.random_state = random_state
        self.randomize = randomize

    def _get_factor(self) -> float:
        if isinstance(self.f, Iterable):
            rng = np.random.default_rng(self.random_state)
            return rng.integers(self.f[0], self.f[1])
        return self.f

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        n = len(df)

        factor = self._get_factor()
        sample_n = max(int(n * factor), 1)

        if self.randomize:
            rng = np.random.default_rng(self.random_state)
            indexes = rng.choice(range(len(df)), size=sample_n, replace=True)
            return df.iloc[indexes]
        else:
            indexes = np.tile(range(len(df)), factor)
            return df.iloc[indexes]

class VolumeChangeDownsample(IssueTransfomer):
    def __init__(self, f: FloatRange = 0.5, random_state: Seed = None, randomize: bool = True):
        self.f = f
        self.random_state = random_state
        self.randomize = randomize

    def _get_fraction(self) -> float:
        if isinstance(self.f, Iterable):
            rng = np.random.default_rng(self.random_state)
            return rng.uniform(self.f[0], self.f[1])
        return self.f

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        n = len(df)
        sample_n = max(int(n * self._get_fraction()), 1)

        if self.randomize:
            rng = np.random.default_rng(self.random_state)
            indexes = rng.choice(range(len(df)), size=sample_n, replace=False)
            return df.iloc[indexes]
        else:
            return df.iloc[:sample_n]

class DistributionChange(IssueTransfomer):
    # Doesn't change the row count
    def __init__(self, p: FloatRange = 0.1, take_last: bool = True, random_state: Seed = None,):
        self.p = p
        self.take_last = take_last
        self.random_state = random_state

    def _get_prob(self) -> float:
        if isinstance(self.p, Iterable):
            rng = np.random.default_rng(self.random_state)
            return rng.uniform(self.p[0], self.p[1])
        return self.p
    
    def _process_column(self, column: pd.Series, p: float, take_last: bool):
        not_na_mask = column.notna()

        new_values = column.dropna().sort_values()
        new_sample_size = max(int(len(new_values) * p), 1)

        sample_values = (
            new_values.tail(new_sample_size)
            if take_last
            else new_values.head(new_sample_size)
        )

        repeated_values = np.resize(sample_values, not_na_mask.sum())

        column.iloc[not_na_mask] = repeated_values
        return column

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        new_df = df.copy()

        p = self._get_prob()
        return new_df.apply(lambda col: self._process_column(col, p, self.take_last))
    
from joblib import Parallel, delayed
class DistributionChangeV2(IssueTransfomer):
    # Doesn't change the row count
    def __init__(self, p: FloatRange = 0.1, take_last: bool = True, random_state: Seed = None,):
        self.p = p
        self.take_last = take_last
        self.random_state = random_state

    def _get_prob(self) -> float:
        if isinstance(self.p, Iterable):
            rng = np.random.default_rng(self.random_state)
            return rng.uniform(self.p[0], self.p[1])
        return self.p
    
    def _process_column(self, column: pd.Series, p: float, take_last: bool):
        not_na_mask = column.notna()

        new_values = column.dropna().sort_values()
        new_sample_size = max(int(len(new_values) * p), 1)

        sample_values = (
            new_values.tail(new_sample_size)
            if take_last
            else new_values.head(new_sample_size)
        )

        repeated_values = np.resize(sample_values, not_na_mask.sum())

        column.iloc[not_na_mask] = repeated_values
        return column

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        # new_df = df.copy()

        p = self._get_prob()
        n_jobs = -1  # Using -1 will use all available cores
        results = Parallel(n_jobs=n_jobs)(delayed(self._process_column)(df[col], p, self.take_last) for col in df.columns)

        return pd.concat(results, axis=1)
    

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
class DistributionChangeV3(IssueTransfomer):
    # Doesn't change the row count
    def __init__(self, p: FloatRange = 0.1, take_last: bool = True, random_state: Seed = None,):
        self.p = p
        self.take_last = take_last
        self.random_state = random_state

    def _get_prob(self) -> float:
        if isinstance(self.p, Iterable):
            rng = np.random.default_rng(self.random_state)
            return rng.uniform(self.p[0], self.p[1])
        return self.p
    
    def _process_column(self, args):
        column, p, take_last = args
        not_na_mask = column.notna()

        new_values = column.dropna().sort_values()
        new_sample_size = max(int(len(new_values) * p), 1)

        sample_values = (
            new_values.tail(new_sample_size)
            if take_last
            else new_values.head(new_sample_size)
        )

        repeated_values = np.resize(sample_values, not_na_mask.sum())

        column.loc[not_na_mask] = repeated_values
        return column

    def _transform(self, df: pd.DataFrame) -> pd.Series:

        p = self._get_prob()
        args = ((df[col], p, self.take_last) for col in df.columns)

        with ThreadPoolExecutor(max_workers=None) as executor:  # `None` will use as many workers as there are CPUs
            results = executor.map(self._process_column, args)
    
        return pd.concat(list(results), axis=1)