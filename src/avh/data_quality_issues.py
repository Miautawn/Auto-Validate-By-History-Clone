import math
from typing import Tuple, List, Union, Any, Dict
import multiprocessing as mp
from itertools import product
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

class IssueTransfomer(BaseEstimator, TransformerMixin):
    """
    Should work for every column

    Not necessarally efficiant
    """

    def __repr__(self):
        return f"{self.__class__.__name__}{str(self.get_params())}"

    def fit(self, df: pd.DataFrame, y=None, **kwargs):
        assert self._is_dataframe_compatable(df), f"{self.__class__.__name__} is not compatable with profided dataframe'"
        return self._fit(df, **kwargs)

    def _fit(self, df: pd.DataFrame, **kwargs):
        return self

    def transform(self, df: pd.DataFrame, y=None) -> pd.Series:
        #check_is_fitted(self)
        new_df = self._transform(df)
        return new_df.reset_index(drop=True)

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        return df

    def _is_dataframe_compatable(self, df: pd.DataFrame) -> bool:
        return True

class NumericIssueTransformer(IssueTransfomer):
    def _is_dataframe_compatable(self, df: pd.DataFrame) -> bool:
        return len(self.numeric_columns_) > 0

    def fit(self, df: pd.DataFrame, y=None, **kwargs):
        self.numeric_columns_ = df.select_dtypes("number").columns
        assert self._is_dataframe_compatable(df), f"{self.__class__.__name__} is not compatable with profided dataframe"
        return self._fit(df, **kwargs)


class CategoricalIssueTransformer(IssueTransfomer):
    def _is_dataframe_compatable(self, df: pd.DataFrame) -> bool:
        return len(self.categorical_columns_) > 0

    def fit(self, df: pd.DataFrame, y=None, **kwargs):
        self.categorical_columns_ = df.select_dtypes(exclude="number").columns
        assert self._is_dataframe_compatable(df), f"{self.__class__.__name__} is not compatable with profided dataframe"
        return self._fit(df, **kwargs)


class SchemaChange(IssueTransfomer):
    def __init__(self, p: float=0.5):
        self.p = p

    def _fit(self, df: pd.DataFrame, **kwargs):
        self.dtype_metadata_ = {
            dtype: df.select_dtypes(dtype).columns for dtype in df.dtypes.unique()
        }

        for dtype, columns in self.dtype_metadata_.items():
            assert (
                len(columns) > 1
            ), f"Column of dtype {dtype} does not have enough neighboars of the same type"

        return self

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        new_df = df.copy()

        n = new_df.shape[0]
        sample_n = max(int(n * self.p), 1)
        indexes = np.random.choice(df.index, size=sample_n, replace=False)
        for dtype_columns in self.dtype_metadata_.values():
            for idx, column in enumerate(dtype_columns):
                next_column_name = dtype_columns[(idx + 1) % len(dtype_columns)]
                new_df.loc[indexes, column] = df.loc[indexes, next_column_name]

        return new_df


class IncreasedNulls(IssueTransfomer):
    def __init__(self, p: float = 0.5):
        self.p = p

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        new_df = df.copy()

        n = new_df.shape[0]
        sample_n = max(int(n * self.p), 1)
        indexes = np.random.choice(df.index, size=sample_n, replace=False)
        new_df.loc[indexes, :] = np.nan
        return new_df


class VolumeChange(IssueTransfomer):
    def __init__(self, f: float = 2):
        """
        Performs random upsampling, downsampling
        If factor is > 1, then it's treated as a multiplyer for upsampling data volume
        If factor is < 1, then it's treated as a % of data to keep when downsampling
        """
        self.f = f

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        n = df.shape[0]
        sample_n = max(int(n * self.f), 1)
        indexes = np.random.choice(
            df.index, sample_n, replace=True if self.f > 1 else False
        )

        return df.loc[indexes]


class DistributionChange(IssueTransfomer):
    def __init__(self, p: float = 0.1, take_last: bool = True):
        self.p = p
        self.take_last = take_last

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        new_df = df.apply(lambda x: x.sort_values().values, axis=0)

        n = df.shape[0]
        sample_n = max(int(n * self.p), 1)
        sample_tile_count = math.ceil(n / sample_n)

        sample_idx = (
            new_df.index[-sample_n:] if self.take_last else new_df.index[:sample_n]
        )
        sample_idx = np.tile(sample_idx, sample_tile_count)[:n]
        return new_df.loc[sample_idx]


class UnitChange(NumericIssueTransformer):
    def __init__(self, m: int = 2):
        self.m = m

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        new_df = df.copy()
        new_df.loc[:, self.numeric_columns_] *= self.m
        return new_df


class CasingChange(CategoricalIssueTransformer):
    def __init__(self, p: float = 0.5):
        self.p = p

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        new_df = df.copy()

        n = new_df.shape[0]
        n_samples = max(int(n * self.p), 1)
        indexes = np.random.choice(new_df.index, size=n_samples, replace=False)

        new_df.loc[indexes, self.categorical_columns_] = new_df.loc[
            indexes, self.categorical_columns_
        ].apply(lambda x: x.str.swapcase(), axis=0)
        return new_df

class DQIssueDatasetTransformer(BaseEstimator, TransformerMixin):
    """
    Produces D(C) for declared issue transfomers
        and cartesian product of their parameters
    """

    def __init__(self, *issues):
        self.numeric_issues = []
        self.categorical_issues = []
        self.shared_issues = []
        for issue in issues:
            issue_class = issue[0]
            if issubclass(issue_class, NumericIssueTransformer):
                self.numeric_issues.append(issue)
            elif issubclass(issue_class, CategoricalIssueTransformer):
                self.categorical_issues.append(issue)
            else:
                self.shared_issues.append(issue)

    def fit(self, df: pd.DataFrame, y=None, **kwargs):
        self.columns_ = list(df.columns)
        self.numeric_columns_ = list(df.select_dtypes(include="number").columns)
        self.categorical_columns_ = list(set(self.columns_).difference(set(self.numeric_columns_)))
        return self

    def transform(self, df: pd.DataFrame, y=None):
        dataset = {column: [] for column in self.columns_}

        pbar = tqdm(desc="creating D(C)...")
        for dtype_issues, dtype_columns in self._iterate_by_dtype():
            if not dtype_columns:
                continue
                
            target_df = df[dtype_columns]
            for transformer, parameters in dtype_issues:
                fitted_transformer = transformer().fit(target_df)
                
                for param_comb in self._get_parameter_combination(parameters):
                    fitted_transformer.set_params(**param_comb)
                    fitted_transformer_signature = repr(fitted_transformer)
                    modified_df = fitted_transformer.transform(target_df)
                    
                    for column in dtype_columns:
                        dataset[column].append(
                            (fitted_transformer_signature, modified_df[column])
                        )
                    pbar.update(1)

        pbar.close()
        return dataset

    def _get_parameter_combination(self, params):
        for values in product(*params.values()):
            yield dict(zip(params.keys(), values))

    def _iterate_by_dtype(self):
        yield (self.shared_issues, self.columns_)
        yield (self.numeric_issues, self.numeric_columns_)
        yield (self.categorical_issues, self.categorical_columns_)
