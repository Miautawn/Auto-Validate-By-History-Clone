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

from avh.aliases import Seed
from avh.data_issues._base import IssueTransfomer, NumericIssueTransformer, CategoricalIssueTransformer

class DQIssueDatasetGenerator():
    """
    Produces D(C) for declared issue transfomers
        and cartesian product of their parameters
    """

    def __init__(self, issues: List[Tuple[IssueTransfomer, dict]], random_state: Seed = None, verbose: int = 1):
        self._random_state = random_state
        self._numeric_issues = []
        self._categorical_issues = []
        self._shared_issues = []
        self.verbose = verbose

        for issue in issues:
            issue_class = issue[0]
            if issubclass(issue_class, NumericIssueTransformer):
                self._numeric_issues.append(issue)
            elif issubclass(issue_class, CategoricalIssueTransformer):
                self._categorical_issues.append(issue)
            else:
                self._shared_issues.append(issue)

    @property
    def verbose(self) -> int:
        if self._verbose == 0:
            return False
        return True

    @verbose.setter
    def verbose(self, level: Union[int, bool]):
        assert level >= 0, "Verbosity level must be a positive integer"
        self._verbose = level

    def generate(self, df: pd.DataFrame):
        dataset = {column: [] for column in df.columns}

        pbar = tqdm(desc="creating D(C)...", disable=not self.verbose)
        for dtype_issues, dtype_columns in self._iterate_issues_by_column_dtypes(df):
            if len(dtype_columns) == 0:
                continue
                
            target_df = df[dtype_columns]
            for transformer, parameters in dtype_issues:
                fitted_transformer = transformer().fit(target_df)
                if "random_state" in fitted_transformer.get_params():
                        fitted_transformer.set_params(random_state=self._random_state)
                
                for param_comb in self._get_parameter_combination(parameters):
                    # Note: generaly you should fit the estimator after setting parameters,
                    #   however, we know that in our case it's safe to do so and allows
                    #   for some minimal optimisatinon by not needing to fit after every param change
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

    # def fit(self, df: pd.DataFrame, y=None, **kwargs):
    #     self.columns_ = list(df.columns)
    #     self.numeric_columns_ = list(df.select_dtypes(include="number").columns)
    #     self.categorical_columns_ = list(set(self.columns_).difference(set(self.numeric_columns_)))
    #     return self

    # def transform(self, df: pd.DataFrame, y=None):
    #     dataset = {column: [] for column in self.columns_}

    #     pbar = tqdm(desc="creating D(C)...")
    #     for dtype_issues, dtype_columns in self._iterate_by_dtype():
    #         if not dtype_columns:
    #             continue
                
    #         target_df = df[dtype_columns]
    #         for transformer, parameters in dtype_issues:
    #             fitted_transformer = transformer().fit(target_df)
                
    #             for param_comb in self._get_parameter_combination(parameters):
    #                 fitted_transformer.set_params(**param_comb)
    #                 fitted_transformer_signature = repr(fitted_transformer)
    #                 modified_df = fitted_transformer.transform(target_df)
                    
    #                 for column in dtype_columns:
    #                     dataset[column].append(
    #                         (fitted_transformer_signature, modified_df[column])
    #                     )
    #                 pbar.update(1)

    #     pbar.close()
    #     return dataset

    def _get_parameter_combination(self, params):
        # Put variable parameter values into iterables,
        #   to be compatable to do carterisan product with Itertools.product()
        corrected_params = {
            k: v if isinstance(v, Iterable) else [v] for k, v in params.items()
        }

        for values in product(*corrected_params.values()):
            yield dict(zip(corrected_params.keys(), values))

    def _iterate_issues_by_column_dtypes(self, df: pd.DataFrame) -> Iterable:
        columns = list(df.columns)
        numeric_columns = self._get_numeric_columns(df)
        categorical_columns = self._get_categorical_columns(df)

        yield (self._shared_issues, columns)
        yield (self._numeric_issues, numeric_columns)
        yield (self._categorical_issues, categorical_columns)

    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        return list(df.select_dtypes("number").columns)

    def _get_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        return list(df.select_dtypes(exclude="number").columns)