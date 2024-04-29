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

class IssueTransfomer(BaseEstimator, TransformerMixin):
    """
    Should work for every column

    Not necessarally efficiant
    """

    def __repr__(self):
        return f"{self.__class__.__name__}{str(self.get_params())}"

    def _is_dataframe_compatable(self, df: pd.DataFrame) -> bool:
        return True

    def fit(self, df: pd.DataFrame, y=None, **kwargs):
        assert self._is_dataframe_compatable(df), f"{self.__class__.__name__} is not compatable with profided dataframe'"
        return self._fit(df, **kwargs)

    def _fit(self, df: pd.DataFrame, **kwargs):
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, y=None) -> pd.Series:
        check_is_fitted(self)
        return self._transform(df).reset_index(drop=True)

    def _transform(self, df: pd.DataFrame) -> pd.Series:
        return df
    
class NumericIssueTransformer(IssueTransfomer):
    def _is_dataframe_compatable(self, df: pd.DataFrame) -> bool:
        return len(df.select_dtypes(exclude="number").columns) == 0
    
class CategoricalIssueTransformer(IssueTransfomer):
    def _is_dataframe_compatable(self, df: pd.DataFrame) -> bool:
        return len(df.select_dtypes("number").columns) == 0

