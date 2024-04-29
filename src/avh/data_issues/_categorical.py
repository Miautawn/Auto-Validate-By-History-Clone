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

from avh.aliases import Seed, FloatRange
from avh.data_issues._base import IssueTransfomer, NumericIssueTransformer, CategoricalIssueTransformer
    
class CasingChange(CategoricalIssueTransformer):
    def __init__(self, p: FloatRange = 0.5, random_state: Seed = None, randomize: bool = True):
        self.p = p
        self.random_state = random_state
        self.randomize = randomize

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
            indexes = rng.choice(range(n), size=sample_n, replace=False)
            new_df.iloc[indexes] = new_df.iloc[indexes].apply(lambda x: x.str.swapcase(), axis=0)
        else:
            new_df.iloc[:sample_n] = new_df.iloc[:sample_n].apply(lambda x: x.str.swapcase(), axis=0)
       
        return new_df