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
    
class UnitChange(NumericIssueTransformer):
    def __init__(self, p: FloatRange = 1.0, m: int = 2, random_state: Seed = None, randomize: bool = True):
        self.p = p
        self.m = m
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
            new_df.iloc[indexes] *= self.m
        else:
            new_df.iloc[:sample_n] *= self.m

        return new_df
    
    
class NumericPerturbation(NumericIssueTransformer):
    def __init__(self, p: FloatRange = 0.5, random_state: Seed = None):
        self.p = p
        self.random_state = random_state

    def _get_prob(self) -> float:
        if isinstance(self.p, Iterable):
            rng = np.random.default_rng(self.random_state)
            return rng.uniform(self.p[0], self.p[1])
        return self.p

    def _perturb_characters(self, x: str, p: float, perturbation_indices: np.array, perturbations: np.array):
        char_array = list(x)
        perturbation_length = int(len(char_array) * p)

        if perturbation_length == 0:
            return x

        # Deduplicating scaled perturbation indices to increase randomness.
        # We use dict() instead of set(), since dict save the order of insertion,
        #   thus keeping the shuffling from the generation step.
        deduplicated_perturbation_indinces = list(dict.fromkeys(perturbation_indices))[
            : perturbation_length
        ]

        for perturbation_idx, char_array_idx in enumerate(deduplicated_perturbation_indinces):
            if char_array[char_array_idx].isdigit():
                char_array[char_array_idx] = perturbations[perturbation_idx]

        return "".join(char_array)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        stringified_df = df.astype("string[pyarrow]")
        rng = np.random.default_rng(self.random_state)
        p = self._get_prob()

        notna_mask = df.notna().to_numpy()
        char_counts = stringified_df.map(len, na_action="ignore").to_numpy().T.reshape(-1, 1)[notna_mask.T.flatten()]
        total_elements = notna_mask.sum()
        
        max_char_count = char_counts.max()
        max_perturbed_char_count = int(max_char_count * p)

        # Pre-generating random perturbation indices
        #   Since we have strings of different lengths, we use linear sclaing
        #   to scale them down for each element.
        perturbation_indices = np.tile(range(max_char_count), (total_elements, 1))
        perturbation_indices = rng.permuted(perturbation_indices, axis=1, out=perturbation_indices)[:, :max_perturbed_char_count]

        scaled_perturbation_indices = perturbation_indices * char_counts // max_char_count
        
        # Pre-generating random character perturbations
        perturbation_characters = rng.choice(
            10, size=(total_elements, max_perturbed_char_count), replace=True
        ).astype(str)

        # Creating iterators to feed for .map()
        scaled_perturbation_indices_iter = iter(scaled_perturbation_indices)
        perturbation_characters_iter = iter(perturbation_characters)

        stringified_df = stringified_df.map(
            lambda x: self._perturb_characters(
                x, p, next(scaled_perturbation_indices_iter), next(perturbation_characters_iter)
            ), na_action="ignore"
        )

        # After casting the dataframe into 'string[pyarrow]' dtype, the null values become pd.NA
        # however, after the .map() operation the columns become 'object' dtype
        #   still containing pd.NA values.
        # This combination doesn't allow for a clean conversion back into numpy types,
        # thus we replace the pd.NA values into np.nan so the casting would play nice.
        return stringified_df.fillna(np.nan).astype(df.dtypes)