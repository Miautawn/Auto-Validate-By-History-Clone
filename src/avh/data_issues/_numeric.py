import string
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from avh.aliases import FloatRange, Seed
from avh.data_issues._base import NumericIssueTransformer


class UnitChange(NumericIssueTransformer):
    def __init__(
        self, p: FloatRange = 1.0, m: int = 2, random_state: Seed = None, randomize: bool = True
    ):
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
    def __init__(
        self, p: FloatRange = 0.5, random_state: Seed = None, n_jobs: Optional[int] = None
    ):
        self.p = p
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _get_prob(self) -> float:
        if isinstance(self.p, Iterable):
            rng = np.random.default_rng(self.random_state)
            return rng.uniform(self.p[0], self.p[1])
        return self.p

    def _perturb_characters(self, x, p, perturbation_indices, perturbation_characters):
        char_array = list(str(x))
        char_array_n = len(char_array)
        perturbation_length = int(char_array_n * p)

        scaled_indices = [
            indice % char_array_n for indice in perturbation_indices[:perturbation_length]
        ]

        if perturbation_length == 0:
            return x

        for perturbation_idx, char_array_idx in enumerate(scaled_indices):
            if char_array[char_array_idx].isdigit():
                char_array[char_array_idx] = perturbation_characters[perturbation_idx]

        return "".join(char_array)

    def _transform_sequential(self, df: pd.DataFrame):
        stringified_df = df.astype("string[pyarrow]")
        rng = np.random.default_rng(self.random_state)
        p = self._get_prob()

        notna_mask = df.notna().to_numpy()
        char_counts = (
            stringified_df.map(len, na_action="ignore")
            .to_numpy()
            .T.reshape(-1, 1)[notna_mask.T.flatten()]
        )
        total_elements = notna_mask.sum()

        max_char_count = char_counts.max()
        max_perturbed_char_count = int(max_char_count * p)

        perturbation_indices = np.tile(range(max_char_count), (total_elements, 1))
        perturbation_indices = rng.permuted(
            perturbation_indices, axis=1, out=perturbation_indices
        )[:, :max_perturbed_char_count]

        perturbation_characters = rng.choice(
            list(string.digits), size=(total_elements, max_perturbed_char_count), replace=True
        )

        scaled_perturbation_indices_iter = iter(perturbation_indices)
        perturbation_characters_iter = iter(perturbation_characters)

        stringified_df = stringified_df.map(
            lambda x: self._perturb_characters(
                x, p, next(scaled_perturbation_indices_iter), next(perturbation_characters_iter)
            ),
            na_action="ignore",
        )

        # After casting the dataframe into 'string[pyarrow]' dtype, the null values become pd.NA
        # however, after the .map() operation the columns become 'object' dtype
        #   still containing pd.NA values.
        # This combination doesn't allow for a clean conversion back into numpy types,
        # thus we replace the pd.NA values into np.nan so the casting would play nice.
        return stringified_df.fillna(np.nan).astype(df.dtypes)

    def _parallel_perturb_column(self, col, p, perturbation_indices, perturbation_characters):

        # By God as my witness, this the fastest running code out of all my tried setups.
        # Apperantly, numpy arrays are easily serialized over processes by joblib,
        #   so that's why input and output numpy arrays.
        # However, numpy arrays are super slow to iterate one by one,
        #   so we have to covert them to regular python list.
        output_col = col.tolist()
        perturbation_indices = perturbation_indices.tolist()
        perturbation_characters = perturbation_characters.tolist()
        for i, x in enumerate(output_col):
            if x == np.nan:
                continue
            output_col[i] = np.float32(
                self._perturb_characters(x, p, perturbation_indices[i], perturbation_characters[i])
            )

        return np.array(output_col)

    def _transform_parallel(self, df: pd.DataFrame):

        rng = np.random.default_rng(self.random_state)
        p = self._get_prob()
        total_elements = df.shape[0]

        # estimating max char count
        max_char_count = 30
        max_perturbed_char_count = int(max_char_count * p)

        perturbation_indices = np.tile(range(max_char_count), (total_elements, 1))
        perturbation_indices = rng.permuted(
            perturbation_indices, axis=1, out=perturbation_indices
        )[:, :max_perturbed_char_count]

        # Pre-generating random character perturbations
        perturbation_characters = rng.choice(
            list(string.digits), size=(total_elements, max_perturbed_char_count), replace=True
        )

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._parallel_perturb_column)(
                df[col].to_numpy(), p, perturbation_indices, perturbation_characters
            )
            for col in df.columns
        )

        return pd.DataFrame(np.array(results).T, columns=df.columns).astype(df.dtypes)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.n_jobs is None:
            return self._transform_sequential(df)
        else:
            return self._transform_parallel(df)
