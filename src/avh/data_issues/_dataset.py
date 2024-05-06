from itertools import product
from typing import Dict, Iterable, List, Optional, Tuple, Type, Union

import pandas as pd
from tqdm import tqdm

from avh.aliases import Seed
from avh.data_issues._base import (
    CategoricalIssueTransformer,
    IssueTransfomer,
    NumericIssueTransformer,
)


class DQIssueDatasetGenerator:
    """
    Produces D(C) for declared issue transfomers
        and cartesian product of their parameters
    """

    def __init__(
        self,
        issues: List[Tuple[Type[IssueTransfomer], dict]],
        random_state: Seed = None,
        verbose: int = 1,
        n_jobs: Optional[int] = None,
    ):
        self._random_state = random_state
        self._numeric_issues = []
        self._categorical_issues = []
        self._shared_issues = []
        self._n_jobs = n_jobs
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
        dataset: Dict[str, list] = {column: [] for column in df.columns}

        pbar = tqdm(desc="creating D(C)...", disable=not self.verbose)
        for dtype_issues, dtype_columns in self._iterate_issues_by_column_dtypes(df):
            if len(dtype_columns) == 0:
                continue

            target_df = df[dtype_columns]
            for transformer, parameters in dtype_issues:
                fitted_transformer = transformer().fit(target_df)
                fitted_transformer = self._set_optional_transformer_parameters(fitted_transformer)

                for param_comb in self._get_parameter_combination(parameters):
                    # Note: generaly you should fit the estimator after setting parameters,
                    #   however, we know that in our case it's safe to do so and allows
                    #   for a small optimisatinon by not needing to fit after every param change
                    fitted_transformer.set_params(**param_comb)
                    fitted_transformer_signature = repr(fitted_transformer)
                    modified_df = fitted_transformer.transform(target_df)

                    for column in dtype_columns:
                        dataset[column].append((fitted_transformer_signature, modified_df[column]))
                    pbar.update(1)

        pbar.close()
        return dataset

    def _get_parameter_combination(self, params):
        # Put variable parameter values into iterables,
        #   to be compatable to do carterisan product with Itertools.product()
        corrected_params = {k: v if isinstance(v, Iterable) else [v] for k, v in params.items()}

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

    def _set_optional_transformer_parameters(
        self, transformer: IssueTransfomer
    ) -> IssueTransfomer:
        transformer_params = transformer.get_params()
        if "random_state" in transformer_params:
            transformer.set_params(random_state=self._random_state)
        if "n_jobs" in transformer_params:
            transformer.set_params(n_jobs=self._n_jobs)
        return transformer
