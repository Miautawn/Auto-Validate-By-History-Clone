from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from avh.aliases import Seed
from avh.data_generation._base import DataColumn
from avh.data_issues import IssueTransfomer


class DataGenerationPipeline:
    """
    Combine multiple DataColumn and IssueTransfomer instances
    to produce a pipeline output.

    The applying of DQ issues follow these rules:
        * The issues are applyed in the order they are defined in the parameter
        * Each issue transformer is fed the original output of the columns
            and not the modified versions of them by the previous transformers.
            This is done, to make the effects of issues more independant.
        * The issues are expected to be defined fully for each column.
            That means repeated definitions for the same column will be overriden.
            An exception to this is when 'all' columns are used, since after each
            issue used to modify 'all' column, the output dataframe replaces the
            original reference dataframe. This is done so you could use issues
            for 'all' and specific columns together, without overriding each other.

    Parameters
    ----------
    columns: List[DataColumn]
        A list of DataColumn instances which define the columns of final dataframe.
    issues: List[Tuple[str, List[IssueTransfomer]]]
        A list of tuples defining the issues to inject into a column.
        Each tuple contains:
            * column: str
                The column name to which apply the following data issues
                Use 'all' to inject the issues into all columns.
            * issue_transformers: List[IssueTransfomer]
                A list of IssueTransfomer to apply to the column above

    I recommend this blogpost about how Numpy rng best practices:
    https://albertcthomas.github.io/good-practices-random-number-generators/
    """

    def __init__(
        self,
        columns: List[DataColumn],
        # TODO: add an aliased type field for this monstrosity
        issues: Optional[List[Tuple[str, List[IssueTransfomer]]]] = None,
        random_state: Seed = None,
    ):
        self._columns = columns
        self._issues = [] if issues is None else issues
        self.random_state = random_state  # type: ignore
        # self.iteration = 0

    @property
    def random_state(self) -> np.random.Generator:
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: Seed):
        self._random_state = np.random.default_rng(random_state)
        self._seed_dependencies()

    def _seed_dependencies(self):
        self._seed_column_generators()
        self._seed_issues()

    def _seed_column_generators(self):
        for column in self._columns:
            column.random_state = self.random_state

    def _seed_issues(self):
        for column_issues in self._issues:
            column_name, issues = column_issues
            for issue in issues:
                if "random_state" in issue.get_params():
                    issue.set_params(random_state=self.random_state)

    def generate(self, n: int) -> pd.DataFrame:
        """
        Combines the outputs of each specified DataColumn into a dataframe
            of length n
        """

        # TODO: enable this once we enable the time differencing
        # data = pd.concat(
        #     [column.generate(n, self.iteration) for column in self._columns], axis=1
        # )
        # self.iteration += 1

        data = pd.concat([column.generate(n) for column in self._columns], axis=1)
        data = self._apply_issues(data)
        return data

    def generate_uniform(self, lower: int, higher: int) -> pd.DataFrame:
        """
        Combines the outputs of each specified DataColumn into a dataframe
            of variable length, randomly picked from uniform PDF
        """
        rng = self.random_state
        return self.generate(max(1, rng.integers(lower, higher)))

    def generate_normal(self, mean: int, std: int) -> pd.DataFrame:
        """
        Combines the outputs of each specified DataColumn into a dataframe
            of variable length, randomly picked from normal PDF
        """
        rng = self.random_state
        return self.generate(max(1, int(rng.normal(mean, std))))

    def _apply_issues(self, data: pd.DataFrame) -> pd.DataFrame:
        if self._issues is None:
            return data

        for col, issues in self._issues:
            for issue in issues:
                if col == "all":
                    data = issue.fit_transform(data)
                else:
                    column_dtype = data[col].dtype
                    dtype_columns = data.select_dtypes(column_dtype).columns
                    transformed_column = issue.fit_transform(data[dtype_columns])[col]
                    data = pd.concat([data.drop(col, axis=1), transformed_column], axis=1)

        return data
