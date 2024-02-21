from typing import List, Tuple, Optional, Any, Union, Callable
from abc import ABC, abstractmethod, abstractproperty

import pandas as pd
import numpy as np
from gutenbergpy import textget

from avh.data_quality_issues import IssueTransfomer

# predownloading Moby-dick text for random text generation
raw_book = textget.get_text_by_id(2701)
raw_book = str(textget.strip_headers(raw_book)).replace("\\n", "")

class DataColumn(ABC):
    """
    Abstract column class.
    Blueprint for generating data of specified type & behavior.

    Parameters
    ----------
    name: str
        The name of the column in the final output

    Attributes
    ----------
    name: str
        The name of the column in the final output
    dtype: Any
        The dtype of the column in the final output
    """

    def __init__(self, name: str):
        """
        If subclassed, the child should call the parent constructor
        """
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractproperty
    def dtype(self) -> Any:
        ...

    @abstractmethod
    def generate(self, n: int, i: int = 0) -> pd.Series:
        """
        Output the generated Series of length n

        Parameters
        ----------
        n: int
            The length of the column to output
        i: int
            The i'th call of the column generation.
            Can be useful for modifying the genration parameters based on "time"
        """
        ...


class NumericColumn(DataColumn):
    """
    Abstract numeric column class.
    Blueprint for generating data for specifically numeric columns.

    Parameters
    ----------
    name: str
        The name of the column in the final output
    minimum: Optional[float]
        The minimum value this column should output
    maximum: Optional[float]
        The maximum value this column should ouptput
    dtype: Union[np.float32, np.int32]
        The dtype of this numeric column.
        Currently only accepts np.float32 or np.int32
    parameter_function: Optional[Callable]
        A function which accepts and returns the data generation parameters.
        Can be used to create "moving" columns, where column parameters are
            changed each call. If none, this function won't be applied
        The function must have the following definition:
            ```
            def func(n, i, *args):
                return *args
            ```
            Where:
            - 'n' is the column size during this generation call
            - 'i' is the iteration count (default is 0)
            - '*args' are the generaton parameters
    """

    def __init__(
        self,
        name: str,
        minimum: Optional[float] = -np.inf,
        maximum: Optional[float] = np.inf,
        dtype: Union[np.float32, np.int32] = np.float32,
        parameter_function: Optional[Callable] = None,
    ):
        """
        If subclassed, the child should call the parent constructor
        """
        super().__init__(name)
        assert (
            dtype == np.float32 or dtype == np.int32
        ), "Numeric column can only be of type np.float32 or np.int32"

        self._minimum = minimum
        self._maximum = maximum
        self._dtype = dtype
        self._parameter_function = parameter_function

    @property
    def dtype(self):
        return self._dtype

    @abstractmethod
    def _update_parameters(self, n: int, i: int) -> None:
        """
        Template method for applying the `parameter_function`
            to data generation parameters.

        Parameters
        ----------
        n: int
            The length of the column
        i: int
            The i'th call of the column generation.
        """
        ...

    @abstractmethod
    def _generate(self, n: int) -> np.array:
        """
        Template method for generating data.
        """
        ...

    def generate(self, n: int, i: int = 0) -> pd.Series:
        data = self._generate(n).astype(self.dtype)
        data = np.clip(data, self._minimum, self._maximum)

        if self._parameter_function:
            self._update_parameters(n, i)

        return pd.Series(
            data,
            name=self.name,
            dtype=self.dtype,
        )


class UniformNumericColumn(NumericColumn):
    """
    Concrete numeric column class.
    Generates data by using uniform PDF.

    Parameters
    ----------
    name: str
        The name of the column in the final output
    lower_bound: float
        Lower bound for unifrom PDF
    upper_bound: float
        Upper bound for uniform PDF
    **kwargs:
        Any other parameters will be forwarded back to parent classes
    """

    def __init__(self, name: str, lower_bound: float, upper_bound: float, **kwargs):
        super().__init__(name, **kwargs)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def _generate(self, n: int) -> np.array:
        return np.random.uniform(self._lower_bound, self._upper_bound, n)

    def _update_parameters(self, n: int, i: int):
        self._lower_bound, self._upper_bound = self._parameter_function(
            n, i, self._lower_bound, self._upper_bound
        )


class NormalNumericColumn(NumericColumn):
    """
    Concrete numeric column class.
    Generates data by using normal PDF.

    Parameters
    ----------
    name: str
        The name of the column in the final output
    mean: float
        mean value used for normal PDF
    std: float
        standard deviation used for normal PDF
    **kwargs:
        Any other parameters will be forwarded back to parent classes
    """

    def __init__(self, name: str, mean: float, std: float, **kwargs):
        super().__init__(name, **kwargs)
        self._mean = mean
        self._std = std

    def _generate(self, n: int) -> np.array:
        return np.random.normal(self._mean, self._std, n)

    def _update_parameters(self, n: int, i: int):
        self._mean, self._std = self._parameter_function(n, i, self._mean, self._std)


class CategoricalColumn(DataColumn):
    """
    Abstract categorical/string column class.
    Blueprint for generating data for specifically string based columns.

    Note: this class uses the dtype of "object", since pandas Categorical dtype
        is specifically designed to protect against DQ issues, thus isn't flexible
        for our use case.

    Parameters
    ----------
    name: str
        The name of the column in the final output
    """

    def __init__(self, name: str):
        super().__init__(name)

    @property
    def dtype(self):
        return "object"

    @abstractmethod
    def _generate(self, n: int) -> np.array:
        """
        Template method for generating data.
        """
        ...

    def generate(self, n: int, *args, **kwargs) -> pd.Series:
        data = self._generate(n)
        return pd.Series(
            data,
            name=self.name,
            dtype=self.dtype,
        )


class StaticCategoricalColumn(CategoricalColumn):
    """
    Concrete categorical column class.
    Outputs the column populated by provided values.

    Parameters
    ----------
    name: str
        The name of the column in the final output
    values: List[str]
        A list of values which the column will output.
        The list of values must be equal in length to requested column size.
    **kwargs:
        Any other parameters will be forwarded back to parent classes
    """

    def __init__(self, name: str, values: List[str], **kwargs):
        super().__init__(name, **kwargs)
        self._values = values

    def _generate(self, n: int) -> np.array:
        assert n == len(self._values), (
            f"The StaticCategoricalColumn does not have equal number of values "
            f"to fill a column of size {n}"
        )
        return np.array(self._values)


class RandomCategoricalColumn(CategoricalColumn):
    """
    Concrete categorical column class.
    Outputs the column randomly populated by a pool of values.

    Parameters
    ----------
    name: str
        The name of the column in the final output
    values: Optional[List[str]]
        A list of values which will be used to randomly populate the column.
        If None, the class will output random lorem sentences.
    **kwargs:
        Any other parameters will be forwarded back to parent classes
    """

    def __init__(self, name: str, values: Optional[List[str]] = None, **kwargs):
        super().__init__(name, **kwargs)
        self._values = values

    def _generate(self, n: int) -> np.array:
        if self._values:
            return np.random.choice(self._values, n)
        else:
            random_idx = np.random.randint(0, len(raw_book) - 20, n)
            return np.array(
                [raw_book[random_idx[i] : random_idx[i] + 20] for i in range(n)]
            )


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
    """

    def __init__(
        self,
        columns: List[DataColumn],
        issues: List[Tuple[str, List[IssueTransfomer]]] = [],
    ):
        self._columns = columns
        self.issues = issues
        self.iteration = 1

    def generate(self, n: int) -> pd.DataFrame():
        """
        Combines the outputs of each specified DataColumn into a dataframe
            of length n
        """
        data = pd.concat(
            [column.generate(n, self.iteration) for column in self._columns], axis=1
        )
        data = self._apply_issues(data)
        self.iteration += 1
        return data

    def generate_uniform(self, lower: int, higher: int) -> pd.DataFrame():
        """
        Combines the outputs of each specified DataColumn into a dataframe
            of variable length, randomly picked from uniform PDF
        """
        return self.generate(max(1, np.random.randint(lower, higher)))

    def generate_normal(self, mean: int, std: int) -> pd.DataFrame():
        """
        Combines the outputs of each specified DataColumn into a dataframe
            of variable length, randomly picked from normal PDF
        """
        return self.generate(max(1, int(np.random.normal(mean, std))))

    def _apply_issues(self, data: pd.DataFrame) -> pd.DataFrame:
        for col, issues in self.issues:
            for issue in issues:
                if col == "all":
                    data = issue.fit_transform(data)
                else:
                    column_dtype = data[col].dtype
                    dtype_columns = data.select_dtypes(column_dtype).columns
                    transformed_column = issue.fit_transform(data[dtype_columns])[col]
                    data = pd.concat(
                        [data.drop(col, axis=1), transformed_column], axis=1
                    )

        return data