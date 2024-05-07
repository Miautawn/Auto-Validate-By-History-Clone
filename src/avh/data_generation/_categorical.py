from typing import List

import numpy as np

from avh.data_generation._base import CategoricalColumn


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

    def _generate(self, n: int) -> np.ndarray:
        assert n == len(self._values), (
            f"The StaticCategoricalColumn does not have equal number of values "
            f"to fill a column of size {n}"
        )
        return np.array(self._values, dtype="object")


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

    def __init__(self, name: str, values: List[str], **kwargs):
        super().__init__(name, **kwargs)
        self._values = values

    def _generate(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        return rng.choice(self._values, n, replace=True)
