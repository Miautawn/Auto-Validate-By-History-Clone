import numpy as np

from avh.data_generation._base import NumericColumn


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

    def _generate(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        return rng.uniform(self._lower_bound, self._upper_bound, n)

    def _update_parameters(self, n: int, i: int):
        assert self._parameter_function is not None  # Mypy complains otherwise
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

    def _generate(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        return rng.normal(self._mean, self._std, n)

    def _update_parameters(self, n: int, i: int):
        assert self._parameter_function is not None  # Mypy complains otherwise
        self._mean, self._std = self._parameter_function(n, i, self._mean, self._std)


class BetaNumericColumn(NumericColumn):
    """
    Concrete numeric column class.
    Generates data from beta distribution.

    Parameters
    ----------
    name: str
        The name of the column in the final output
    alfa: float
        alfa parameter of Beta distribution
    beta: float
        beta parameter of Beta distribution
    scale: float
        factor by which the sampled value will be multiplied with
    shift: float
        number by which the sampled value will be shifted by
    **kwargs:
        Any other parameters will be forwarded back to parent classes
    """

    def __init__(self, name: str, alfa: float, beta: float, **kwargs):
        super().__init__(name, **kwargs)
        self._alfa = alfa
        self._beta = beta

    def _generate(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        return rng.beta(self._alfa, self._beta, n)

    def _update_parameters(self, n: int, i: int):
        assert self._parameter_function is not None  # Mypy complains otherwise
        self._alfa, self._beta = self._parameter_function(n, i, self._alfa, self._beta)
