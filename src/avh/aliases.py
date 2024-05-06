from typing import Tuple, TypeAlias, Union

import numpy as np

IntRange: TypeAlias = Union[int, Tuple[int, int]]
FloatRange: TypeAlias = Union[float, Tuple[float, float]]
Seed: TypeAlias = Union[None, int, np.random.Generator]
