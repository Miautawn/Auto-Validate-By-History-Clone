from typing import List, Optional, Set

import pandas as pd

from avh.constraints._base import Constraint


class ConjuctivDQProgram:
    def __init__(
        self,
        constraints: Optional[List[Constraint]] = None,
        recall: Optional[Set[str]] = None,
        contributions: Optional[List[Set[str]]] = None,
    ):
        self.constraints = constraints if constraints else []
        self.recall = recall if recall else set()
        self.contributions = contributions if contributions else []

    def __repr__(self):
        return "{constraints}, FPR = {fpr:4f}".format(
            constraints="\n".join([repr(q) for q in self.constraints]),
            fpr=self.expected_fpr,
        )

    @property
    def expected_fpr(self):
        return sum([q.expected_fpr_ for q in self.constraints])

    # TODO:?
    # def fit()

    def predict(self, column: pd.Series, **kwargs):
        for constraint in self.constraints:
            if not constraint.predict(column, **kwargs):
                return False
        return True
