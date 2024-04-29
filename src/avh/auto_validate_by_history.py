import logging
from typing import List, Dict, Tuple, Callable, Optional, Set, Union
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from tqdm import tqdm

import avh.utility_functions as utils
from avh.metrics import Metric
from avh.constraints import (
    Constraint,
    ConjuctivDQProgram,
    ChebyshevConstraint,
    CLTConstraint
)
from avh.data_issues import (
    IssueTransfomer,
    NumericIssueTransformer,
    CategoricalIssueTransformer,
    IncreasedNulls,
    SchemaChange,
    DistributionChange,
    UnitChange,
    CasingChange,
    DQIssueDatasetGenerator,
    VolumeChangeUpsample,
    VolumeChangeDownsample,
    NumericPerturbation,
)

from copy import deepcopy
from avh.aliases import Seed

class AVH:
    """
    Returns a dictionary with ConjuctivDQProgram for a column
    """

    logger = logging.getLogger(f"{__name__}.AVH")

    def _enable_debug(self, enable: bool):
        self.logger.setLevel(logging.DEBUG if enable else logging.INFO)

    def _reset_verbosity_states(self):
        self._enable_debug(False)

    def __init__(
        self,
        M: List[Metric],
        E: List[Constraint],
        DC: Optional[DQIssueDatasetGenerator] = None,
        columns: Optional[List[str]] = None,
        time_differencing: bool = False,
        random_state: Seed = None,
        verbose: int = 1
    ):

        self.M = M
        self.E = E
        self.DC = DC
        self.columns = columns
        self.time_differencing = time_differencing
        self.random_state = random_state

        self.verbose = verbose

        if self.DC is None:
            self.DC = self._get_default_issue_transformer()

    @property
    def verbose(self) -> int:
        if self._verbose == 0:
            return False
        return True

    @verbose.setter
    def verbose(self, level: Union[int, bool]):
        assert level >= 0, "Verbosity level must be a positive integer"

        self._reset_verbosity_states()
        self._verbose = level

        if level >= 2:
            self._enable_debug(True)

    @utils.debug_timeit(f"{__name__}.AVH")
    def generate(
        self, history: List[pd.DataFrame], fpr_target: float
    ) -> Dict[str, ConjuctivDQProgram]:
        PS = {}

        DC = self.DC.generate(history[-1])
        columns = self.columns if self.columns else list(history[0].columns)

        for column in tqdm(columns, "Generating P(S for columns...", disable=not self.verbose):
            Q = self._generate_constraint_space(
                [run[column] for run in history[:-1]]
            )

            PS[column] = self._generate_conjuctive_dq_program(
                Q, DC[column], fpr_target
            )

        return PS

    def generate_batched(
        self, history: List[pd.DataFrame], fpr_target: float
    ) -> Dict[str, ConjuctivDQProgram]:
        PS = {}

        DC = self.issue_dataset_generator.fit_transform(history[-1])
        columns = self.columns if self.columns else list(history[0].columns)

        Q = {}
        start = time.time()
        for column in tqdm(columns, "Generating Q for columns..."):
            q = self._generate_constraint_space([run[column] for run in history[:-1]])
            Q[column] = q
        end = time.time()
        print(f"Q generation took: {end-start}")

        start = time.time()
        for column in tqdm(columns, "Generating P(S) for columns..."):
            PS[column] = self._generate_conjuctive_dq_program(
                Q[column], DC[column], fpr_target
            )
        end = time.time()
        print(f"PS generation took: {end-start}")

        return PS

    def generate_batched_threaded(
        self, history: List[pd.DataFrame], fpr_target: float
    ) -> Dict[str, ConjuctivDQProgram]:
        PS = {}

        DC = self.issue_dataset_generator.fit_transform(history[-1])
        columns = self.columns if self.columns else list(history[0].columns)

        Q = {}
        start = time.time()
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._generate_constraint_space,
                    [run[column] for run in history[:-1]]
                )
                for column in columns
            ]
            
            for result in tqdm(as_completed(futures), "Generating Q for columns...", total=len(columns)):
                _ = result.result()
                
        end = time.time()
        print(f"Q generation took: {end-start}")

        # start = time.time()
        # for column in tqdm(columns, "Generating P(S) for columns..."):
        #     PS[column] = self._generate_conjuctive_dq_program(
        #         Q[column], DC[column], fpr_target
        #     )
        # end = time.time()
        # print(f"PS generation took: {end-start}")

        return PS

    def generate_batched_parallel(
        self, history: List[pd.DataFrame], fpr_target: float
    ) -> Dict[str, ConjuctivDQProgram]:
        PS = {}

        DC = self.issue_dataset_generator.fit_transform(history[-1])
        columns = self.columns if self.columns else list(history[0].columns)

        Q = {}
        start = time.time()
        arguments = ((column, [run[column] for run in history[:-1]]) for column in columns)
        with mp.Pool() as executor:
            results = executor.imap_unordered(
                self._generate_constraint_space_parallel_args, arguments, chunksize=10
            )
            for column, q in tqdm(results, "Generating Q for columns...", total=len(columns)):
                Q[column] = q
                
        end = time.time()
        print(f"Q generation took: {end-start}")

        start = time.time()
        arguments = ((column, DC[column], Q[column], fpr_target) for column in columns)
        with mp.Pool() as executor:
            results = executor.imap_unordered(
                self._generate_conjuctive_dq_program_parallel_args, arguments, chunksize=10
            )
            for column, ps in tqdm(results, "Generating P(S) for columns...", total=len(columns)):
                PS[column] = ps

        end = time.time()
        print(f"PS generation took: {end-start}")

        return PS

    def _generate_constraint_space_parallel_args(self, args):
        column, history = args
        return column, self._generate_constraint_space(history)

    def _generate_conjuctive_dq_program_parallel_args(self, args):
        column, DC, Q, fpr_target = args
        return column, self._generate_conjuctive_dq_program(Q, DC, fpr_target)

    def generate_parallel(
        self, history: List[pd.DataFrame], fpr_target: float, multiprocess=False
    ) -> Dict[str, ConjuctivDQProgram]:
        PS = {}

        DC = self.issue_dataset_generator.fit_transform(history[-1])
        columns = self.columns if self.columns else list(history[0].columns)

        print("opa")
        arguments = [
            (column, [run[column] for run in history[:-1]], DC[column], fpr_target)
            for column in columns
        ]
        print("let's go!!!")
        with mp.Pool() as executor:
            results = executor.imap_unordered(
                self._apply_stuff, arguments, chunksize=10
            )
            for column, ps in tqdm(results, "creating P(S)...", total=len(columns)):
                PS[column] = ps

        return PS

    def _apply_stuff(self, args):
        column, history, DC, fpr_target = args
        Q = self._generate_constraint_space(history)
        return column, self._generate_conjuctive_dq_program(Q, DC, fpr_target)

    @utils.debug_timeit(f"{__name__}.AVH")
    def _generate_constraint_space(self, history: List[pd.Series]) -> List[Constraint]:
        Q = []
        for metric in self.M:
            if not metric.is_column_compatable(history[0].dtype):
                continue

            metric_history = metric.calculate(history)
            metric_history_std = np.nanstd(metric_history)
            # preprocessed_metric_history = None

            # TODO: put this into another function
            # lag, preprocessing_func = 0, utils.identity
            # if self.time_differencing:
            #     is_stationary, lag, preprocessing_func = self._time_series_difference(
            #         metric_history
            #     )
            #     if not is_stationary:
            #         continue

            #     preprocessed_metric_history = diff(
            #         preprocessing_func(metric_history), lag
            #     )

            for constraint_estimator in self.E:
                if not constraint_estimator.is_metric_compatable(metric):
                    continue

                # TODO: improve this to work with more general cases
                # 'intelligent' beta hyperparameter search optimisation.
                #    The justification is simple:
                #        "in production, no one would need 25% expected FPR,
                #         which comes with beta = 2 * std on Chebyshev,
                #         or 0% which comes after beta = 4 * std on CTL"
                beta_start = (
                    metric_history_std * 5 if(constraint_estimator == ChebyshevConstraint)
                    else metric_history_std
                )
                beta_end = (
                    metric_history_std * 10 if(constraint_estimator == ChebyshevConstraint)
                    else metric_history_std * 4
                )
                
                beta_increment_n = 10 if metric_history_std != 0.0 else 1
                for beta in np.linspace(beta_start, beta_end, beta_increment_n):
                    q = constraint_estimator(
                        metric,
                        # differencing_lag=lag,
                        # preprocessing_func=preprocessing_func,
                    ).fit(
                        metric_history,
                        beta=beta,
                        hotload_history=True,
                        # preprocessed_metric_history=preprocessed_metric_history,
                    )
                    Q.append(q)
        return Q

    @utils.debug_timeit(f"{__name__}.AVH")
    def _precalculate_constraint_recalls(
        self, Q: List[Constraint], DC: List[Tuple[str, pd.Series]]
    ) -> List[Set[str]]:
        return [
            {issue for issue, data in DC if not constraint.predict(data)}
            for constraint in Q
        ]

    @utils.debug_timeit(f"{__name__}.AVH")
    def _precalculate_constraint_recalls_fast(
        self, Q: List[Constraint], DC: List[Tuple[str, pd.Series]]
    ) -> List[Set[str]]:
        """
        Serves the exact same purpose as _precalculate_constraint_recalls
            but tries to optimise the calculations by precalculating the metric values
            for common constraint predictions.

        This optimisation implementation is highly coupled with current Q space generation,
            since it expects common-metric constraints to be clustered.
        """
        individual_recalls = [set({}) for constraint in Q]

        for issue, data in DC:
            cached_metric = Q[0].metric
            precalculated_metric = cached_metric.calculate(data)
            for idx, constraint in enumerate(Q):
                if not issubclass(constraint.metric, cached_metric):
                    cached_metric = constraint.metric
                    precalculated_metric = cached_metric.calculate(data)
                if not constraint._predict(precalculated_metric):
                    individual_recalls[idx].add(issue)

        return individual_recalls

    @utils.debug_timeit(f"{__name__}.AVH")
    def _find_optimal_singleton_conjuctive_dq_program(
        self, Q: List[Constraint], constraint_recalls: List[Set[str]], fpr_target: float
    ) -> ConjuctivDQProgram:
        best_singleton_constraint_idx = np.argmax(
            [
                len(recall) if Q[idx].expected_fpr_ < fpr_target else 0
                for idx, recall in enumerate(constraint_recalls)
            ]
        )

        return ConjuctivDQProgram(
            constraints=[Q[best_singleton_constraint_idx]],
            recall=constraint_recalls[best_singleton_constraint_idx],
            contributions=[constraint_recalls[best_singleton_constraint_idx]],
        )

    @utils.debug_timeit(f"{__name__}.AVH")
    def _find_optimal_conjunctive_dq_program(
        self, Q: List[Constraint], constraint_recalls: List[Set[str]], fpr_target: float
    ) -> ConjuctivDQProgram:
        current_fpr = 0.0
        q_indexes = list(range(len(Q)))
        ps = ConjuctivDQProgram()
        while current_fpr < fpr_target and len(q_indexes) != 0:
            recall_increments = [
                constraint_recalls[idx].difference(ps.recall) for idx in q_indexes
            ]

            # stop if there are no more recall improvements possible
            if len(max(recall_increments)) == 0:
                break

            best_idx = np.argmax(
                [
                    len(recall_set) / (Q[idx].expected_fpr_ + 1)    # +1 is to avoid division by 0
                    for idx, recall_set in zip(q_indexes, recall_increments)
                ]
            )

            best_constraint = Q[q_indexes[best_idx]]
            if best_constraint.expected_fpr_ + current_fpr <= fpr_target:
                current_fpr += best_constraint.expected_fpr_
                ps.constraints.append(best_constraint)
                ps.recall.update(recall_increments[best_idx])
                ps.contributions.append(recall_increments[best_idx])

            q_indexes.pop(best_idx)

        return ps


    @utils.debug_timeit(f"{__name__}.AVH")
    def _generate_conjuctive_dq_program(
        self, Q: List[Constraint], DC: List[Tuple[str, pd.Series]], fpr_target: float
    ):
        individual_recalls = self._precalculate_constraint_recalls_fast(Q, DC)

        ps_singleton = self._find_optimal_singleton_conjuctive_dq_program(
            Q, individual_recalls, fpr_target
        )

        ps = self._find_optimal_conjunctive_dq_program(
            Q, individual_recalls, fpr_target
        )

        return ps if len(ps.recall) > len(ps_singleton.recall) else ps_singleton

    # @utils.timeit_decorator
    def _time_series_difference(
        self, metric_history: List[float]
    ) -> Tuple[bool, int, Callable]:
        """
        Performs time series differencing search to find stationary form
            of the provided metric history distribution.

        Returns:
        bool - whether the stationarity was achieved
        int - found lag window that achieved stationarity
        Callable - metric preprocessing function
        """

        def is_stationary(metric_history):
            return adfuller(metric_history)[1] <= 0.05

        def search_for_stationarity(metric_history):
            for l in range(1, 8):
                metric_history_with_lag = metric_history.diff(l)[l:]
                if is_stationary(metric_history_with_lag):
                    return True, l
            return False, 0

        if is_stationary(metric_history):
            return True, 0, utils.identity

        # Perform lag transformation
        status, window = search_for_stationarity(metric_history)
        if status:
            return status, window, utils.identity

        # Perform lag transformation with log transformation
        log_metric_history = pd.Series(safe_log(metric_history))
        status, window = search_for_stationarity(log_metric_history)
        if status:
            return status, window, utils.safe_log

        return False, 0, identity

    def _get_default_issue_transformer(self) -> DQIssueDatasetGenerator:
        """
        Constructs a DQIssueDatasetTransformer instance
            with DQ issues and parameter space described in the paper
        """

        return DQIssueDatasetGenerator(
            issues=[
                (SchemaChange, {"p": [0.1, 0.5, 1.0]}),
                (UnitChange, {"p": [0.1, 1.0], "m": [10, 100, 1000]}),
                (IncreasedNulls, {"p": [0.1, 0.5, 1.0]}),
                (VolumeChangeUpsample, {"f": [2, 10]}),
                (VolumeChangeDownsample, {"f": [0.5, 0.1]}),
                (DistributionChange, {"p": [0.1, 0.5], "take_last": [True, False]}),
                (NumericPerturbation, {"p": [0.1, 0.5, 1.0]}),
                # (CasingChange, {"p": [0.01, 0.1, 1.0]}),
            ],
            random_state=self.random_state,
            verbose = self._verbose
        )