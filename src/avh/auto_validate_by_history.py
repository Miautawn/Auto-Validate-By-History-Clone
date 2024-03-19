import logging
from typing import List, Dict, Tuple, Callable, Optional, Set
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
from avh.data_quality_issues import (
    DQIssueDatasetTransformer,
    SchemaChange,
    UnitChange,
    CasingChange,
    IncreasedNulls,
    VolumeChange,
    DistributionChange
)

class AVH:
    """
    Returns a dictionary with ConjuctivDQProgram for a column
    """

    logger = logging.getLogger(f"{__name__}.AVH")

    def __init__(
        self,
        M: List[Metric],
        E: List[Constraint],
        DC: Optional[DQIssueDatasetTransformer] = None,
        columns: Optional[List[str]] = None,
        time_differencing: bool = False,
    ):
        self.M = M
        self.E = E
        self.columns = columns
        self.time_differencing = time_differencing

        self.issue_dataset_generator = (
            DC if DC else self._get_default_issue_transformer()
        )

    def enable_debug(self, enable: bool):
        self.logger.setLevel(logging.DEBUG if enable else logging.INFO)

    def generate(
        self, history: List[pd.DataFrame], fpr_target: float, multiprocess=False
    ) -> Dict[str, ConjuctivDQProgram]:
        PS = {}

        DC = self.issue_dataset_generator.fit_transform(history[-1])
        columns = self.columns if self.columns else list(history[0].columns)

        for column in tqdm(columns, "Generating P(S for columns..."):
            start = time.time()
            Q = self._generate_constraint_space(
                [run[column] for run in history[:-1]]
            )
            end = time.time()
            DEBUG_MODE and print(f"Q generation took: {end-start}")

            start = time.time()
            PS[column] = self._generate_conjuctive_dq_program(
                Q, DC[column], fpr_target
            )
            end = time.time()
            print(f"PS generation took: {end-start}")

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

    # @utils.timeit_decorator
    def _generate_constraint_space(self, history: List[pd.Series]) -> List[Constraint]:
        Q = []
        for metric in self.M:
            if not metric.is_column_compatable(history[0]):
                continue

            metric_history = metric.calculate(history)
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

                # 'intelligent' beta hyperparameter search optimisation.
                #    The justification is simple:
                #        "in production, no one would need 25% expected FPR,
                #         which comes with beta = 2 * std on Chebyshev,
                #         or 0% which comes after beta = 4 * std on CTL"
                std = np.nanstd(metric_history)
                beta_start = (
                    std * 5 if(constraint_estimator == ChebyshevConstraint)
                    else std
                )
                beta_end = (
                    std * 10 if(constraint_estimator == ChebyshevConstraint)
                    else std * 4
                )
                
                for beta in np.linspace(beta_start, beta_end, (10 if std != 0.0 else 1)):
                    q = constraint_estimator(
                        metric,
                        # differencing_lag=lag,
                        # preprocessing_func=preprocessing_func,
                    ).fit(
                        metric_history,
                        beta=beta,
                        # hotload_history=True,
                        # preprocessed_metric_history=preprocessed_metric_history,
                    )
                    Q.append(q)
        return Q

    # @utils.timeit_decorator
    def _generate_conjuctive_dq_program(
        self, Q: List[Constraint], DC: List[Tuple[str, pd.Series]], fpr_target: float
    ):
        # precalculating recall for each constraint
        start = time.time()
        individual_recalls = [
            {issue for issue, data in DC if not constraint.predict(data)}
            for constraint in Q
        ]
        end = time.time()
        print(f"recall calculations took: {end - start}")

        # finding the best singleton Q
        start = time.time()
        singleton_idx = np.argmax(
            [
                len(recall) if Q[idx].expected_fpr_ < fpr_target else 0
                for idx, recall in enumerate(individual_recalls)
            ]
        )
        PS_singleton = ConjuctivDQProgram(
            constraints=[Q[singleton_idx]],
            recall=individual_recalls[singleton_idx],
            contributions=[individual_recalls[singleton_idx]],
        )
        end = time.time()
        print(f"Singleton finding took: {end - start}")

        # finding the best set of Q based on their recall contributions
        start = time.time()
        current_fpr = 0.0
        Q_indexes = list(range(len(Q)))
        PS = ConjuctivDQProgram()
        while current_fpr < fpr_target and Q_indexes:
            recall_increments = [
                individual_recalls[idx].difference(PS.recall) for idx in Q_indexes
            ]

            if len(max(recall_increments)) == 0:
                break

            best_idx = np.argmax(
                [
                    len(recall) / (Q[idx].expected_fpr_ + 1)
                    for idx, recall in zip(Q_indexes, recall_increments)
                ]
            )

            best_constraint = Q[Q_indexes[best_idx]]
            if best_constraint.expected_fpr_ + current_fpr <= fpr_target:
                current_fpr += best_constraint.expected_fpr_
                PS.constraints.append(best_constraint)
                PS.recall.update(recall_increments[best_idx])
                PS.contributions.append(recall_increments[best_idx])

            Q_indexes.pop(best_idx)
        end = time.time()
        print(f"Main loop took: {end - start}")
        return PS if len(PS.recall) > len(PS_singleton.recall) else PS_singleton

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

    def _get_default_issue_transformer(self) -> DQIssueDatasetTransformer:
        """
        Constructs a DQIssueDatasetTransformer instance
            with DQ issues and parameter space described in the paper
        """

        return DQIssueDatasetTransformer(
            (SchemaChange, {"p": [0.1, 0.5, 1.0]}),
            (UnitChange, {"m": [10, 100, 1000]}),
            (CasingChange, {"p": [0.01, 0.1, 1.0]}),
            (IncreasedNulls, {"p": [0.1, 0.5, 1.0]}),
            (VolumeChange, {"f": [0.1, 0.5, 2.0, 10.0]}),
            (DistributionChange, {"p": [0.1, 0.5], "take_last": [True, False]}),
        )