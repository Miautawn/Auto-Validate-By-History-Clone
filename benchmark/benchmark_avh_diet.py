from typing import List, Tuple, Iterable, Optional
from scipy.stats import ks_2samp
from joblib import Parallel, delayed
from avh.auto_validate_by_history import AVH
from avh.constraints import Constraint, ConjuctivDQProgram
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import pathlib

class AutoValidateByHistory:
    def __init__(self, train_window_size: int = 30, fpr_budgets: Optional[Iterable] = None):
        self.train_window_size = train_window_size
        self.fpr_budgets = (
            np.array(fpr_budgets)
            if fpr_budgets is not None
            else [
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
            ]
        )

    def _optimally_construct_ps(
            self, avh: AVH, Q: List[Constraint], constraint_recalls: List[set], fpr_budget: float
        ) -> ConjuctivDQProgram:
        """
        A helper function that uses hidden AVH methods to calculate PS.
        The PS creation steps are identical to avh._generate_conjuctive_dq_program().

        This method is used to speed up the calculations for different fpr values 
            when individual constrain recalls are already precalculated.
        """
        PS_singleton = avh._find_optimal_singleton_conjuctive_dq_program(
            Q, constraint_recalls, fpr_budget
        )

        PS_conjunctive = avh._find_optimal_conjunctive_dq_program(
            Q, constraint_recalls, fpr_budget
        )

        return (
            PS_conjunctive
            if len(PS_conjunctive.recall) >= len(PS_singleton.recall)
            else PS_singleton
        )

    def _test_precision(self, column_history: List[pd.DataFrame], column: str):
        fp_per_threshold = np.zeros(shape=len(self.fpr_budgets))

        avh = AVH(columns=[column], verbose=0, random_state=42, optimise_search_space=False)
        dc_generator = avh._get_default_issue_dataset_generator()
        
        for i in range(self.total_windows_):
            train_h = column_history[i : i + self.train_window_size]
            test_h = column_history[i + self.train_window_size]

            Q = avh._generate_constraint_space(
                [run[column] for run in train_h], optimise_search_space=False
            )
            DC = dc_generator.generate(train_h[-1])[column]
            constraint_recalls = avh._precalculate_constraint_recalls_fast(Q, DC)

            for j, fpr_budget in enumerate(self.fpr_budgets):
                PS = self._optimally_construct_ps(avh, Q, constraint_recalls, fpr_budget)

                column_prediction = not PS.predict(test_h[column])
                fp_per_threshold[j] += column_prediction

        return fp_per_threshold


    def _test_recall(
            self,
            column_history: List[pd.DataFrame],
            column_perturbations: List[Tuple[str, pd.Series]],
            column: str
        ) -> np.ndarray:
        tp_per_threshold = np.zeros(shape=len(self.fpr_budgets))

        avh = AVH(columns=[column], verbose=0, random_state=42, optimise_search_space=False)
        dc_generator = avh._get_default_issue_dataset_generator()

        train_h = column_history[:self.train_window_size]

        Q = avh._generate_constraint_space(
                [run[column] for run in train_h], optimise_search_space=False
            )
        DC = dc_generator.generate(train_h[-1])[column]
        constraint_recalls = avh._precalculate_constraint_recalls_fast(Q, DC)

        for j, fpr_budget in enumerate(self.fpr_budgets):
            PS = self._optimally_construct_ps(avh, Q, constraint_recalls, fpr_budget)

            column_predictions = [not PS.predict(data) for issue, data in column_perturbations]
            tp_per_threshold[j] += sum(column_predictions)

        return tp_per_threshold


    def _test_algorithm_worker(
            self,
            column_history: List[pd.DataFrame],
            column_perturbations: List[Tuple[str, pd.Series]],
            column: str
        ) -> Tuple[np.ndarray, np.ndarray]:
        fp_per_threshold = self._test_precision(column_history, column)
        tp_per_thershold = self._test_recall(column_history, column_perturbations, column)

        return fp_per_threshold, tp_per_thershold
    
    def test_algorithm(
            self,
            column_history: List[List[pd.DataFrame]],
            column_perturbations: List[List[Tuple[str, pd.Series]]]
        ) -> Tuple[np.ndarray, np.ndarray]:
        self.total_history_size_ = len(column_history)
        self.total_windows_ = self.total_history_size_ - self.train_window_size
        
        columns = list([column_set.columns[0] for column_set in column_history[0]])
        results = Parallel(n_jobs=-1, timeout=9999, return_as="generator")(
            delayed(self._test_algorithm_worker)
            ([run[i] for run in column_history], column_perturbations[i], col) for i, col in enumerate(columns)
        )

        col_fp_per_threshold, col_tp_per_threshold = [], []
        for fp_array, tp_array in tqdm(results, total=len(columns)):
            col_fp_per_threshold.append(fp_array)
            col_tp_per_threshold.append(tp_array)

        return  np.array(col_fp_per_threshold), np.array(col_tp_per_threshold)


if __name__ == "__main__":

    benchmark_dir = pathlib.Path(__file__).parent
    with open(f"{benchmark_dir}/benchmark_data.pickle", "rb") as f:
        benchmark_data = pickle.load(f)

    column_history = benchmark_data["column_history"]
    column_perturbations = benchmark_data["column_perturbations"]

    ks = AutoValidateByHistory()
    col_fp_per_threshold, col_tp_per_threshold = ks.test_algorithm(column_history, column_perturbations)

    # calculate the actual metrics
    col_precision_per_threshold = col_tp_per_threshold / (col_fp_per_threshold + col_tp_per_threshold)
    col_recall_per_threshold = col_tp_per_threshold / 23    # number of col perturbations (i.e. recall tests)

    avg_precision_per_threshold = col_precision_per_threshold.mean(axis=0)
    avg_recall_per_threshold = col_recall_per_threshold.mean(axis=0)

    metrics = {"precision": avg_precision_per_threshold, "recall": avg_recall_per_threshold}
    with open(f"{benchmark_dir}/benchmark_avh_diet_metrics.pickle", "wb") as f:
        pickle.dump(metrics, f)

    
