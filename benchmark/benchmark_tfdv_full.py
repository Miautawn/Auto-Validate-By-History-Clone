from typing import List, Tuple, Iterable, Optional
from scipy.stats import ks_2samp
from joblib import Parallel, delayed
from avh.auto_validate_by_history import AVH
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from avh.metrics import JsDivergence

import pathlib

class TVDF:
    def __init__(self, train_window_size: int = 30, thresholds: Optional[Iterable] = None):
        self.train_window_size = train_window_size
        self.thresholds = (
            np.array(thresholds)
            if thresholds is not None
            else np.array([0.000001, 0.00001, 0.0001, 0.001, 0.01, 
                          0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
                          0.9, 1.0, 10, 100])
        )

    def _test_precision(self, column_history: List[pd.DataFrame], column: str) -> np.ndarray:
        fp_per_threshold = np.zeros(shape=len(self.thresholds))
        for j in range(self.total_windows_):
            train_sample = pd.concat(column_history[j: j + self.train_window_size])[column]
            test_sample = column_history[j + self.train_window_size][column]

            js_divergence = JsDivergence.calculate(test_sample, train_sample)
            fp_per_threshold += js_divergence > self.thresholds

        return fp_per_threshold


    def _test_recall(
            self,
            column_history: List[pd.DataFrame],
            column_perturbations: List[Tuple[str, pd.Series]],
            column: str
        ) -> np.ndarray:
        tp_per_threshold = np.zeros(shape=len(self.thresholds))

        train_sample = pd.concat(column_history[:self.train_window_size])[column]

        for recall_test in column_perturbations:
            test_sample = recall_test[1]

            # if the test sample is empty, automatically return a true positive
            #   since it's realistic to think that such a drastic change should be caught by default
            if test_sample.count() == 0:
                tp_per_threshold += 1
            else:
                js_divergence = JsDivergence.calculate(test_sample, train_sample)
                tp_per_threshold += js_divergence > self.thresholds

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

    tfdv = TVDF()
    col_fp_per_threshold, col_tp_per_threshold = tfdv.test_algorithm(column_history, column_perturbations)

    # calculate the actual metrics
    col_precision_per_threshold = col_tp_per_threshold / (col_fp_per_threshold + col_tp_per_threshold)
    col_recall_per_threshold = col_tp_per_threshold / 23    # number of col perturbations (i.e. recall tests)

    avg_precision_per_threshold = col_precision_per_threshold.mean(axis=0)
    avg_recall_per_threshold = col_recall_per_threshold.mean(axis=0)

    metrics = {"precision": avg_precision_per_threshold, "recall": avg_recall_per_threshold}
    with open(f"{benchmark_dir}/benchmark_tfdv_full_metrics.pickle", "wb") as f:
        pickle.dump(metrics, f)

    
