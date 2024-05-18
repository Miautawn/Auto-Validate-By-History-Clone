from typing import List, Tuple, Iterable, Optional
from scipy.stats import ks_2samp
from joblib import Parallel, delayed
from avh.auto_validate_by_history import AVH
import avh.metrics as metrics
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

import pathlib

class LOC:
    def __init__(self, train_window_size: int = 30, thresholds: Optional[Iterable] = None):
        self.train_window_size = train_window_size
        self.thresholds = (
            np.array(thresholds)
            if thresholds is not None
            else np.arange(0, 1, 0.01)
        )
        self.scaler = StandardScaler()
        self.lof = LocalOutlierFactor(n_neighbors=30, contamination="auto", novelty=True)

    def _test_precision(self, feature_history: np.ndarray) -> np.ndarray:
        fp_per_threshold = np.zeros(shape=len(self.thresholds))

        for j in range(self.total_windows_):
            train_x = feature_history[j : j + self.train_window_size]
            test_x = feature_history[j + self.train_window_size].reshape(1, -1)

            train_x_scaled = self.scaler.fit_transform(train_x)
            test_x_scaled = self.scaler.transform(test_x)

            self.lof.fit(train_x_scaled)
            shifted_local_outlier_factor = self.lof.decision_function(test_x_scaled)

            fp_per_threshold += shifted_local_outlier_factor < self.thresholds

        return fp_per_threshold


    def _test_recall(
        self,
        feature_history: np.ndarray,
        column_perturbations: List[Tuple[str, pd.Series]]
    ) -> np.ndarray:
        tp_per_threshold = np.zeros(shape=len(self.thresholds))

        train_x = feature_history[:self.train_window_size]

        train_x_scaled = self.scaler.fit_transform(train_x)
        self.lof.fit(train_x_scaled)

        for recall_test in column_perturbations:
            test_x = self._extract_features(recall_test[1]).reshape(1, -1)
            test_x_scaled = self.scaler.transform(test_x)

            shifted_local_outlier_factor = self.lof.decision_function(test_x_scaled)
            tp_per_threshold += shifted_local_outlier_factor < self.thresholds

        return tp_per_threshold
    
    def _extract_features(self, data: pd.Series) -> np.ndarray:
        row_count = metrics.RowCount.calculate(data)
        min_val = metrics.Min.calculate(data)
        max_val = metrics.Max.calculate(data)
        mean_val = metrics.Mean.calculate(data)
        median_val = metrics.Median.calculate(data)
        sum_val = metrics.Sum.calculate(data)
        range_val = metrics.Range.calculate(data)
        distinct_ratio = metrics.DistinctRatio.calculate(data)
        complete_ratio = metrics.CompleteRatio.calculate(data)

        features = [
            row_count,
            min_val,
            max_val,
            mean_val,
            median_val,
            sum_val,
            range_val,
            distinct_ratio,
            complete_ratio,
        ]

        return np.array(features)


    def _test_algorithm_worker(
            self,
            column_history: List[pd.DataFrame],
            column_perturbations: List[Tuple[str, pd.Series]],
            column: str
        ) -> Tuple[np.ndarray, np.ndarray]:
        feature_history = [self._extract_features(run[column]) for run in column_history]
        fp_per_threshold = self._test_precision(feature_history)
        tp_per_thershold = self._test_recall(feature_history, column_perturbations)

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

    ks = LOC()
    col_fp_per_threshold, col_tp_per_threshold = ks.test_algorithm(column_history, column_perturbations)

    # calculate the actual metrics
    col_precision_per_threshold = col_tp_per_threshold / (col_fp_per_threshold + col_tp_per_threshold)
    col_recall_per_threshold = col_tp_per_threshold / 23    # number of col perturbations (i.e. recall tests)

    avg_precision_per_threshold = col_precision_per_threshold.mean(axis=0)
    avg_recall_per_threshold = col_recall_per_threshold.mean(axis=0)

    metrics = {"precision": avg_precision_per_threshold, "recall": avg_recall_per_threshold}
    with open(f"{benchmark_dir}/benchmark_lof_metrics.pickle", "wb") as f:
        pickle.dump(metrics, f)