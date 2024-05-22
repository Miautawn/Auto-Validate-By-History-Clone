"""
Unfortunately, the tensorflow-data-validation is really incompatable with 
    currently selected dependencies, which makes it super hard to install it
    besides them.

As an alternative, you can install only the dependencies needed for tfdv and run this
    script independantly!

To do so, first clean the current virtual environment:
```bash
poetry env remove --all
```

Then install all the necessary libraries:
```bash
poetry shell
>> pip install tensorflow-data-validation, tqdm
```
"""

import tensorflow_data_validation as tfdv
from typing import List, Tuple, Iterable, Optional
from joblib import Parallel, delayed
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import pathlib

class TVDF:
    def __init__(self, train_window_size: int = 30, thresholds: Optional[Iterable] = None):
        self.train_window_size = train_window_size
        self.thresholds = (
            np.array(thresholds)
            if thresholds is not None
            else np.arange(0, 1.0, 0.05)
        )

    def _parse_prediction(self, anomalies) -> bool:
        """
        Check if the TFDV report object contains reported anomalies
        """
        for feature_name, anomaly in anomalies.anomaly_info.items():
            if anomaly.short_description != 'No anomalies found':
                return True
        return False

    def _test_precision(self, column_history: List[pd.DataFrame], column: str) -> np.ndarray:
        fp_per_threshold = np.zeros(shape=len(self.thresholds))
        for i in range(self.total_windows_):
            train_df = pd.concat(column_history[i: i + self.train_window_size])[[column]].reset_index(drop=True)
            test_df = column_history[i + self.train_window_size][[column]]

            train_stats = tfdv.generate_statistics_from_dataframe(train_df)
            train_schema = tfdv.infer_schema(statistics=train_stats)
            test_stats = tfdv.generate_statistics_from_dataframe(test_df)

            for j, threshold in enumerate(self.thresholds):
                feature = tfdv.get_feature(train_schema, column)
                feature.drift_comparator.jensen_shannon_divergence.threshold = threshold

                prediction = tfdv.validate_statistics(
                    statistics=test_stats, schema=train_schema, previous_statistics=train_stats
                )
                fp_per_threshold[j] += self._parse_prediction(prediction)

        return fp_per_threshold


    def _test_recall(
            self,
            column_history: List[pd.DataFrame],
            column_perturbations: List[Tuple[str, pd.Series]],
            column: str
        ) -> np.ndarray:
        tp_per_threshold = np.zeros(shape=len(self.thresholds))

        train_df = pd.concat(column_history[:self.train_window_size])[[column]].reset_index(drop=True)

        train_stats = tfdv.generate_statistics_from_dataframe(train_df)
        train_schema = tfdv.infer_schema(statistics=train_stats)

        for j, threshold in enumerate(self.thresholds):
            feature = tfdv.get_feature(train_schema, column)
            feature.drift_comparator.jensen_shannon_divergence.threshold = threshold

            for issue, data in column_perturbations:
                test_df = data.to_frame()
                test_stats = tfdv.generate_statistics_from_dataframe(test_df)

                prediction = tfdv.validate_statistics(
                    statistics=test_stats, schema=train_schema, previous_statistics=train_stats
                )
                tp_per_threshold[j] += self._parse_prediction(prediction)

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

    tfdv_alg = TVDF()
    col_fp_per_threshold, col_tp_per_threshold = tfdv_alg.test_algorithm(column_history, column_perturbations)

    # calculate the actual metrics
    col_precision_per_threshold = col_tp_per_threshold / (col_fp_per_threshold + col_tp_per_threshold)
    col_recall_per_threshold = col_tp_per_threshold / 23    # number of col perturbations (i.e. recall tests)

    avg_precision_per_threshold = col_precision_per_threshold.mean(axis=0)
    avg_recall_per_threshold = col_recall_per_threshold.mean(axis=0)

    metrics = {"precision": avg_precision_per_threshold, "recall": avg_recall_per_threshold}
    with open(f"{benchmark_dir}/benchmark_tfdv_full_metrics.pickle", "wb") as f:
        pickle.dump(metrics, f)

    
