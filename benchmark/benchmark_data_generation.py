
from typing import List, Tuple
import numpy as np
import pandas as pd
import pickle
import pathlib
from tqdm import tqdm
from avh.data_generation import UniformNumericColumn, NormalNumericColumn, BetaNumericColumn, DataGenerationPipeline
from avh.auto_validate_by_history import AVH

def generate_uniform_column(rng, dtype, sign, scale, shift, column_name):
    low = scale
    high = low + shift

    if sign == -1:
        low, high = high * sign, low * sign

    return UniformNumericColumn(column_name, low, high, dtype=dtype)


def generate_normal_column(rng, dtype, sign, scale, shift, column_name):
    mean = shift * sign
    std = scale
    return NormalNumericColumn(column_name, mean, std, dtype=dtype)


def generate_beta_column(rng, dtype, sign, scale, shift, column_name):
    alfa = rng.uniform(0.1, 10)
    beta = rng.uniform(0.1, 10)
    return BetaNumericColumn(column_name, alfa, beta, scale=scale * sign, shift=shift, dtype=dtype)

def generate_column(rng, column_name, dtype=None, distribution=None, sign=None):

    if dtype is None:
        dtype = rng.choice([np.int32, np.float32], p=[0.3, 0.7])
    if distribution is None:
        distribution = rng.choice(["uniform", "normal", "beta"], p=[0.1, 0.1, 0.8])
    if sign is None:
        sign = rng.choice([-1, 1], p=[0.2, 0.8])

    if dtype == np.int32:
        scale = rng.integers(1, 1000)
        shift = rng.integers(1, 1000)
    else:
        scale = 10 ** rng.uniform(np.log10(0.001), np.log10(1000))
        shift = 10 ** rng.uniform(np.log10(0.001), np.log10(1000))

    if distribution == "uniform":
        column = generate_uniform_column(rng, dtype, sign, scale, shift, column_name)

    elif distribution == "normal":
        column = generate_normal_column(rng, dtype, sign, scale, shift, column_name)

    elif distribution == "beta":
        column = generate_beta_column(rng, dtype, sign, scale, shift, column_name)

    return column


def generate_column_history(n: int, rng: np.random.Generator) -> List[List[pd.DataFrame]]:

    # Each 'column' will comprised of 2 column pipeline,
    #   where first column will be the actual data column,
    #   while the other one will be a supporting column,
    #   used for data issue generation such as schema change
    numeric_columns_pipelines = []
    for i in range(n):

        column_name = f"numeric_{i}"
        column_neighbor_name = f"numeric_{i}_neighbor"

        column = generate_column(rng, column_name)
        neighbor_column = generate_column(rng, column_neighbor_name, dtype=column.dtype)

        column_pipeline = DataGenerationPipeline([column, neighbor_column], random_state=rng)
        numeric_columns_pipelines.append(column_pipeline)

    data = [
        [column_pipeline.generate_normal(20000, 1000) for column_pipeline in numeric_columns_pipelines]
        for i, _ in enumerate(tqdm(range(60), desc="Generating column executions..."))
    ]

    return data

def generate_column_perturbations(
        column_history: List[List[pd.DataFrame]], rng: np.random.Generator, 
    ) -> List[List[Tuple[str, pd.Series]]]:

    dq_generator = AVH()._get_default_issue_dataset_generator(random_state=rng)

    # we generate column perturbations for the 31'st execution
    # where each perturbation will be a separate recall test
    column_perturbations = []
    for column_set in tqdm(column_history[30], desc="Generaing DQ sets for columns.."):
        target_column = column_set.columns[0]
        column_perturbation_set = dq_generator.generate(column_set)[target_column]

        column_perturbations.append(column_perturbation_set)

    return column_perturbations


if __name__ == "__main__":

    rng = np.random.default_rng(42)
    n = 10

    column_history = generate_column_history(n, rng)
    column_perturbations = generate_column_perturbations(column_history, rng)

    benchmark_dir = pathlib.Path(__file__).parent
    with open(f"{benchmark_dir}/benchmark_data.pickle", "wb") as f:
        benchmark_data = {
            "column_history": column_history,
            "column_perturbations": column_perturbations
        }
        pickle.dump(benchmark_data, f)

    