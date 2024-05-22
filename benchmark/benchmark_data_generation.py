
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import pickle
import pathlib
from tqdm import tqdm
from avh.data_issues import IncreasedNulls
from avh.data_generation import UniformNumericColumn, NormalNumericColumn, BetaNumericColumn, DataGenerationPipeline
from avh.auto_validate_by_history import AVH

def generate_column_issues(rng, column_name, dtype=None):
    column_issues = []

    # Adding null values, if dtype is not integer
    # Since you can't have int columns with np.nan values :(
    if dtype is not np.int32:
        if rng.choice([True, False], p=[0.3, 0.7]) == True:
            null_lower_bound = 0.0
            null_upper_bound = rng.uniform(0.1, 0.5)
            column_issues.append(IncreasedNulls(p=(null_lower_bound, null_upper_bound)))

    return (column_name, column_issues)


def generate_uniform_column(rng, dtype, sign, scale, shift, column_name):
    low = scale
    high = low + shift

    if sign == -1:
        low, high = high * sign, low * sign

    parameter_func = None
    if rng.choice([True, False]) == True:
        increase = shift * 0.1 * sign
        parameter_func = lambda scale, shift, l, u: (
            scale,
            shift + increase,
            l,
            u,
        )

    return UniformNumericColumn(
        column_name, low, high, dtype=dtype, parameter_function=parameter_func
    )


def generate_normal_column(rng, dtype, sign, scale, shift, column_name):
    mean = shift * sign
    std = scale

    parameter_func = None
    if rng.choice([True, False]) == True:
        increase = shift * 0.1 * sign
        parameter_func = lambda scale, shift, mu, sigma: (
            scale,
            shift + increase,
            mu,
            sigma,
        )

    return NormalNumericColumn(
        column_name, mean, std, dtype=dtype, parameter_function=parameter_func
    )


def generate_beta_column(rng, dtype, sign, scale, shift, column_name):
    alfa = rng.uniform(0.1, 10)
    beta = rng.uniform(0.1, 10)

    parameter_func = None
    if rng.choice([True, False]) == True:
        increase = shift * 0.1 * sign
        parameter_func = lambda scale, shift, alfa, beta: (
            scale,
            shift + increase,
            alfa,
            beta,
        )

    return BetaNumericColumn(
        column_name,
        alfa,
        beta,
        scale=scale,
        shift=shift * sign,
        dtype=dtype,
        parameter_function=parameter_func,
    )


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


def generate_column_history(
    n_col: int, n_hist: int, rng: np.random.Generator
) -> List[List[pd.DataFrame]]:

    # Each 'column' will comprised of 2 column pipeline,
    #   where first column will be the actual data column,
    #   while the other one will be a supporting column,
    #   used for data issue generation such as schema change
    column_pipelines = []
    column_pipeline_patterns = []
    column_pipeline_pattern_settings = []
    for i in range(n_col):

        col_name = f"numeric_{i}"
        col_neighbour_name = f"numeric_{i}_neighbour"

        col = generate_column(rng, col_name)
        col_issues = generate_column_issues(rng, col_name, dtype=col.dtype)

        neighbour_col = generate_column(rng, col_neighbour_name, dtype=col.dtype)
        neighbour_col_issues = generate_column_issues(rng, col_neighbour_name, dtype=col.dtype)

        column_pipeline = DataGenerationPipeline(
            columns=[col, neighbour_col],
            issues=[col_issues, neighbour_col_issues],
            random_state=rng,
        )

        pipeline_pattern = rng.choice(
            ["constant", "normal", "seasonal", "constant_growth", "periodic_growth"],
            p=[0.1, 0.2, 0.2, 0.25, 0.25],
        )
        pipeline_pattern_settings = {}
        if pipeline_pattern == "seasonal":
            pipeline_pattern_settings["period"] = rng.integers(1, 5)

        if pipeline_pattern == "constant_growth":
            pipeline_pattern_settings["increment"] = rng.uniform(100, 200)
            
        if pipeline_pattern == "periodic_growth":
            pipeline_pattern_settings["period"] = rng.integers(2, 8)
            pipeline_pattern_settings["increment"] = rng.uniform(100, 500)

        column_pipelines.append(column_pipeline)
        column_pipeline_patterns.append(pipeline_pattern)
        column_pipeline_pattern_settings.append(pipeline_pattern_settings)

    # generating the pieline outputs
    data = []
    for i, _ in enumerate(tqdm(range(n_hist), desc="Generating column executions...")):
        time_step = []
        for (
            column_pipeline, pattern, pattern_settings,
        ) in zip(column_pipelines, column_pipeline_patterns, column_pipeline_pattern_settings):
            if pattern == "constant":
                pipeline_execution = column_pipeline.generate(20000)

            elif pattern == "normal":
                pipeline_execution = column_pipeline.generate_normal(20000, 1000)

            elif pattern == "seasonal":
                normalised_unit = pattern_settings["period"] * (2 * np.pi) / n_hist
                y = np.cos(i * normalised_unit)
                pipeline_execution = column_pipeline.generate(int(y * 5000 + 25000))

            elif pattern == "constant_growth":
                increment = int(i * pattern_settings["increment"])
                pipeline_execution = column_pipeline.generate(20000 + increment)

            elif pattern == "periodic_growth":
                increment = int((i // pattern_settings["period"]) * pattern_settings["increment"])
                pipeline_execution = column_pipeline.generate(20000 + increment)

            time_step.append(pipeline_execution)
        data.append(time_step)

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
    n_hist = 60
    n_col = 30

    column_history = generate_column_history(n_col, n_hist, rng)
    column_perturbations = generate_column_perturbations(column_history, rng)

    benchmark_dir = pathlib.Path(__file__).parent
    with open(f"{benchmark_dir}/benchmark_data.pickle", "wb") as f:
        benchmark_data = {
            "column_history": column_history,
            "column_perturbations": column_perturbations
        }
        pickle.dump(benchmark_data, f)

    