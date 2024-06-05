# Benchmark
Here you can find the code that was used to run the benchmark for implementation of the AVH algorithm.

The following baselines were used to compare the AVH algorithm:
* [Google TFDV](https://www.tensorflow.org/tfx/data_validation/get_started)
* [Two-sample Kolmogorov-Smirnov test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
* [Local Outlier Factor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
* [HealthESN](https://github.com/TimeEval/TimeEval-algorithms/tree/main/health_esn)

## Usage
To replicate the benchmark, use the following workflow:
1. Generate the synthetic benchmark data using the `benchmark_data_generation.py` script:
```bash
poetry run python benchmark_data_generation.py
```
2. Run any or every algorithm to generate their respective classification metrics:
```bash
poetry run python benchmark_avh.py
```
3. Run the final visualisation notebook `benchmark_visualisations.ipynb` to get the final precision-recall plots for all the run algorithms.
