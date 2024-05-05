* [Auto-Validate-By-History](#auto-validate-by-history)
* [Setup](#setup)
* [Usage](#usage)

# Auto-Validate-By-History
AVH (auto-validate-by-history) is a data quality validation method first described in the paper **Auto-Validate by-History: Auto-Program Data Quality Constraints to Validate Recurring Data Pipelines** (Dezhan Tu et al., [2023](https://arxiv.org/abs/2306.02421)).

The authors provide an official repository for their version of the implementation, however, at the time of writing it is empty, which urged us to create our own.
* https://github.com/River12/Auto-Validate-by-History

# Setup
This project uses [Poetry](https://python-poetry.org/docs/#installation) - python packaging and dependancy management tool.

Install poetry by running the following:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

I higly advise configuring poetry to create the virtual environments in the project directories itself, thus not cluterring your file system in random places:
```bash
poetry config virtualenvs.in-project true
```

Clone the repository and install the necessary dependancy + create virtual env:
```bash
clone https://github.com/Miautawn/Auto-Validate-By-History-Clone.git
cd Auto-Validate-By-History-Clone
poetry install
```

You should now be able to run all the example notebooks and create your own scripts!

# Usage
Below is an elamentary example on how to use the provided tools:
```python
# To begin with, we need to define our data quality metric space
#   that AVH algorithm will consider.
#
# In this case, to ilustrate the point, we'll be using only the RowCount and the CompleteRation
#   metrics that measure the number of records in the table and null value ratio accordingly.
M = [
    RowCount,
    CompleteRatio,
]

# Now, we have to define the constraint estimator space
#   which the AVH algorithm will consider.
#
# Check the documentation for available options,
#   but in the majority of cases these would work just fine.
E = [CLTConstraint, ChebyshevConstraint]

# Now you have to get some data from which to generate the data quality constraints from!
# It's pretty easy to model virtual data tables with our provided classes like so:
#   * columns - what colums and what type should make up the table
#   * issues - should the natural data have any quirks?
pipeline = DataGenerationPipeline(
    columns=[
        NormalNumericColumn("money", 1000, 50),
        NormalNumericColumn("height", 180, 10),
    ],
    issues=[
        ("money", [IncreasedNulls(0.05)]),
    ],
)

# Simulating the data pipeline, we have to generate several runs of this table,
#   thus getting a distribution of columns for AVH to work with.
#
# In this case, we'll generate 30 "data pipeline" executions of size ~ N(10000, 30)
H = [pipeline.generate_normal(10000, 30) for i in range(30)]

# Finally, let's see what data quality constraints does the AVH generate for our data:
PS = AVH(M, E, columns=["money"]).generate(H, fpr_target=0.01)
PS["money"]
>> CLTConstraint(0.9500 <= CompleteRatio <= 0.9500, FPR = 0.0077), FPR = 0.007661

# As you can see, the AVH algorithm while correct in saying that 95% of DataCompleteness
#   is a statistical invariate (as specified in our data generation) and anything outside it would
#   mean that data is anomalous from our previous 'training data', it does not take the context
#   of the metric into the account: in production, you would probably desire CompleteRatio metric to be
#   in the range of [95% - 100%], thus it is left to the user to make necessary adjustments.
```
