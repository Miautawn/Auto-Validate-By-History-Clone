* [Auto-Validate-By-History](#auto-validate-by-history)
* [Setup](#setup)
* [Usage](#usage)

# Auto-Validate-By-History
AVH (auto-validate-by-history) is a data quality validation method first described in the paper **Auto-Validate by-History: Auto-Program Data Quality Constraints to Validate Recurring Data Pipelines** (Dezhan Tu et al., [2023](https://arxiv.org/abs/2306.02421)).

The authors provide an official repository for their version of the implementation, however, at the time of writing it is empty, which urged us to create our own.
* https://github.com/River12/Auto-Validate-by-History

# Features
The package tries to implement all the features described in the paper, however, some are still missing at this point & will be implemented with future releases!
## Unsupported features
- [x] Stationarity testing (currently only stationary data is supported).
- [ ] Categorical data issues & metrics (currently only numerical data is fully supported).

# Setup
## Installation
The package is published on pypi index, so you can simply download it from there:
```bash
pip install miautawn-auto-validate-by-history
```

## Build from source
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

By this stage you should be able to use the package and run the notebook examples!
However, to finally build the package, simply run the following command which will generate both source distribution `.tar.gz` and built distribution `.whl` files.
```bash
poetry build
```

# Usage
Below is an elamentary example on how to use the provided tools:
```python
from avh.data_generation import DataGenerationPipeline, NormalNumericColumn, BetaNumericColumn

from avh.data_issues import IncreasedNulls
from avh.auto_validate_by_history import AVH

# To begin with, we nned to collect a history of data in a form of list of dataframes.

# It's pretty easy to model virtual data tables with our provided classes like so:
#   * columns - what colums and what type should make up the table?
#   * issues - should the natural data have any quirks?
pipeline = DataGenerationPipeline(
    columns=[
        NormalNumericColumn("money", 1000, 50),
        NormalNumericColumn("height", 180, 10),
    ],
    issues=[
        ("money", [IncreasedNulls(0.05)]),
    ],
    random_state=42
)

# In this case, we'll generate 30 "data pipeline" executions of size ~ N(10000, 30)
H = [pipeline.generate_normal(10000, 30) for i in range(30)]

# Finally, let's see what data quality constraints does the AVH generate
#   for the 'money' column of our data:
PS = AVH(columns=["money"], random_state=42).generate(H, fpr_target=0.05)

ps_money = PS["money"]
>> CLTConstraint(0.9495 <= CompleteRatio <= 0.9515, FPR = 0.0005), FPR = 0.045966

# As you can see, the AVH algorithm correctly identifies a statistical invariate
#   which in our case was data completeness ratio of 95% (as specified in our data generation).
#
#   To the algorithm, any deviation outside this narrow metric interval would appear as an
#   anomaly and thus would trigger the generated data quality constraint.
new_data = pipeline.generate_normal(1000, 30)
new_data_w_issues = IncreasedNulls(p=0.5).fit_transform(new_data)

ps_money.predict(new_data["money"])
>> True     # the constraint holds
ps_money.predict(new_data_w_issues["money"])
>> False    # the constraint doesn't hold

# Naturally, in a real-world scenario you'd like to have as complete data as possible.
# The library allows the user to make necessary adjustments if one wishes to do so:
ps_money.constraints[0].u_upper_ = 1.0
ps_money
>> CLTConstraint(0.9495 <= CompleteRatio <= 1.0, FPR = 0.0005), FPR = 0.045966
```
