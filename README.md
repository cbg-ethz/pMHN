[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![build](https://github.com/cbg-ethz/pMHN/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/cbg-ethz/pMHN/actions/workflows/test.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Personalised Mutual Hazard Networks

Principled probabilistic modelling with Mutual Hazard Networks.

  - **Documentation:** [https://cbg-ethz.github.io/pMHN](https://cbg-ethz.github.io/pMHN)
  - **Source code:** [https://github.com/cbg-ethz/pMHN](https://github.com/cbg-ethz/pMHN)
  - **Bug reports:** [https://github.com/cbg-ethz/pMHN/issues](https://github.com/cbg-ethz/pMHN/issues)

## Running the workflows

To facilitate reproducibility we use [Snakemake](https://snakemake.readthedocs.io/).
We recommend creating a new virtual environment (e.g., using [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)) and installing Snakemake as described in their documentation.

Once the environment is set, the package can be installed using
```bash
$ pip install -e .  # Note -e which will allow modifying the code when needed
```

## Contributing

We use [Poetry](https://python-poetry.org/) to control dependencies.

When Poetry is installed, clone the repository and type

```bash
$ poetry install --with dev
```

to install the package with the dependencies used for development.

At this stage you should be able to use [Pytest](https://docs.pytest.org/) to run unit tests:

```bash
$ poetry run pytest
```

Alternatively, you may want to work inside Poetry environment:
```bash
$ poetry shell
$ pytest
```

When you submit a pull request, automated continuous integration tests will be run.
They include unit tests as well as code quality checks.
To run the code quality checks automatically at each commit made, we use [pre-commit](https://pre-commit.com/).
To activate it run:

```bash
$ poetry shell  # If it is not already active
$ pre-commit install
```

