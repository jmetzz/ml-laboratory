# Pandas Profiling Lab

Trying out the pandas profiling library

## Project Organization

Differently from other data science template projects, I deliberately removed `data`, `models`, `reports`  directories.
I don't think these artefacts should be in the code repository, but rather in dedicated object stores.

Here is a sneak peek of the directory structure you will find in the generated project:

```
├── README.md           <- The top-level README for developers using this project.
│
├── config/             <- A place for configuration files necessary to run the code.
│
├── notebooks/          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
|
├── scripts/            <- A place any kind of utility scripts that are not part of the main
|                         code.
│
├── src/                <- Source code for use in this project.
│   └── cli.py          <- A `typer` app you can use as example.
│
├── tests/              <- Source code for use in this project.
│   └── test_cli.py     <- A unit test example that tests the execution of the s`rc/cli.py` module.
├── pyproject.toml      <- The global configuration file and where you should specify the dependencies
|                          for this project. This allows for easy reproducibility of your experiments,
|                          and working in collaboration with your team.
├── Makefile            <- Makefile with utility commands like `make deps` or `make lint`.
|                          Run `make help` to see all options available.
└── LICENSE
```

## Preparing the environment


> NOTE: `poetry` is used to manage the dependencies for this project.

To create the python environment and install the project dependencies,
enter the directory you created the project and run

```bash
make deps
```

Apart from the creations of the python environment, `make deps` will also initialize this project as a `git` repository
and setup `pre-commit` hooks.

Run `make help` to see other make tasks available.


## Executing the code


Make sure you activate the correct environment before executing any script/notebook.

```bash
poetry shell
```

Alternatively, run everything prefixing with `poetry run`.
