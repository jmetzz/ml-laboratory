# Repository for hummingbird evaluation

Hummingbird compiles trained ML models into tensor computation for faster inference.

See https://github.com/microsoft/hummingbird


## Setup 

Clone the repository and go into its root directory.

Create a python environment with the necessary dependencies.

```
make interpreter deps
```

> If you want to know all the available make tasks, run `make help`.


## Example notebook

To run the example notebook, start a `jupiter lab` instance:

```
cd $PROJECT_HOME
poetry run jupyter lab
```
