ml-laboratory
==============================

My playground for machine learning tests.

> Be aware that most of the code is not complete and might not work as expected.


Preparing the environment
-------------------------

Conda is the preferred environment manager, but if not available on you system 
`virtualenv` and `virtualenvwrapper` will be used.

To install `miniconda` go to [this link](https://conda.io/en/master/miniconda.html)

Assuming the url is still valid, you can also run:

```bash
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
$ bash ~/miniconda.sh -p $HOME/miniconda

# Check if the path is correctly set. Otherwise, set it on the user profile
# export PATH="$HOME/miniconda/bin:$PATH"
```


> NOTE: specific data science packages used in this project are listed in the `environment.yml` file only. 
> In case you are not using `conda` to manage the runtime environment, make sure you include the necessary 
> dependencies in the `requirements.txt` file so they get installed by pip.


Executing the code
------------------

Check `Makefile` commands to create environment and run the code.

You can use `jupyter notebook` to run the notebooks available in the project. 
Make sure you activate the correct environment first and the execute the notebook:

```bash
$ conda env list
base                  *  /usr/local/miniconda3
ml-laboratory            /usr/local/miniconda3/envs/ml-laboratory
$
$ source activate ml-laboratory
$ cd path/to/the/project
$ jupyter notebook
```


Project Organization
--------------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── main           <- Holds python code implementing other business logic.
    │   │   │                 Added to comply with PyBuilder structure.
    │   │   └── python
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>



