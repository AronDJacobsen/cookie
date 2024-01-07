# A Cookie from CookieCutter

testing cookie cutter.
Template: https://github.com/SkafteNicki/mlops_template

Reproducibility: the scientific method

> Observe -> Question -> Hypotheses -> Experiment -> Conclude -> Result -> Observe -> ...


## Setup

#### Clone the repository and data

```bash
git clone https://github.com/AronDJacobsen/cookie.git
cd https://github.com/AronDJacobsen/cookie.git
dvc pull
```
*You need access to the google drive data folder*


#### Precompiled environment

```bash
conda env create -f environment.yml
```

#### Manual installation

Otherwise, run the following:

```bash
conda create -n mlops_project python=3.10
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Get the data

```bash
make data  # runs the make_dataset.py file, try it!
```


## Train the model

```bash
make train  # runs the make_dataset.py file, try it!
```


## Evaluate


#### Predict

```bash
python cookie/predict_model.py \
    models/trained_model.pth \
    data/example_images
```

#### Visualize

currently

```bash
python visualize_model.py
```





---

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── Cookiecutter test  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   ├── predict_model.py <- script for predicting from a model
│   └── visualize_model.py <- script for visualization from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

---

# My notes



### environment.yml

To save the current environment into a YAML file, you can use the `conda env export` command. Here's how you can do it:

**Export the environment to YAML:**

   Run the following command to export the current environment to a YAML file (e.g., `environment.yml`):

   ```bash
   conda env export > environment.yml
   ```

   This command exports the environment to a YAML file, and you can use the `>` operator to redirect the output to a file.



--

### requirements.txt

Exhaustive list based on environment:

`pip list --format=freeze > requirements.txt`

*remove end string from torch and torchvision and torch==2.0.1*


or minimalistic list based on directory:

`pip install pipreqs`

*Compatibility check: https://pytorch.org/get-started/previous-versions/*


### Ruff

```bash
ruff check .
```

```bash
ruff format .
```

### Working with git and dvc


Desired flow:
   ```bach
   dvc add -> git add -> git commit -> git tag -> dvc push -> git push.
   ```

Pushing
   ```bash
   dvc add data/ # identify/track changes from last commit
   git add data.dvc # stage in git, not necessary if auto-staging
   git commit -m "Add or update data using DVC"
   git tag -a "v1.0" -m "Release version 1.0" # to help navigate versions 
   dvc push # push dvc tracked data to remote storage
   git push # push the git commit and tag
   ```

If you've tagged a release in both Git and DVC:

1. To go back to a specific Git commit and DVC data version:
   ```bash
   git checkout <tag_name>
   dvc checkout
   ```

2. To go back to a specific DVC data version without changing your Git commit:
   ```bash
   dvc checkout -rev <tag_name>
   ```


### Docker

`--name {container_name}`

#### Training


Build image:
```bash
docker build -f dockerfiles/train_model.dockerfile . -t train_model:latest
```

Train:
```bash
docker run --name experiment1 \
   -v $(pwd)/data/processed:/data/processed/ \
   -v $(pwd)/models:/models/ \
   train_model:latest
```

Transfer files from container to local:
```bash
docker cp experiment1:/models/trained_model.pth $(pwd)/models/trained_model.pth
```


#### Predicting

Build image:
```bash
docker build -f dockerfiles/predict_model.dockerfile . -t predict_model:latest
```

Run prediction script:
```bash
docker run --name predict --rm \
   -v $(pwd)/data/example_images:/data/example_images \
   -v $(pwd)/models:/models \
   predict_model:latest \
   /models/trained_model.pth \
   /data/example_images
```

Transfer files:
```bash
docker cp experiment1:/reports/figures/predictions.txt $(pwd)/reports/figures/predictions.txt
```

### Sweep

wandb sweep --project fashion-mnist "./conf/sweep/lr_sweep.yaml" 

