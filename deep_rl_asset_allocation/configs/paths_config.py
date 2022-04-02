"""Paths for dirs and files used throughout this repo"""

import os
import shutil
from pathlib import Path

# init dir
parent_dir = str(Path(__file__).parents[1])

data_dir = os.path.join(parent_dir, 'data')
data_csv_dir = os.path.join(data_dir, 'csv')

results_dir = os.path.join(parent_dir, 'results')
results_trained_models_dir = os.path.join(results_dir, 'trained_models')
results_figs_dir = os.path.join(results_dir, 'figs')
results_csv_dir = os.path.join(results_dir, 'csv')
results_tensorboard_dir = os.path.join(results_dir, 'tensorboard')

# Removing these directories to clear past results
DIR_LIST = [
    results_tensorboard_dir,
]
for DIR in DIR_LIST:
    if os.path.exists(DIR):
        shutil.rmtree(DIR)
        # print(f'Removed directory: {DIR}')

# Ensuring these directories exist
DIR_LIST = [
    data_dir,
    data_csv_dir,
    results_dir,
    results_trained_models_dir,
    results_figs_dir,
    results_csv_dir,
    results_tensorboard_dir,
]
for DIR in DIR_LIST:
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    # print(f'Initialized directory: {DIR}')

# data from csv files
TRAINING_DATA_FILE = os.path.join(data_csv_dir, "^DJI.csv")
# TRAINING_DATA_FILE = os.path.join(data_csv_dir, "dow_30_2009_2020.csv")
TESTING_DATA_FILE = os.path.join(data_csv_dir, "DJIA.csv")

# data from pre-processing pipeline
PREPROCESSED_DATA_FILE = os.path.join(data_csv_dir, "preprocessed_djia_data.csv")
