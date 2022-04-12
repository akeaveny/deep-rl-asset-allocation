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

# data from csv files
AGGREGATE_DJIA_DATA_FILE = os.path.join(data_csv_dir, "DJIA_aggregated.csv")
AGGREGATE_DJIA_ADJ_CLOSE_FILE = os.path.join(data_csv_dir, "DJIA_adj_close.csv")
TRAINING_DATA_FILE = AGGREGATE_DJIA_DATA_FILE
TESTING_DATA_FILE = os.path.join(data_csv_dir, "daily", "^DJI.csv")

# data from pre-processing pipeline
PREPROCESSED_DATA_FILE = os.path.join(data_csv_dir, "preprocessed_djia_data.csv")

# Creating each directory for experiments, etc.
DIR_LIST = [
    data_dir,
    data_csv_dir,
    results_dir,
]
for DIR in DIR_LIST:
    if not os.path.exists(DIR):
        # make the directory
        os.makedirs(DIR)
        # print(f'Initialized directory: {DIR}')
