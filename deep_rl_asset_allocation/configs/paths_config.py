"""Paths for dirs and files used throughout this repo"""

import os
from pathlib import Path

# TODO: ensure directories exist
parent_dir = str(Path(__file__).parents[1])
data_dir = os.path.join(parent_dir, 'data')
data_csv_dir = os.path.join(data_dir, 'csv')
results_dir = os.path.join(parent_dir, 'results')
results_trained_models_dir = os.path.join(results_dir, 'trained_models')
results_figs_dir = os.path.join(results_dir, 'figs')
results_csv_dir = os.path.join(results_dir, 'csv')

# data from csv files
TRAINING_DATA_FILE = os.path.join(data_csv_dir, "dow_30_2009_2020.csv")
TESTING_DATA_FILE = os.path.join(data_csv_dir, "DJIA.csv")

# data from pre-processing pipeline
PREPROCESSED_DATA_FILE = os.path.join(data_csv_dir, "preprocessed_djia_data.csv")
