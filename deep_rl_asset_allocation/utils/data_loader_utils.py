"""Function to load pre-processed df into train, val and test (unique) df"""

import datetime
import os

import deep_rl_asset_allocation.preprocessing.data_preprocessing as data_preprocessing
import pandas as pd
from deep_rl_asset_allocation.configs import data_config, paths_config


def load_preprocessed_djia_data(training_data_file: str = paths_config.TRAINING_DATA_FILE,
                                preprocessed_data_file: str = paths_config.PREPROCESSED_DATA_FILE) -> pd.DataFrame:

    # read and preprocess training data
    if os.path.exists(preprocessed_data_file):
        print(f'Found prevouisly saved pre-processed data: {preprocessed_data_file}')
        df = pd.read_csv(preprocessed_data_file)
        # Format date column
        df["Date"] = df.apply(data_preprocessing.convert_datadate_to_datetime, axis=1)
    else:
        print(f'Starting pre-processing pipeline ..')
        df = data_preprocessing.preprocess_djia_data(training_data_file)
        _save_preprocessed_dataset(preprocessed_data=df, filename=preprocessed_data_file)
        print(f'Saved pre-processed data to: {preprocessed_data_file}')
    return df


def _save_preprocessed_dataset(preprocessed_data: pd.DataFrame, filename: str):
    preprocessed_data.to_csv(filename, index=False)


def get_data_between_dates(df, start, end):
    """split the dataset into training or testing using date"""
    type(df.Date[0])
    data = df[(df.Date >= start) & (df.Date < end)]
    data = data.sort_values(['Date', 'Ticker'], ignore_index=True)
    # reset index based on date
    data.index = data.Date.factorize()[0]
    return data


def get_train_val_test_djia_data(
    training_data_file: str = paths_config.TRAINING_DATA_FILE,
    preprocessed_data_file: str = paths_config.PREPROCESSED_DATA_FILE,
    training_start: datetime.date = data_config.TRAINING_START,
    training_end: datetime.date = data_config.TRAINING_END,
    validation_start: datetime.date = data_config.VALIDATION_START,
    validation_end: datetime.date = data_config.VALIDATION_END,
    testing_start: datetime.date = data_config.TESTING_START,
    testing_end: datetime.date = data_config.TESTING_END,
) -> dict:

    # load df from csv file
    df = load_preprocessed_djia_data(training_data_file, preprocessed_data_file)

    # split data
    df_train = get_data_between_dates(df, training_start, training_end)
    df_val = get_data_between_dates(df, validation_start, validation_end)
    df_test = get_data_between_dates(df, testing_start, testing_end)

    # return as a dict
    djia_data = {
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
    }

    return djia_data
