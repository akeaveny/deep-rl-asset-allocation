import os
import shutil

import numpy as np
import torch
from deep_rl_asset_allocation.configs import (data_config, env_config, paths_config)


def set_random_seed(seed=env_config.SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def init_directories_for_training():
    # Removing these directories to clear past results
    DIR_LIST = [
        paths_config.results_trained_models_dir,
        paths_config.results_figs_dir,
        paths_config.results_csv_dir,
        paths_config.results_tensorboard_dir,
    ]
    for DIR in DIR_LIST:
        if os.path.exists(DIR):
            shutil.rmtree(DIR)
            print(f'Removed directory: {DIR}')

    # Creating each directory for experiments, etc.
    DIR_LIST = [
        paths_config.results_trained_models_dir,
        paths_config.results_figs_dir,
        paths_config.results_csv_dir,
        paths_config.results_tensorboard_dir,
    ]
    SUB_DIR_LIST = [
        "test",
        "train",
        "val",
    ]
    for DIR in DIR_LIST:
        if not os.path.exists(DIR):
            # make the directory
            os.makedirs(DIR)
            print(f'Initialized directory: {DIR}')
            # making sub directories for train, val and test results
            if DIR == paths_config.results_figs_dir or DIR == paths_config.results_csv_dir:
                for SUB_DIR in SUB_DIR_LIST:
                    _SUB_DIR = os.path.join(DIR, SUB_DIR)
                    os.makedirs(_SUB_DIR)
                    print(f'\tInitialized subdirectory: {_SUB_DIR}')
