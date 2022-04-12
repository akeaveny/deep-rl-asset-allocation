import datetime

from deep_rl_asset_allocation.configs import data_config
from deep_rl_asset_allocation.utils import data_loader_utils

djia_data_dict = data_loader_utils.get_train_val_test_djia_data()
