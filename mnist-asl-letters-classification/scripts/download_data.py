import sys
import os

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
print(module_path)

sys.path.append(module_path)

import data_utils

data_utils.download_kaggle_dataset(
  dataset_name="datamunge/sign-language-mnist",
  data_dir=os.path.join(os.getcwd(), "data"),
  unzip=True
)