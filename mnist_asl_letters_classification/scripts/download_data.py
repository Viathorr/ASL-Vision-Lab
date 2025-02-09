import os
import utils.data_utils as du


du.download_kaggle_dataset(
  dataset_name="datamunge/sign-language-mnist",
  data_dir=os.path.join(os.getcwd(), "mnist-asl-letters-classification", "data"),
  unzip=True
)