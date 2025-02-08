from torch.utils.data import Dataset
import pandas as pd
import os
import sys

data_utils_path = os.path.abspath(os.path.join(os.getcwd(), "utils"))
sys.path.append(data_utils_path)

import data_utils as du


class ASLAlphabetMNISTDataset(Dataset):
  def __init__(self, dataframe: pd.DataFrame, transforms=None):
    super(ASLAlphabetMNISTDataset, self).__init__()

    self.df = dataframe
    self.transforms = transforms

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    image_data = self.df.iloc[idx].values
    label = image_data[0]
    pixels = image_data[1:]

    image = du.pixel_values_to_image(pixels)

    if self.transforms:
      image = self.transforms(image)

    return image, label