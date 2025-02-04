import os
import kaggle
import zipfile

def download_kaggle_dataset(dataset_name: str, data_dir: str, unzip: bool = False):
  """
  Downloads a dataset from Kaggle and saves it to a specified directory.

  Args:
    dataset_name (str): The name of the dataset to download.
    data_dir (str): The directory to save the dataset to.
    unzip (bool, optional): Whether to automatically unzip the downloaded dataset. Defaults to False.

  Raises:
    RuntimeError: If the Kaggle API key has not been set up.
  """
  # Make sure to have your Kaggle API key saved in your home directory
  # as ~/.kaggle/kaggle.json or C:\Users\<username>\.kaggle\kaggle.json
  kaggle.api.authenticate()
  
  os.makedirs(data_dir, exist_ok=True)

  print(f"Downloading dataset: `{dataset_name}` ...") 
  kaggle.api.dataset_download_files(dataset_name, path=data_dir, unzip=unzip)
  
  print("✅ Dataset downloaded successfully.")
      
      
def unzip_dataset(data_dir: str):
  """
  Unzips a dataset saved in a directory.

  Args:
    data_dir (str): The directory containing the dataset to unzip.
  """
  for file in os.listdir(data_dir):
    if file.endswith(".zip"):
      with zipfile.ZipFile(os.path.join(data_dir, file), "r") as zip_ref:
        zip_ref.extractall(data_dir) 
        
  print("✅ Dataset unzipped successfully.")
  
  