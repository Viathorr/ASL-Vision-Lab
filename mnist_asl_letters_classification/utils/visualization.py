import numpy as np
import pandas as pd
from PIL import Image  # Python Imaging Library for image manipulation
import matplotlib
import matplotlib.pyplot as plt  # Plotting library for visualization

import torch
import wandb

import mnist_asl_letters_classification.utils.image_processing as ip


# Create a dictionary mapping numbers (0-25) to alphabet letters (A-Z) (for ASL MNIST dataset)
label_to_letter = {i: chr(65 + i) for i in range(26)}


def display_mnist_images_from_df(df: pd.DataFrame, image_num: int = 8) -> matplotlib.figure.Figure:
  """
  Display a grid of MNIST images from a DataFrame.

  Args:
    df (pandas.DataFrame): The DataFrame containing the MNIST images.
    image_num (int): The number of images to display. Defaults to 8.

  Returns:
    matplotlib.figure.Figure: The generated figure.
  """
  n_rows = int(np.ceil(image_num / 4))
  fig, axes = plt.subplots(n_rows, 4, figsize=(12, 4 * n_rows))
  axes = axes.flatten()
    
  idxs = np.random.choice(df.shape[0], image_num, replace=False)

  for i, idx in enumerate(idxs):
    image_data = df.iloc[idx].values  # numpy array
    image = ip.pixel_values_to_image(image_data[1:])

    axes[i].imshow(image, cmap="gray")
    axes[i].set_title(f"Letter '{label_to_letter[image_data[0]]}'")
    axes[i].axis("off")

  plt.tight_layout()
  plt.show()
  
  return fig
    

def visualize_asl_mnist_preds(model: torch.nn.Module, test_dataloader: torch.utils.data.DataLoader, device: str, n_images: int = 8, wandb_run: wandb.sdk.wandb_run.Run = None, log_preds: bool = False) -> matplotlib.figure.Figure:
  """
  Visualize predictions for ASL MNIST dataset.
  
  Args:
    model (torch.nn.Module): The model to use for prediction.
    test_dataloader (torch.utils.data.DataLoader): The test dataloader.
    device (str): The device to use for inference.
    n_images (int, optional): The number of images to visualize. Defaults to 8.
    wandb_run (wandb.sdk.wandb_run.Run, optional): The Weights & Biases run to log to. Defaults to None.
    log_preds (bool, optional): Whether to log the predictions to Weights & Biases. Defaults to False.

  Returns:
    matplotlib.figure.Figure: The generated figure.
  """
  torch.manual_seed(42)
  images, labels = next(iter(test_dataloader))
  
  model.to(device).eval()
  with torch.inference_mode():
    y_pred_logits = model(images.to(device))
    y_preds = torch.argmax(y_pred_logits, dim=1).cpu()

    correct = [y_preds[i] == labels[i] for i in range(n_images)]

  mean = 0.1307
  std = 0.3081
  images = torch.clip((images * std + mean) * 255, 0, 255).type(torch.uint8)

  if log_preds:
    wandb_run.log({"predictions": [wandb.Image(img.squeeze(0).numpy(), caption=f"Pred: {p.item()} ('{label_to_letter[p.item()]}'), True Label: {l.item()} ('{label_to_letter[l.item()]}')") for img, p, l in zip(images[:10], y_preds.cpu()[:10], labels[:10])]})

  n_rows = (n_images + 3) // 4
  fig, axes = plt.subplots(n_rows, 4, figsize=(16, n_rows * 4))
  axes = axes.flatten()

  for i in range(n_images):
    axes[i].imshow(images[i].squeeze(0), cmap="gray")
    axes[i].set_title(f"p='{label_to_letter[y_preds[i].item()]}'{f' => true=`{label_to_letter[labels[i].item()]}`' if not correct else ''}", color="g" if correct[i] else "r")
    axes[i].axis("off")

  for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

  plt.show()
  
  return fig
