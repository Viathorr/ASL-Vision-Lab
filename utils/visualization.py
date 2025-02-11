import os
import sys
import numpy as np
import pandas as pd
# import cv2 
from PIL import Image  # Python Imaging Library for image manipulation
import matplotlib
import matplotlib.pyplot as plt  # Plotting library for visualization
from sklearn.metrics import confusion_matrix
import seaborn as sns

import torch
import wandb

import utils.data_utils as du

FIGS_DIR = os.path.abspath(os.path.join("..", "reports", "figures"))

# Create a dictionary mapping numbers (0-25) to alphabet letters (A-Z) (for ASL MNIST dataset)
label_to_letter = {i: chr(65 + i) for i in range(26)}


def save_figure(fig: matplotlib.figure.Figure, filename: str, dirname: str = "general") -> None:
  """
  Saves a given figure to a specified folder.

  Args:
    fig (matplotlib.figure.Figure): Figure object to save.
    filename (str): Name of the file (without extension).
    dirname (str): Name of the directory to save the figure in. Defaults to "general".

  Returns:
    None: This function saves the figure to disk.
  """
  folder = os.path.join(FIGS_DIR, dirname)
  
  os.makedirs(folder, exist_ok=True)
  save_path = os.path.join(folder, f"{filename}.png")
  
  fig.savefig(save_path, bbox_inches='tight')
  
  print(f"✅ Figure saved.")
  
  
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
    image = du.pixel_values_to_image(image_data[1:])

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


def plot_loss_curves(results: dict) -> None:
  """
  Plot the loss and accuracy curves for training and validation sets.

  Args:
    results (dict): The dictionary containing the results from training.
  """
  loss = results["train_loss"]
  val_loss = results["val_loss"]

  acc = results["train_acc"]
  val_acc = results["val_acc"]

  epochs = range(len(results["train_loss"]))

  plt.figure(figsize=(15, 7))

  # Plot the loss
  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, label="train_loss")
  plt.plot(epochs, val_loss, label="val_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()

  # Plot the accuracy
  plt.subplot(1, 2, 2)
  plt.plot(epochs, acc, label="train_accuracy")
  plt.plot(epochs, val_acc, label="val_accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()
  
  
def plot_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, class_names: list) -> matplotlib.figure.Figure:
  """
  Plot the confusion matrix.

  Args:
    y_true (torch.Tensor): True labels.
    y_pred (torch.Tensor): Predicted labels.
    class_names (list): List of class names.
    
  Returns:
    matplotlib.figure.Figure: The generated figure.
  """
  cm = confusion_matrix(y_true, y_pred)

  fig = plt.figure(figsize=(12, 12))
  sns.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 14}, linewidths=1, square=True, cmap="coolwarm", xticklabels=class_names, yticklabels=class_names)

  plt.xlabel("Predicted Label", fontsize=14)
  plt.ylabel("True Label", fontsize=14)
  plt.title("Confusion Matrix", fontsize=14)

  plt.tight_layout()
  plt.show()
  
  return fig