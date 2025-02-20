import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt  # Plotting library for visualization
from sklearn.metrics import confusion_matrix
import seaborn as sns

import torch


FIGS_DIR = os.path.abspath(os.path.join("..", "reports", "figures"))


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