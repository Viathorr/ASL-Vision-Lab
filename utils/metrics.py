import numpy as np
import torch


def accuracy(y_preds: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> float:
  """
  Compute accuracy of predicted labels against true labels.

  Args:
    y_preds (torch.Tensor): Predicted labels of shape (*, num_classes) or (*, 1)
    y_true (torch.Tensor): True labels of shape (*, 1)
    num_classes (int): Number of classes

  Returns:
    float: Accuracy of predicted labels in percentage
  """
  if y_preds.shape != y_true.shape:
    if y_preds.shape[1] == num_classes:  # if raw logits are passed instead of predicted labels
      y_preds = torch.argmax(y_preds, dim=1, keepdim=True)
    else:
      raise ValueError(f"Shape mismatch: Expected `y_preds` of shape either {y_true.shape} or (*, {num_classes}), but got {y_preds.shape}")
      
  correct_preds = (y_preds == y_true).sum().item()
  
  return (correct_preds / y_true.numel()) * 100