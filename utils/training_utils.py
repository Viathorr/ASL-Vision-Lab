import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm


def train_step(model: nn.Module,
              train_dataloader: DataLoader,
              criterion: nn.Module,
              acc_fn: callable,
              num_classes: int,
              optimizer: torch.optim.Optimizer,
              device: torch.device,
              run: wandb.sdk.wandb_run.Run = None) -> tuple[float, float]:
  """
  Performs a single training step on the multiclass classification model.

  Args:
    model (nn.Module): The model to be trained.
    train_dataloader (DataLoader): The DataLoader for the training set.
    criterion (nn.Module): The loss function to use.  
    acc_fn (callable): The accuracy function to use.
    num_classes (int): The number of classes in the dataset.
    optimizer (torch.optim.Optimizer): The optimizer to use.
    device (torch.device): The device to use (e.g. 'cpu' or 'cuda').
    run (wandb.sdk.wandb_run.Run, optional): The Weights and Biases run object. Defaults to None.

  Returns:
    tuple[float, float]: The average loss and accuracy over the training set.
  """
  train_loss, train_acc = 0, 0

  model.train()

  for X, y in train_dataloader:
    X, y = X.to(device), y.to(device)

    y_pred_logits = model(X)
    y_preds = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)

    loss = criterion(y_pred_logits, y)
    train_loss += loss.item()

    acc = acc_fn(y_preds, y, num_classes)
    train_acc += acc
  
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

  train_loss = train_loss / len(train_dataloader)
  train_acc = train_acc / len(train_dataloader)

  if run:
    run.log({"train_loss": train_loss, "train_accuracy": train_acc})
  
  return train_loss, train_acc


def val_step(model: nn.Module,
            val_dataloader: DataLoader,
            criterion: nn.Module,
            acc_fn: callable,
            num_classes: int,
            device: torch.device,
            run: wandb.sdk.wandb_run.Run = None) -> tuple[float, float]:
  """
  Performs a single validation step on the multiclass classification model. 

  Args:
    model (nn.Module): The model to be validated.
    val_dataloader (DataLoader): The DataLoader for the validation set.
    criterion (nn.Module): The loss function to use.
    acc_fn (callable): The accuracy function to use.
    num_classes (int): The number of classes in the dataset.
    device (torch.device): The device to use (e.g. 'cpu' or 'cuda').
    run (wandb.sdk.wandb_run.Run, optional): The Weights and Biases run object. Defaults to None.

  Returns:
    tuple[float, float]: The average loss and accuracy over the validation set.
  """
  val_loss, val_acc = 0, 0

  model.eval()

  with torch.inference_mode():
    for X, y in val_dataloader:
      X, y = X.to(device), y.to(device)

      y_pred_logits = model(X)
      y_preds = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)

      loss = criterion(y_pred_logits, y)
      val_loss += loss.item()

      acc = acc_fn(y_preds, y, num_classes)
      val_acc += acc

  val_loss = val_loss / len(val_dataloader)
  val_acc = val_acc / len(val_dataloader)

  if run:
    run.log({"val_loss": val_loss, "val_accuracy": val_acc})

  return val_loss, val_acc


def train_model(model: nn.Module,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              criterion: nn.Module,
              acc_fn: callable,
              num_classes: int,
              optimizer: torch.optim.Optimizer,
              num_epochs: int,
              device: torch.device,
              run: wandb.sdk.wandb_run.Run = None,
              lr_scheduler=None) -> dict:
  """
  Trains the given multiclass classification model.

  Args:
    model (nn.Module): The model to be trained.
    train_dataloader (DataLoader): The DataLoader for the training set.
    val_dataloader (DataLoader): The DataLoader for the validation set.
    criterion (nn.Module): The loss function to use.
    acc_fn (callable): The accuracy function to use.
    num_classes (int): The number of classes in the dataset.
    optimizer (torch.optim.Optimizer): The optimizer to use.
    num_epochs (int): The number of epochs to train for.
    device (torch.device): The device to use (e.g. 'cpu' or 'cuda').
    run (wandb.sdk.wandb_run.Run, optional): The Weights and Biases run object. Defaults to None.
    lr_scheduler (optional): The learning rate scheduler to use. Defaults to None.

  Returns:
    dict: A dictionary containing the training and validation loss and accuracy for each epoch.
  """
  results = {"train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []}

  for epoch in tqdm(range(num_epochs)):
    train_loss, train_acc = train_step(model, train_dataloader, criterion, acc_fn, num_classes, optimizer, device)
    val_loss, val_acc = val_step(model, val_dataloader, criterion, acc_fn, num_classes, device)

    if lr_scheduler: 
      if run: 
        current_lr = lr_scheduler.get_last_lr()[0] 
        run.log({"learning_rate": current_lr})

      lr_scheduler.step(val_acc)

    print(f"Epoch {epoch + 1}: | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["val_loss"].append(val_loss)
    results["val_acc"].append(val_acc)
  
  return results


def get_predictions(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
  """ 
  Get predictions for a given model and dataloader.

  Args:
    model (nn.Module): The model to use for prediction.
    dataloader (DataLoader): The dataloader to use for prediction.
    device (torch.device): The device to use for inference.

  Returns:
    tuple[np.ndarray, np.ndarray]: A tuple containing the true labels and predicted labels.
  """
  model.eval()
  all_preds, all_labels = [], []

  with torch.inference_mode():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)

      y_pred_logits = model(X)
      y_preds = torch.argmax(y_pred_logits, dim=1)

      all_preds.extend(y_preds.cpu().numpy())
      all_labels.extend(y.cpu().numpy())

  return np.array(all_labels), np.array(all_preds)