import os
import torch
import torchmetrics
from torch.utils.data import DataLoader
import pandas as pd
from mnist_asl_letters_classification.models.asl_classifier_cnn_model import ASLAlphabetClassifier
from datasets.asl_mnist_dataset import ASLAlphabetMNISTDataset
from mnist_asl_letters_classification.utils.transforms import asl_mnist_test_transforms
from utils.training_utils import get_predictions

current_dir = os.path.dirname(os.path.abspath(__file__))
num_classes = 26

# Setup device agnostic code
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} device.")

# Hyperparameters
BATCH_SIZE = 16
NUM_WORKERS = torch.cuda.device_count() if device == torch.device("cuda") else 0

# Loading data
try:
  test_datapath = os.path.join(current_dir, "..", "data", "sign_mnist_test.csv")
  test_datapath = os.path.abspath(test_datapath)

  test_df = pd.read_csv(test_datapath)
except FileNotFoundError:
  print("Dataset not found. Please download the dataset and place it in the data directory.")
  exit()
  
test_dataset = ASLAlphabetMNISTDataset(test_df, transforms=asl_mnist_test_transforms)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

saved_model_path = os.path.abspath(os.path.join(current_dir, "..", "models", "checkpoints", "asl_mnist_classifier_cnn1.pth"))

model = ASLAlphabetClassifier(in_channels=1, num_classes=num_classes)
model.load_state_dict(torch.load(saved_model_path, map_location=device))
model.to(device)

y_true, y_preds = get_predictions(model, test_dataloader, device)

y_true = torch.tensor(y_true).to(device)
y_preds = torch.tensor(y_preds).to(device)

# F1-Score, accuracy, recall, precision
f1 = torchmetrics.functional.f1_score(y_preds, y_true, task="multiclass", num_classes=num_classes, average="weighted")
acc = torchmetrics.functional.accuracy(y_preds, y_true, task="multiclass", num_classes=num_classes, average="weighted")
recall = torchmetrics.functional.recall(y_preds, y_true, task="multiclass", num_classes=num_classes, average="weighted")
precision = torchmetrics.functional.precision(y_preds, y_true, task="multiclass", num_classes=num_classes, average="weighted")

print(f"F1-Score: {f1:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")