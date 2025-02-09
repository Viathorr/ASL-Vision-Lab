import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from utils.transforms import asl_mnist_train_transforms, asl_mnist_test_transforms
from utils.training_utils import train_model
from utils.metrics import accuracy
from mnist_asl_letters_classification.models.asl_classifier_cnn_model import ASLAlphabetClassifier
from datasets.asl_mnist_dataset import ASLAlphabetMNISTDataset
import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device.")

# Hyperparameters
BATCH_SIZE = 16
NUM_WORKERS = torch.cuda.device_count() if device == "cuda" else 0
LR = 5e-2 
EPOCHS = 30

# Load the data
try:
  train_datapath = os.path.join(current_dir, "..", "data", "sign_mnist_train.csv") 
  train_datapath = os.path.abspath(train_datapath)
  
  test_datapath = os.path.join(current_dir, "..", "data", "sign_mnist_test.csv")
  test_datapath = os.path.abspath(test_datapath)
  
  train_df = pd.read_csv(train_datapath)
  test_df = pd.read_csv(test_datapath)
except FileNotFoundError:
  print("Dataset not found. Please download the dataset and place it in the data directory.")
  exit()

train_dataset = ASLAlphabetMNISTDataset(train_df, transforms=asl_mnist_train_transforms)
test_dataset = ASLAlphabetMNISTDataset(test_df, transforms=asl_mnist_test_transforms)

print(f"Dataset sizes: train = {len(train_dataset)}, test = {len(test_dataset)}") 

# Dataloaders
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"Number of batches in train dataloader: {len(train_dataloader)}, test dataloader: {len(test_dataloader)}")

# Model
num_classes = 26
model = ASLAlphabetClassifier(in_channels=1, num_classes=num_classes).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

# Loss function
criterion = nn.CrossEntropyLoss()

# Training
start_time = time.time()

results = train_model(model, train_dataloader, test_dataloader, criterion, accuracy, num_classes, optimizer, num_epochs=EPOCHS, lr_scheduler=lr_scheduler, device=device)

end_time = time.time()
training_time = end_time - start_time

print(f"Total training time: {training_time:.2f} seconds")


# Save the model
save_model_path = os.path.abspath(os.path.join(current_dir, "..", "models", "checkpoints", "asl_mnist_classifier_cnn1.pth"))

torch.save(model.state_dict(), save_model_path)