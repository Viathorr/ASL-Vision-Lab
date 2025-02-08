import torch
import torch.nn as nn
import torch.nn.functional as F


class ASLAlphabetClassifier(nn.Module):
  def __init__(self, in_channels, num_classes):
    super(ASLAlphabetClassifier, self).__init__()

    self.conv_block1 = nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding="same"),
      nn.BatchNorm2d(num_features=128)
    )
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.dropout1 = nn.Dropout(p=0.25)

    self.conv_block2 = nn.Sequential(
      nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding="same"),
      nn.BatchNorm2d(num_features=64)
    )
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.dropout2 = nn.Dropout(p=0.25)

    self.conv_block3 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding="valid"),
      nn.BatchNorm2d(num_features=32)
    )
    
    # Global Average Pooling
    self.gap = nn.AdaptiveAvgPool2d(output_size=1)  # (batch, 32, 1, 1) needs to be flattened\
    
    self.fc = nn.Linear(32, num_classes)

  def forward(self, x):
    x = self.dropout1(F.relu(self.pool1(self.conv_block1(x))))
    x = self.dropout2(F.relu(self.pool2(self.conv_block2(x))))
    x = F.relu(self.conv_block3(x))
    
    return self.fc(torch.flatten(self.gap(x), 1))