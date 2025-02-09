from torchvision import transforms


asl_mnist_train_transforms = transforms.Compose([
  transforms.RandomHorizontalFlip(),
  transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR, fill=128),
  transforms.ToTensor(), 
  transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # Normalize with MNIST mean and std
])

asl_mnist_test_transforms = transforms.Compose([
  transforms.ToTensor(), 
  transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])