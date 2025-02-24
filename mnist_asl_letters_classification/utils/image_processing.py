from PIL import Image
import torchvision.transforms as transforms
import io
import numpy as np
from utils.transforms import asl_mnist_inference_transforms

def preprocess_image(image_bytes):
  """
  Preprocess an image byte stream for inference.

  Args:
    image_bytes: an image byte stream

  Returns:
    a preprocessed 3D tensor of size (1, 28, 28)
  """
  image = Image.open(io.BytesIO(image_bytes))
  image = asl_mnist_inference_transforms(image).unsqueeze(0)  
  return image


# Defining a function to convert pixel values to images
def pixel_values_to_image(pixel_values: list) -> Image:
  """
  Convert a row of 784 pixel values into a 28x28 grayscale image.

  Args:
    pixel_values (list or numpy array): A row of 784 pixel values (0-255).

  Returns:
    PIL.Image: The generated image.
  """
  # Ensure the input is a numpy array
  pixel_values = np.array(pixel_values, dtype=np.uint8)

  image_array = pixel_values.reshape(28, 28)
  
  image = Image.fromarray(image_array, mode='L')  # 'L' mode for grayscale

  return image