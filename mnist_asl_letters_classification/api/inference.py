import os
import torch
from models.asl_classifier_cnn_model import ASLAlphabetClassifier
from utils.image_processing import preprocess_image


current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(current_dir, "..", "models", "checkpoints", "asl_mnist_classifier_cnn1.pth"))

model = ASLAlphabetClassifier(in_channels=1, num_classes=26)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

class_names = [chr(i) for i in range(65, 91)]


def predict(image):
  """
  Predicts the class of a given image using a pre-trained ASL alphabet classifier.

  Args:
    image (PIL.Image or numpy.ndarray): The input image to be classified.

  Returns:
    tuple: A tuple containing:
      - predicted_class (int): The index of the predicted class.
      - class_name (str): The corresponding class name from 'A' to 'Z', excluding 'J' and 'Z'.
      - predicted_prob (float): The predicted probability of the predicted class.
  """
  preprocessed_image = preprocess_image(image)
  
  with torch.inference_mode():
    output = model(preprocessed_image)
    y_probs = torch.softmax(output, dim=1)
    predicted_prob, predicted_class = torch.max(y_probs, dim=1)
    
    return predicted_class.item(), class_names[predicted_class.item()], predicted_prob.item()
