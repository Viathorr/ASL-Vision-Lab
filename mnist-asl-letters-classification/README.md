# ASL Letters Classification with PyTorch and Sing Language MNIST Dataset (achieving $99\%$ Accuracy)

### The American Sign Language letter database of hand gestures represent a multi-class problem with 24 classes of letters (excluding J and Z which require motion).

![image.png](data\american_sign_language.PNG)

## Dataset

- **Source:** ([Kaggle link](https://www.kaggle.com/datasets/datamunge/sign-language-mnist))
- **Format:** Similar to the classic MNIST dataset
- **Classes:** 24 ASL letters (A-Y, excluding J and Z)
- **Training set:** 27,455 samples
- **Test set:** 7,172 samples
- **Image resolution:** 28x28 pixels, grayscale
- **Labels:** Represented as integers (0-25), corresponding to the letters A-Y

## Model Architecture

The CNN model used for classification consists of **three convolutional layers** with **_batch normalization_**, **_dropout_** for regularization, and **_global average pooling_** for feature aggregation.

## Tracking with Weights & Biases (WandB)

To monitor training metrics, visualize performance, and compare experiments, Weights & Biases was integrated into the training pipeline. [**Link**](https://wandb.ai/viathorrr/ASL%20Alphabet%20Classification%20with%20PyTorch?nw=nwuserviathorr)
