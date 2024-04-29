# MNIST Handwritten Digit Recognition with PyTorch MLP

This project demonstrates the implementation of a Multi-Layer Perceptron (MLP) neural network for handwritten digit recognition using the MNIST dataset and PyTorch.

## Project Overview

The goal of this project is to train an MLP model to accurately classify handwritten digits from the MNIST dataset. The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels.

The MLP model architecture consists of an input layer, two hidden layers with ReLU activation, and an output layer with softmax activation. The model is trained using the cross-entropy loss function and optimized using the Adam optimizer.

## Requirements

To run this project, you need the following dependencies:

- Python (version 3.6 or higher)
- PyTorch (version 1.9.0 or higher)
- torchvision (version 0.10.0 or higher)
- NumPy (version 1.21.0 or higher)
- Matplotlib (version 3.4.2 or higher)
- ONNX (version 1.10.0 or higher)
- Netron (version 5.4.0 or higher) or other visualization tools

You can install the required Python packages using pip:
pip install torch torchvision numpy matplotlib onnx netron

## Dataset

The MNIST dataset is automatically downloaded using PyTorch's `torchvision.datasets.MNIST` class. The dataset is split into a training set (60,000 images) and a test set (10,000 images). The images are normalized and transformed into PyTorch tensors.

## Model Architecture

The MLP model architecture is defined in the `MNISTClassifier` class, which inherits from `nn.Module`. The model consists of the following layers:

- Input layer: 784 nodes (28x28 flattened image)
- Hidden layer 1: 512 nodes with ReLU activation
- Hidden layer 2: 256 nodes with ReLU activation
- Output layer: 10 nodes with softmax activation

## Training

The model is trained using the training set for a specified number of epochs. The training loop iterates over the training data, performs forward and backward passes, and updates the model parameters using the Adam optimizer. The training progress is displayed, showing the average loss per epoch.

## Evaluation

After training, the model is evaluated on the test set. The accuracy of the model's predictions on the test set is calculated and displayed.

## Visualization

The project includes visualization of the model's architecture and predictions using ONNX and visualization tools like Netron. The trained model is exported to the ONNX format, which can be visualized using Netron or other compatible tools.

## Usage

1. Clone the repository:

git clone https://github.com/your-username/mnist-mlp-pytorch.git

2. Install the required dependencies.

3. Run the `train.py` script to train the MLP model:

python train.py

4. The trained model will be saved as `mnist_mlp.pth`, and the ONNX model will be exported as `mnist_mlp.onnx`.

5. Visualize the model architecture using Netron or other visualization tools by opening the `mnist_mlp.onnx` file.

## Results

The trained MLP model achieves an accuracy of over 95% on the MNIST test set. The visualization of the model architecture provides insights into the network structure and the flow of data through the layers.

## License

This project is licensed under the [MIT License](LICENSE).

