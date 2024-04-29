MNIST Digit Recognition with PyTorch MLP
This repository contains a PyTorch implementation of a simple Multi-Layer Perceptron (MLP) neural network for the task of handwritten digit recognition on the famous MNIST dataset.
Description
The MNIST dataset is a widely used benchmark in the field of machine learning and computer vision, consisting of 70,000 grayscale images of handwritten digits (0-9). The goal of this project is to train a simple feed-forward neural network to accurately classify these handwritten digits.
The model implemented in this repository is a basic MLP with the following architecture:

Input Layer: 784 nodes (28 x 28 flattened image)
Hidden Layer 1: 512 nodes with ReLU activation
Hidden Layer 2: 256 nodes with ReLU activation
Output Layer: 10 nodes (one for each digit class) with softmax activation

The model is trained using the PyTorch library, which provides a flexible and efficient framework for building and training neural networks.
Usage

Clone the repository: git clone https://github.com/username/mnist-pytorch-mlp.git
Install the required dependencies: pip install -r requirements.txt
Run the train.py script to train the model: python train.py
The script will download the MNIST dataset, preprocess the data, train the model, and report the final test accuracy.

Contents

train.py: The main script to train the MLP model on the MNIST dataset.
model.py: Contains the definition of the MLP model architecture.
utils.py: Utility functions for data loading, preprocessing, and visualization.
requirements.txt: List of required Python packages and their versions.
README.md: This file, containing a brief description of the project and usage instructions.

Contributing
Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
License
This project is licensed under the MIT License.
Acknowledgments

The MNIST dataset was originally created by Yann LeCun and others at AT&T Labs.
This implementation is based on the PyTorch tutorial for the MNIST digit recognition task.