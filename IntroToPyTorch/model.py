# Build Model
# Build the Neural Network

# Neural networks comprise of layers/modules that performm operations on data.
# "torch.nn" namespace provides all the building blocks you need to build your own neural network.
# every module in PyTorch subclasses the "nn.MModule".
# neural network is a module that consists of other modules (layers).
# neural network is a nested structure of modules

# building neural network to classify images in the FashionMNIST dataset.

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Get Device for Training
# train model on a hardware accelerator like GPU, if it is available
# it checks if torch.cuda is available, if not: CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
# ours printed: Using cpu device

# Define the Class
# define neural network by subclassing nn.Module
# initialize the neural networks in __init__
# every nn.Module subclass immplements the operations on input data in the "forward" method
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),              # what is ReLU - Rectified Linear Unit Function ????
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# creating an instance of "NeuralNetwork", and move it to the "device", and print its structure.
# "device" is cpu in our case here
model = NeuralNetwork().to(device)
print(model)



# to use the model, we need to pass the input data into it.
# this executes the model's "forward" function along with other stuffs
# Never call model.forward() directly!!

# Calling the model on the input returns a 10-dimensional tensor with raw predicted values for each class.
# we get the prediction probabilities by passing it through an instance of "nn.Softmax" module.
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# Model Layers
# break down the layers in the FashionMNIST mmodel.
# To illustrate it, we will take a sample minibatch of 3 images of size 28*28 and see what happens to it as we pass it through the networkk.
input_image = torch.rand(3,28,28)
print(input_image.size())

# nn.Flatten
# we initialize the nn.Flatten layer to convert each 2D 28*28 image into a contiguous array of 784 pixel values
# (the minimbatcch dimension (at dim=0) is maintained).
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# nn.Linear
# the linear layer is a module that applies a linear transformation on the input using its stored weights and biases.
# wait, what weights and biases did I use on this???
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# nn.ReLU
# Non-linear ativations are what create the complex mappings between the model's inputs and outputs.
# They are applied after linear transformations to introduce 'nonlinearity', helping neural networks learn a wide variety of phenomena.
# In this model, we use nn.ReLU between our linear layers, but there's other activations to introduce non-linearity in your model.
# what is the "other activations"...????
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# nn.Sequential
# nn.Sequential - an ordered container of modules
# the data is passed through all the modules in the same order as defined. 
# you can use sequential containers to put together a quickk network like "seq_modules"
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

# nn.Softmax
# the last linear layer of the neural network returns logits - raw values in [-infty, infty] - which are passed to the "nn.Softmax" module
# the logits are scaled to values [0,1] representing the model's predicted probabilities for each class.
# "dim" parammeter indicates the dimension along which the values must sum to 1.
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)




# Model Parameters
# mmany layers inside a neural network are 'parammeterized', i.e. have associated weights and biases that are optimized during training.
# subclassing "nn.Module" automatically tracks all fields defined inside your model object,
# and makes all parameters using your model's "parameters()" or "named_parameters()" methods.

# In this exaple, we iterate over each parameter, and print its size and a preview of its values.
print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}\n")


# further reading
# torch.nn
# https://pytorch.org/docs/stable/nn.html

