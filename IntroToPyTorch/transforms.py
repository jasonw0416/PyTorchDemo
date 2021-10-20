# Transforms

# data is not always in final processed form that is required for training macchine learning algorithms
# we use TRANSFORMMS to perform some manipulation of data and make it suitable for training

# All TorcchVision datasets have two parameters:
#   - transform: modify the features
#   - target_transform: modify the labels

# FashionMMNIST features are in PIL Image format, and the labels are integers
# for training, we need the features as normmalized tensors and the labels as one-hot encoded tensors.
# to makke these transformations, we use "ToTensor" and "Lambda"

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    # ToTensor()
    # ToTensor converts a PIL image or NumPy ndarray into a FloatTensor and scales the image's pixel intensity values in the range[0., 1.]  
    transform=ToTensor(),

    # Lambda Transforms
    # Lambda transforms apply any user-defined lambda function. <--?
    # Here, we define a function to turn the integer into a one-hot encoded tensor.
    # it first creates a tensor of size 10 (the number of labels in our dataset) and calls scatter_ which assigns a value=1 on the index as given by the label y.
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# my question is: why do you need to use ToTensor() if you have from_numpy() function?? maybe??
# maybe because that change of tensor is different fromm initializing the dataset


