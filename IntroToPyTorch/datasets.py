# Datasets & DataLoaders

# dataset code is separated from our modeling code
# two primitives: torch.utils.data.DataLoader and torch.utils.data.Dataset that allow you to use pre-loaded datasets as well as your own
# Datasets stores the samples and their corresponding labels
# DataLoaders wraps an iterable around the Dataset to enable easy access to the samples
# preloaded dataset: Image Datasets, Text Datasets, and Audio Datasets <-- you can find them on the website
# FashionMNIST is a dataset with a lot of training materials
# Load FashionMNIST with following parameters:
#   - root: the path where the train/test data is stored
#   - train: specifies training or test dataset
#   - download=True: downloads the data from the internet if it's not available at root
#   - transform and target_transform: specify the feature and label transformations

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


# Iterating and Visualizing the Dataset

# index Datasets manually like a list: training_data[index]
labels_map = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover", 
    3: "Dress", 
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
for i in range (1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
# not sure what training_data and what this ^^^^ does...
# the things that are printed, are they one of the training data?
# or did computer generate it from their trained...?
# prob the former not latter


# Creating a Custom Dataset for your files
# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
# FashionMNIST immages are stored in a directory img_dir, and their labels are stored in a CSV file annotations_file.

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    # __init__
    # run once when instantiating Dataset object.
    # Initialize the directory contating images, annotations file, and both transforms
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # __len__
    # the __len__ function returns the number of sammples in our dataset
    def __len__(self):
        return len(self.img_labels)

    # __getitem__
    # loads and returns a sample from the dataset at the given index idx
    # Based on the index, it identifies the image’s location on disk, 
    # converts that to a tensor using read_image, 
    # retrieves the corresponding label from the csv data in self.img_labels, 
    # calls the transform functions on them (if applicable), 
    # and returns the tensor image and corresponding label in a tuple.

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



# Preparing your data for training with DataLoaders
# Dataset retrieves dataset's features and labels one sample at a time
# we want to pass samples in minibatches, reshuffulte the data at every epoch to reducce model overfitting and use Python's multiprocessing to speed up data retrieval

# DataLoader is an iterable that abstracts this complexity for us in an easy API

from torch.utils.data import DataLoader

# loading dataset into the DataLoader and interate through them
# after we iterate over all batches the data is shuffled for finergrained control over the data loading order
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Iterate through the DataLoader
# Each iteration returns a batch of train_features and train_labels

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

# further reading
# torch.utils.data API
# https://pytorch.org/docs/stable/data.html
