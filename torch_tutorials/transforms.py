# transforms are manipulation of data that is suitable for training

# transform modifies the features
# target_transform modifies the labels

# FashionMNIST features are PIL image format, labels are integers
# for training, we need features as normalized tensors 
# and labels as one-hot encoded tensors

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data", 
    train=True, 
    download=True,
    transform=ToTensor(), 
    target_transform=Lambda(lambda y: torch.zeros(
        10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# ToTensor() converts a PIL image or numpy ndarray into a FloatTensor 
# and scales pixel intensity in the range [0., 1.]

# Lambda transforms applies any user-defined lambda function
# here, first create a zero tensor of size 10 and calls scatter_,
# which assigns a value=1 on the index given by label y
# dim specifies which dimension we are operating on, the one chosen is the one that varies

