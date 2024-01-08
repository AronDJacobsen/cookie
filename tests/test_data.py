import torch
from torchvision import datasets, transforms
import os
import pytest

from cookie.data.dataloader import get_dataloaders  

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data

# we can check if the data files exist
processed_data_path = _PATH_DATA + "/processed/processed_data.pt"  # Replace with the actual path to your processed data file

@pytest.mark.skipif(not os.path.exists(processed_data_path), reason="Data files not found")
def test_data():
    trainloader, valloader = get_dataloaders(processed_data_path)

    # Check if the length of the training set is correct
    assert len(trainloader.dataset) + len(valloader.dataset) == 70000, "The length of the training set is incorrect"

    # Check the shape of each datapoint in the training set
    for data in trainloader.dataset:
        assert data[0].shape == torch.Size([1, 28, 28]) or data[0].shape == torch.Size([784]), "The shape of the training set is incorrect"

    # Check the shape of each datapoint in the validation set
    for data in valloader.dataset:
        assert data[0].shape == torch.Size([1, 28, 28]) or data[0].shape == torch.Size([784]), "The shape of the validation set is incorrect"



if __name__ == "__main__":
    test_data()