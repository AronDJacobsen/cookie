import torch
from torchvision import datasets, transforms
from data.make_dataset import get_dataloaders  # Replace 'your_module_name' with the actual name of the module containing get_dataloaders





def test_data():
    processed_data_path = _PATH_DATA + "/processed/processed_data.pt"  # Replace with the actual path to your processed data file
    trainloader, valloader = get_dataloaders(processed_data_path)

    # Check if the length of the training set is correct
    assert len(trainloader.dataset) == 25000 or len(trainloader.dataset) == 40000

    # Check if the length of the validation set is correct
    assert len(valloader.dataset) == 5000

    # Check the shape of each datapoint in the training set
    for data in trainloader.dataset:
        assert data[0].shape == torch.Size([1, 28, 28]) or data[0].shape == torch.Size([784])

    # Check the shape of each datapoint in the validation set
    for data in valloader.dataset:
        assert data[0].shape == torch.Size([1, 28, 28]) or data[0].shape == torch.Size([784])

    # Check if all labels are represented in the training set
    unique_labels_train = torch.unique(torch.cat(trainloader.dataset.tensors[1]))
    assert unique_labels_train.shape[0] == len(set(trainloader.dataset.tensors[1]))

    # Check if all labels are represented in the validation set
    unique_labels_val = torch.unique(torch.cat(valloader.dataset.tensors[1]))
    assert unique_labels_val.shape[0] == len(set(valloader.dataset.tensors[1]))



if __name__ == "__main__":
    test_data()