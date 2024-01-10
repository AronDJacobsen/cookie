import os

import torch
from torchvision import datasets, transforms


def get_dataloaders(processed_data_path):
    if not os.path.exists(processed_data_path):
        raise Exception("Processed data not found at {}".format(processed_data_path))

    processed_data = torch.load(processed_data_path)

    # Separate images and labels
    processed_images, processed_labels = zip(*processed_data)

    # Convert processed data back to a PyTorch dataset
    processed_dataset = torch.utils.data.TensorDataset(torch.cat(processed_images), torch.cat(processed_labels))

    train_ratio = 0.8
    num_samples = len(processed_dataset)
    num_train = int(train_ratio * num_samples)
    num_val = num_samples - num_train

    # Use random_split to get a list of datasets
    train_data, val_data = torch.utils.data.random_split(processed_dataset, [num_train, num_val])

    batch_size = 64
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return trainloader, valloader
