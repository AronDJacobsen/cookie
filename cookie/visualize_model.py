import os

import torch
from visualize.visualize import tsne_visualization

from data.dataloader import get_dataloaders
from models.model import load_checkpoint

if __name__ == "__main__":
    # load model
    model_name = "trained_model.pth"
    model = load_checkpoint(os.path.join("models", model_name))
    # load data
    processed_data_path = os.getcwd() + os.sep + "data/processed/processed_data.pt"
    trainloader, valloader = get_dataloaders(processed_data_path)

    # Specify the index of the layer from which to extract the intermediate representation
    intermediate_layer_index = 2  # Adjust this index based on your network architecture

    # Forward pass to get the intermediate representation
    val_data = []
    val_labels = []
    for images, labels in valloader:
        images.resize_(images.size()[0], 784)
        labels.resize_(labels.size()[0])
        intermediate_representation = model.extract_intermediate(images, intermediate_layer_index)
        val_data.extend(intermediate_representation)
        val_labels.extend(labels)
    # visualize
    # turn into numpy array
    val_data = torch.stack(val_data).detach().numpy()
    val_labels = torch.stack(val_labels).detach().numpy()

    class_labels = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle Boot",
    ]

    tsne_visualization(val_data, val_labels, class_labels)
