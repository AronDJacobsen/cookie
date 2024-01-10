import argparse
import os
import random

import numpy as np
import torch
from PIL import Image
from rich.logging import RichHandler
from torchvision import transforms

import wandb
from models.model import load_checkpoint, predict_single_sample

"""
Run command:


python cookie/predict_model.py \
    models/trained_model.pth \
    data/example_images


"""


# Initialize wandb
wandb.init(project="fashion-mnist", name="predict1")


def load_images(data_path):
    # Load images from the given path
    if data_path.endswith(".npy"):
        images = np.load(data_path)
        img_names = [f"image_{i}" for i in range(len(images))]
    else:
        # Assume it's a folder with raw images
        # Modify this based on your actual data loading logic
        img_names = sorted(os.listdir(data_path))
        images = [Image.open(os.getcwd() + os.sep + data_path + os.sep + f) for f in sorted(os.listdir(data_path))]
        # Convert images to numpy array or torch tensor as needed

    # Define the transform for the test data (consistent with Fashion MNIST normalization)
    test_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    normalized_images = [test_transform(image).view(-1) for image in images]

    # web_testset = ImageFolder(root=os.getcwd() + os.sep + data_path, transform=transform)
    # web_testset = ImageFolder(root=os.getcwd() + os.sep + data_path, transform=test_transform)

    return img_names, normalized_images


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns:
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model
    """
    model.eval()  # Set the model to evaluation mode
    device = next(model.parameters()).device  # Get the device of the model

    predictions_list = []

    with torch.no_grad():
        for batch in dataloader:
            # Make sure the batch is a tensor
            batch = torch.stack(batch)

            # Move the batch to the device of the model
            batch = batch.to(device)

            # Perform prediction on the batch
            predictions_batch = model(batch)

            # Append the predictions to the list
            predictions_list.append(predictions_batch)

    # Concatenate the predictions along the specified dimension
    predictions = torch.cat(predictions_list, dim=0)

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using a pre-trained model")
    parser.add_argument(
        "model_path", type=str, nargs="?", help="Path to the pre-trained model file", default="models/trained_model.pth"
    )
    parser.add_argument(
        "data_path",
        type=str,
        nargs="?",
        help="Path to the input data file (e.g., folder or numpy file)",
        default="data/example_images",
    )
    args = parser.parse_args()

    # Load the pre-trained model
    model = load_checkpoint(args.model_path)

    # TODO: transform the input data
    img_names, images = load_images(args.data_path)
    # Convert the list of tensors to a single tensor
    all_images = torch.stack(images)

    # Convert to DataLoader
    dataset = torch.utils.data.TensorDataset(all_images)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)

    # Class labels
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

    my_table = wandb.Table(columns=["image", "label", "prediction"])

    # Make predictions
    predictions = predict(model, dataloader)

    # Save predictions to a text file
    with open("reports/figures/predictions.txt", "w") as file:
        for i, row in enumerate(predictions.tolist()):
            predicted_class, class_name = predict_single_sample(model, all_images[i], class_labels)
            file.write(f"Image: {img_names[i]}: Predicted Class: {predicted_class}, Class Name: {class_name}\n")
            file.write("\t".join(map(str, row)) + "\n")

            # Log your predictions to W&B
            # Assuming all_images[i] is a torch tensor representing an image
            wandb_image = wandb.Image(all_images[i].view(28, 28).numpy(), caption=img_names[i])
            my_table.add_data(wandb_image, img_names[i], class_name)

    # Log your Table to W&B
    wandb.log({"mnist_predictions": my_table})
    print("Finished")
