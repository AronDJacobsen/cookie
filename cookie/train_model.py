import logging
import os
import random

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from rich.logging import RichHandler
from torch import nn, optim

import wandb
from data.dataloader import get_dataloaders
from models.model import Network, save_model

log = logging.getLogger(__name__)
# log.root.handlers[0] = RichHandler(markup=True)  # set rich handler

# https://wandb.ai/adrishd/hydra-example/reports/Configuring-W-B-Projects-with-Hydra--VmlldzoxNTA2MzQw


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validation(model, testloader, criterion):
    """Validate the model on the testdata by calculating the sum of mean loss and mean accuracy for each test batch.

    Arguments:
        model: torch network
        testloader: torch.utils.data.DataLoader, dataloader of test set
        criterion: loss function
    """
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:
        images = images.resize_(images.size()[0], 784)
        labels = labels.resize_(labels.size()[0])
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = labels.data == ps.max(1)[1]
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy


# define hydra main entry point
@hydra.main(config_path="../conf", config_name="config.yaml")
def train(config):
    """Train model."""
    # configureation
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    # wandb
    # Initialize wandb
    # wandb.init(project="fashion-mnist", name="experiment1")
    # Initialize WandB
    run = wandb.init(project=config.wandb.project)

    # hparams = config.experiment
    h_model = config.model
    h_training = config.training
    # model specific

    model = Network(h_model.n_input, h_model.n_output, h_model.hidden_layers)
    # Magic
    wandb.watch(model, log_freq=100)  # steps

    # training specific
    print_every = h_training.print_every
    seed = h_training.seed
    torch.manual_seed(seed)
    epochs = h_training.epochs
    lr = h_training.lr

    # Log hyperparameters
    wandb.config.update({"epochs": epochs, "lr": lr})

    # initialize
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    repo_path = os.getcwd().split("cookie")[0] + "cookie"
    processed_data_path = repo_path + os.sep + h_training.dataset_path
    trainloader, valloader = get_dataloaders(processed_data_path)
    steps = 0
    running_loss = 0
    loss_list = []
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1

            # Flatten images into a 784 long vector
            images.resize_(images.size()[0], 784)
            labels.resize_(labels.size()[0])
            optimizer.zero_grad()

            # Forward and backward passes
            output = model.forward(images)
            loss = criterion(output, labels)
            # Log metrics
            wandb.log({"Training Loss": loss.item()})
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, valloader, criterion)
                # Log metrics
                wandb.log({"Test Loss": test_loss / len(valloader), "Test Accuracy": accuracy / len(valloader)})
                log.info(
                    "Epoch: {}/{}.. Training Loss: {:.3f}.. Test Loss: {:.3f}.. Test Accuracy: {:.3f}".format(
                        e + 1, epochs, running_loss / print_every, test_loss / len(valloader), accuracy / len(valloader)
                    )
                )

                running_loss = 0

                # Make sure dropout and grads are on for training
                model.train()

                # if loss < 1.7:
                #    break

    # Save a figure off the loss
    plt.figure(figsize=(10, 10))
    plt.plot(loss_list)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig(repo_path + "/reports/figures/loss.png")

    save_path = repo_path + "/models/trained_model.pth"
    save_model(model, save_path)


if __name__ == "__main__":
    train()
