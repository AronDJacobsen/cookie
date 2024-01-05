import os
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

from models.model import Network, save_model
from data.get_data import get_data



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


def train(model, trainloader, testloader, criterion, optimizer=None, epochs=5, print_every=40):
    """Train model."""
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
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

            output = model.forward(images)
            loss = criterion(output, labels)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)

                print(
                    "Epoch: {}/{}.. ".format(e + 1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                    "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
                    "Test Accuracy: {:.3f}".format(accuracy / len(testloader)),
                )

                running_loss = 0

                # Make sure dropout and grads are on for training
                model.train()
                
                #if loss < 1.7:
                #    break

    # Save a figure off the loss
    plot = plt.figure(figsize=(10, 10))
    plt.plot(loss_list)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.savefig(os.getcwd() + '/reports/figures/loss.png')



if __name__ == "__main__":
    model = Network(784, 10, [512, 256, 128])
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    processed_data_path = os.getcwd() + os.sep + 'data/processed/processed_data.pt'
    trainloader, valloader = get_dataloaders(processed_data_path)

    train(model, trainloader, valloader, criterion, optimizer=None, epochs=1, print_every=40)

    save_path = 'models/trained_model.pth'
    save_model(model, save_path)
