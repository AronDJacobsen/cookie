import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os



class Network(nn.Module):
    """Builds a feedforward network with arbitrary hidden layers.

    Arguments:
        input_size: integer, size of the input layer
        output_size: integer, size of the output layer
        hidden_layers: list of integers, the sizes of the hidden layers

    """

    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()
        # Input to a hidden layer
        self.n_hidden = len(hidden_layers)
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        """Forward pass through the network, returns the output logits."""
        for each in self.hidden_layers:
            x = nn.functional.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return nn.functional.log_softmax(x, dim=1)

    def extract_intermediate(self, x, intermediate_layer=2):
        """Extract intermediate representation from a specified layer."""
        for idx, each in enumerate(self.hidden_layers):
            x = F.relu(each(x))
            x = self.dropout(x)

            if idx == intermediate_layer:
                return x

        return None

### Helper function

def predict_single_sample(model: torch.nn.Module, sample: torch.Tensor, class_labels: list) -> tuple:
    """Run prediction for a single sample using the given model.
    
    Args:
        model: model to use for prediction
        sample: input sample tensor
        class_labels: list of class labels
    
    Returns:
        Tuple containing the predicted label and the corresponding class name
    """
    model.eval()  # Set the model to evaluation mode
    device = next(model.parameters()).device  # Get the device of the model

    with torch.no_grad():
        # Move the sample to the device of the model
        sample = sample.to(device)
        
        # Perform prediction on the sample
        prediction = model(sample.unsqueeze(0))  # Add a batch dimension
        
    probs = torch.exp(prediction)
    _, predicted_class = probs.max(1)
    class_name = class_labels[predicted_class.item()]

    return predicted_class.item(), class_name




def save_model(model, save_path):
    checkpoint = {
        'input_size': 784,
        'output_size': 10,
        'hidden_layers': [each.out_features for each in model.hidden_layers],
        'state_dict': model.state_dict()
    }

    torch.save(checkpoint, save_path)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
