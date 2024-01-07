from omegaconf import OmegaConf
import hydra
import os

from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from torch import nn, optim
import torch

from data.get_data import get_dataloaders




class Network(LightningModule):
    """Builds a feedforward network with arbitrary hidden layers.

    Arguments:
        input_size: integer, size of the input layer
        output_size: integer, size of the output layer
        hidden_layers: list of integers, the sizes of the hidden layers

    """

    def __init__(self, config):
        super().__init__()
        self.model_name = "SimpleFeedForward"
        self.classifier = nn.Sequential()
        self.lr = config.training.lr
        self.dropout = config.model.dropout
        self.hidden_layers = config.model.hidden_layers
        self.n_hidden = len(self.hidden_layers)

        # Define a model
        self.classifier.add_module('fc1', nn.Linear(config.model.n_input, self.hidden_layers[0]))
        self.classifier.add_module('relu1', nn.ReLU())
        self.classifier.add_module('dropout1', nn.Dropout(p=self.dropout))
        for i  in range(self.n_hidden - 1):
            self.classifier.add_module('fc{}'.format(i+2), nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]))
            self.classifier.add_module('relu{}'.format(i+2), nn.ReLU())
            self.classifier.add_module('dropout{}'.format(i+2), nn.Dropout(p=self.dropout))
        self.classifier.add_module('output', nn.Linear(self.hidden_layers[-1], config.model.n_output))
        self.classifier.add_module('softmax', nn.LogSoftmax(dim=1))

        self.criterion = nn.NLLLoss()

    def forward(self, x):
        """Forward pass through the network, returns the output logits."""
        return self.classifier(x)

    def extract_intermediate(self, x, intermediate_layer=2):
        """Extract intermediate representation from a specified layer."""
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if i == intermediate_layer:
                return x
        return None

    def training_step(self, batch, batch_idx):
        """Lightning calls this inside the training loop."""
        x, y = batch
        x.resize_(x.size()[0], 784)
        y.resize_(y.size()[0])
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """Lightning calls this inside the validation loop."""
        x, y = batch
        x.resize_(x.size()[0], 784)
        y.resize_(y.size()[0])
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def configure_optimizers(self):
        """Lightning calls this to setup optimizers."""
        return optim.Adam(self.parameters(), lr=self.lr)

    def accuracy(self, y_hat, y):
        """Calculate accuracy."""
                
        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(y_hat)
        # Class with highest probability is our predicted class, compare with true label
        equality = y.data == ps.max(1)[1]
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy = equality.type_as(torch.FloatTensor()).mean()
        return accuracy

@hydra.main(config_path="../conf", config_name="config.yaml")
def run_experiment(config):
    """Train model."""
    # configureation
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    # Define a model
    model = Network(config)  # this is our LightningModule

    # Get data
    repo_path = os.getcwd().split("cookie")[0] + "cookie"
    processed_data_path = repo_path + os.sep + config.training.dataset_path
    train_dataloader, val_dataloader = get_dataloaders(processed_data_path)

    # Define callbacks
    model_path = repo_path + os.sep + config.training.model_path + os.sep
    model_name = model.model_name + str(len(os.listdir(model_path)))
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path, filename=model_name + "-{epoch:02d}-{val_loss:.2f}", monitor="val_loss", mode="min"
        )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
        )

    # Final arguments
    trainer = Trainer(max_epochs=config.training.epochs, 
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=pl.loggers.WandbLogger(project=config.wandb.project)
        )
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    
    run_experiment()




