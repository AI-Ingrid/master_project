from random import seed
from preprocess import preprocess
from datasets import create_datasets_and_dataloaders
from neural_nets import create_neural_net
from train_models import train_model
from test_models import test_model


def main():
    """ The function running the entire pipeline of the project """
    seed(0)

    # Preprocess the data from videos to frames with labels
    preprocess()

    # Create datasets and dataloaders
    train_dataset, test_dataset, train_dataloader, test_dataloader = create_datasets_and_dataloaders()

    # Create neural network
    neural_net = create_neural_net()

    # Train model
    trainer = train_model(neural_net, train_dataloader)

    # Test model
    test_model(trainer, train_dataloader, test_dataloader, neural_net)


if __name__ == "__main__":
    main()
