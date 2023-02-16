import typing
import time
import collections
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torchmetrics.classification as tm
import torch
from tqdm import tqdm
from utils.neural_nets_utils import to_cuda,  decode_one_hot_encoded_labels, save_checkpoint, load_best_checkpoint


def compute_loss_and_accuracy(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_criterion: torch.nn.modules.loss._Loss,
        num_airway_segment_classes: int,
        num_direction_classes: int):
    """
    Computes the average loss and the accuracy over the whole dataset
    in dataloader.
    Args:
        dataloder: Validation/Test dataloader
        model: torch.nn.Module
        loss_criterion: The loss criterion, e.g: torch.nn.CrossEntropyLoss()
    Returns:
        [average_loss, accuracy]: both scalar.
    """
    average_loss = 0
    f1_airway_segment = 0
    f1_direction = 0
    num_samples = 0
    batch_size = 0

    # Handle unbalanced dataset with the use of F1 Macro Score
    f1_airway_segment_metric = tm.F1Score(average='macro', task='multilabel', num_classes=num_airway_segment_classes)
    f1_direction_metric = tm.F1Score(average='macro', task='multilabel', num_classes=num_direction_classes)

    with torch.no_grad():
        for (X_batch, Y_batch) in tqdm(dataloader):
            # Transfer images/labels to GPU VRAM, if possible
            X_batch = to_cuda(X_batch)
            Y_batch = to_cuda(Y_batch)

            # Forward pass the images through our model
            output_probs = model(X_batch)

            predictions = torch.softmax(output_probs, dim=1)

            decoded_airway_segment_targets = decode_one_hot_encoded_labels(Y_batch[0])
            decoded_direction_targets = decode_one_hot_encoded_labels(Y_batch[1])

            airway_segment_targets = torch.tensor(np.array(decoded_airway_segment_targets))
            direction_targets = torch.tensor(np.array(decoded_direction_targets))

            num_samples += Y_batch[0].shape[0]

            # Compute F1 Score
            f1_airway_segment += f1_airway_segment_metric(predictions.cpu(), airway_segment_targets.cpu())
            f1_direction += f1_direction_metric(predictions.cpu(), direction_targets.cpu())

            # Compute Loss
            average_loss += loss_criterion(output_probs, Y_batch)
            batch_size += 1

    average_loss = average_loss / batch_size
    f1_airway_segment = f1_airway_segment / batch_size
    f1_direction = f1_direction / batch_size

    print(f'F1 Airway Segment Score: {f1_airway_segment}')
    print(f'F1 Direction Score: {f1_direction}')
    print(f'Loss: {average_loss}')

    return average_loss, f1_airway_segment, f1_direction


class Trainer:

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 early_stop_count: int,
                 epochs: int,
                 num_validations: int,
                 model: torch.nn.Module,
                 train_dataloader: typing.List[torch.utils.data.DataLoader],
                 validation_dataloader: typing.List[torch.utils.data.DataLoader],
                 fps: int,
                 num_airway_segment_classes: int,
                 num_direction_classes: int):
        """
            Initialize our trainer class.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_count = early_stop_count
        self.epochs = epochs
        self.num_validations = num_validations

        # Initialize the model
        self.model = model

        # Load our dataset
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        # Set variables
        self.fps = fps
        self.num_airway_segment_classes = num_airway_segment_classes
        self.num_direction_classes = num_direction_classes

        # Set loss criterion
        self.loss_criterion = torch.nn.CrossEntropyLoss()

        # Transfer model to GPU VRAM, if possible.
        self.model = to_cuda(self.model)

        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          self.learning_rate)

        # Validate the model everytime we pass through 1/num_validations of the dataset
        self.num_steps_per_val = len(self.train_dataloader) // self.num_validations
        self.global_step = 0
        self.start_time = time.time()

        # Tracking losses and accuracy
        self.train_history = dict(
            airway_segment_loss=collections.OrderedDict(),
            direction_loss=collections.OrderedDict(),
            combined_loss=collections.OrderedDict(),
            airway_segment_acc=collections.OrderedDict(),
            direction_acc=collections.OrderedDict(),
            combined_acc=collections.OrderedDict(),
        )
        self.validation_history = dict(
            airway_segment_loss=collections.OrderedDict(),
            direction_loss=collections.OrderedDict(),
            combined_loss=collections.OrderedDict(),
            airway_segment_acc=collections.OrderedDict(),
            direction_acc=collections.OrderedDict(),
            combined_acc=collections.OrderedDict(),
        )
        self.checkpoint_dir = pathlib.Path(f"checkpoints_{self.fps}")

    def validation_step(self):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()

        train_airway_segment_loss, train_direction_loss, train_acc = compute_loss_and_accuracy(self.train_dataloader, self.model, self.loss_criterion, self.num_airway_segment_classes, self.num_direction_classes)
        self.train_history["airway_segment_loss"][self.global_step] = train_airway_segment_loss
        self.train_history["direction_loss"][self.global_step] = train_direction_loss
        self.train_history["accuracy"][self.global_step] = train_acc

        val_airway_segment_loss, val_direction_loss, val_acc = compute_loss_and_accuracy(self.validation_dataloader, self.model, self.loss_criterion, self.num_airway_segment_classes, self.num_direction_classes)
        self.validation_history["airway_segment_loss"][self.global_step] = val_airway_segment_loss
        self.validation_history["direction_loss"][self.global_step] = val_direction_loss
        self.validation_history["combined_loss"][self.global_step] = val_airway_segment_loss + val_direction_loss
        self.validation_history["accuracy"][self.global_step] = val_acc

        used_time = time.time() - self.start_time
        print(
            f"Epoch: {self.epoch:>1}",
            f"Batches per seconds: {self.global_step / used_time:.2f}",
            f"Global step: {self.global_step:>6}",
            f"Validation Airway Segment Loss: {val_airway_segment_loss:.2f}",
            f"Validation Direction Loss: {val_direction_loss:.2f}",
            f"Validation Accuracy: {val_acc:.3f}",
            f"Train Accuracy: {train_acc:.3f}",
            sep=", ")
        self.model.train()

    def should_early_stop(self):
        """
            Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        val_loss = self.validation_history["combined_loss"]
        if len(val_loss) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = list(val_loss.values())[-self.early_stop_count:]
        first_loss = relevant_loss[0]
        if first_loss == min(relevant_loss):
            print("Early stop criteria met")
            return True
        return False

    def train_step(self, X_batch, Y_batch):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.
        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        X_batch = to_cuda(X_batch)
        Y_batch = to_cuda(Y_batch)

        # Perform the forward pass
        predictions = self.model(X_batch)

        # Compute the cross entropy loss for the batch
        airway_segment_loss = self.loss_criterion(predictions, Y_batch[0])
        direction_loss = self.loss_criterion(predictions, Y_batch[1])

        # Compute combined loss
        # TODO: vekting på lossa ln(num_classes)
        # TODO: L1 * w1 + L2* w2 (vekt-forholdstall)
        combined_loss = airway_segment_loss + direction_loss

        # Backpropagation
        combined_loss.backward()

        # Gradient descent step
        self.optimizer.step()

        # Reset all computed gradients to 0
        self.optimizer.zero_grad()

        return airway_segment_loss.detach().cpu().item(), direction_loss.detach().cpu().item(), combined_loss.detach().cpu().item()

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """

        def should_validate_model():
            return self.global_step % self.num_steps_per_val == 0

        for epoch in range(self.epochs):
            self.epoch = epoch
            print("Epoch: ", epoch)

            # Perform a full pass through all the training samples
            for X_batch, Y_batch in tqdm(self.train_dataloader):
                airway_segment_loss, direction_loss, combined_loss = self.train_step(X_batch, Y_batch)
                self.train_history["airway_segment_loss"][self.global_step] = airway_segment_loss
                self.train_history["direction_loss"][self.global_step] = direction_loss
                self.train_history["combined_loss"][self.global_step] = combined_loss

                self.global_step += 1

                # Compute loss/accuracy for validation set
                if should_validate_model():
                    self.validation_step()
                    self.save_model()
                    if self.should_early_stop():
                        print("Early stopping.")
                        return

    def save_model(self):
        def is_best_model():
            """
                Returns True if current model has the lowest validation loss
            """
            val_loss = self.validation_history["loss"]
            validation_losses = list(val_loss.values())
            return validation_losses[-1] == min(validation_losses)

        state_dict = self.model.state_dict()
        filepath = self.checkpoint_dir.joinpath(f"{self.global_step}.ckpt")

        save_checkpoint(state_dict, filepath, is_best_model())

    def load_best_model(self):
        state_dict = load_best_checkpoint(self.checkpoint_dir)
        if state_dict is None:
            print(
                f"Could not load best checkpoint. Did not find under: {self.checkpoint_dir}")
            return
        self.model.load_state_dict(state_dict)


def plot_loss(loss_dict: dict, label: str = None, npoints_to_average=1, plot_variance=True):
    """
    Args:
        loss_dict: a dictionary where keys are the global step and values are the given loss / accuracy
        label: a string to use as label in plot legend
        npoints_to_average: Number of points to average plot
    """
    global_steps = list(loss_dict.keys())
    loss = list(loss_dict.values())
    if npoints_to_average == 1 or not plot_variance:
        plt.plot(global_steps, loss, label=label)
        return

    npoints_to_average = 10
    num_points = len(loss) // npoints_to_average
    mean_loss = []
    loss_std = []
    steps = []
    for i in range(num_points):
        points = loss[i * npoints_to_average:(i + 1) * npoints_to_average]
        step = global_steps[i * npoints_to_average + npoints_to_average // 2]
        mean_loss.append(np.mean(points))
        loss_std.append(np.std(points))
        steps.append(step)
    plt.plot(steps, mean_loss,
             label=f"{label} (mean over {npoints_to_average} steps)")
    plt.fill_between(
        steps, np.array(mean_loss) -
               np.array(loss_std), np.array(mean_loss) + loss_std,
        alpha=.2, label=f"{label} variance over {npoints_to_average} steps")


def create_plots(trainer: Trainer, path: str, name: str):
    plot_path = pathlib.Path(path)
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    # Airway Segment Loss
    plot_loss(trainer.train_history["airway_segment_loss"], label="Training combined loss")
    plot_loss(trainer.validation_history["airway_segment_loss"], label="Validation combined loss")

    # Direction Loss
    plot_loss(trainer.train_history["direction_loss"], label="Training combined loss")
    plot_loss(trainer.validation_history["direction_loss"], label="Validation combined loss")

    # Combined loss
    plot_loss(trainer.train_history["combined_loss"], label="Training combined loss")
    plot_loss(trainer.validation_history["combined_loss"], label="Validation combined loss")

    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plot_loss(trainer.train_history["accuracy"], label="Training Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}.png"))
    plt.show()


def train_model(batch_size, learning_rate, early_stop_count, epochs, num_validations,
                neural_net, train_dataloader, validation_dataloader, fps, train_plot_path,
                train_plot_name, num_airway_segment_classes, num_direction_classes):
    print("TRAINING")
    # faster inference and training if set to True
    #torch.backends.cudnn.benchmark = True
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        num_validations,
        neural_net,
        train_dataloader,
        validation_dataloader,
        fps,
        num_airway_segment_classes,
        num_direction_classes
    )
    trainer.train()

    # Visualize training
    create_plots(trainer, train_plot_path, train_plot_name)

    return trainer