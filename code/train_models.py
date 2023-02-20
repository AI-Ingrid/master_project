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
    loss_airway = 0
    loss_direction = 0
    f1_airway = 0
    f1_direction = 0
    batch_size = 0

    # Handle unbalanced dataset with the use of F1 Macro Score
    #TODO: feil her: Usikker på om F1Score håndterer 3 dim input
    f1_airway_segment_metric = tm.F1Score(average='macro', task='multilabel', num_classes=num_airway_segment_classes)
    f1_direction_metric = tm.F1Score(average='macro', task='multilabel', num_classes=num_direction_classes)

    with torch.no_grad():
        for (X_batch, Y_batch) in tqdm(dataloader):
            # Transfer images/labels to GPU VRAM, if possible
            X_batch = to_cuda(X_batch)

            # Forward pass the images through the model
            predictions = model(X_batch)

            # Get the prediction probabilities
            predictions_airway = torch.softmax(predictions[0], dim=1)     # [16, 10, 27]
            predictions_direction = torch.softmax(predictions[1], dim=1)  # [16, 10, 2]

            # TODO: HER ER FEILEN decode funksjon er laga for 2D til 1D ikke for 3D til 2D
            # Decode the Y_batch for airway and direction separately and convert to Tensors
            #airway_decoded = decode_one_hot_encoded_labels(Y_batch[0])     # []
            #direction_decoded = decode_one_hot_encoded_labels(Y_batch[1])  # []

            # Get the ground truths in correct format
            ground_truth_airway = torch.tensor(np.array(Y_batch[0], dtype='int64'))
            ground_truth_direction = torch.tensor(np.array(Y_batch[1], dtype='int64'))

            print("SOFTMAX")
            print("airway shape: ", predictions_airway.shape)
            print("direction shape: ", predictions_direction.shape)

            print("TENSORS")
            print("airway shape: ", ground_truth_airway.shape)
            print("direction shape: ", ground_truth_direction.shape)

            # Compute F1 Score
            f1_airway += f1_airway_segment_metric(predictions_airway.cpu(), ground_truth_airway.cpu())
            f1_direction += f1_direction_metric(predictions_direction.cpu(), ground_truth_direction.cpu())

            # Compute Loss
            loss_airway += loss_criterion(predictions_airway, ground_truth_airway)
            loss_direction += loss_criterion(predictions_direction, ground_truth_direction)

            batch_size += 1

    loss_airway = loss_airway / batch_size
    loss_direction = loss_direction / batch_size
    f1_airway = f1_airway / batch_size
    f1_direction = f1_direction / batch_size

    print(f'F1 Airway Segment Score: {f1_airway}')
    print(f'F1 Direction Score: {f1_direction}')
    print(f'Loss Airway Segment: {loss_airway}')
    print(f'Loss Direction: {loss_direction}')

    return loss_airway, loss_direction, f1_airway, f1_direction


def compute_combined_loss(airway_loss, direction_loss):
    # TODO: vekting på lossa ln(num_classes)
    # TODO: L1 * w1 + L2* w2 (vekt-forholdstall)
    return airway_loss + direction_loss


def compute_combined_accuracy(airway_acc, direction_acc):
    # TODO: vekting på lossa ln(num_classes)
    # TODO: L1 * w1 + L2* w2 (vekt-forholdstall)
    return airway_acc + direction_acc


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
                 num_direction_classes: int,
                 num_frames_in_stack: int):
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
        self.num_frames_in_stack = num_frames_in_stack
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
        # Compute train loss and accuracy
        train_airway_loss, train_direction_loss, train_airway_acc, train_direction_acc = compute_loss_and_accuracy(
                                                                                        self.train_dataloader, self.model,
                                                                                        self.loss_criterion, self.num_airway_segment_classes,
                                                                                        self.num_direction_classes)
        # Store training accuracy
        self.train_history["airway_accuracy"][self.global_step] = train_airway_acc
        self.train_history["direction_accuracy"][self.global_step] = train_direction_acc
        self.train_history["combined_accuracy"][self.global_step] = compute_combined_accuracy(train_airway_acc, train_direction_acc)

        # Compute validation loss and accuracy
        val_airway_loss, val_direction_loss, val_airway_acc, val_direction_acc = compute_loss_and_accuracy(
                                                                                self.validation_dataloader,
                                                                                self.model, self.loss_criterion,
                                                                                self.num_airway_segment_classes,
                                                                                self.num_direction_classes)
        # Store validation loss and accuracy
        self.validation_history["airway_segment_loss"][self.global_step] = val_airway_loss
        self.validation_history["direction_loss"][self.global_step] = val_direction_loss
        self.validation_history["combined_loss"][self.global_step] = compute_combined_loss(val_airway_loss, val_direction_loss)
        self.validation_history["airway_accuracy"][self.global_step] = val_airway_acc
        self.validation_history["direction_accuracy"][self.global_step] = val_direction_acc
        self.validation_history["combined_accuracy"][self.global_step] = compute_combined_accuracy(val_airway_acc, val_direction_acc)

        used_time = time.time() - self.start_time
        print(
            f"Epoch: {self.epoch:>1}",
            f"Batches per seconds: {self.global_step / used_time:.2f}",
            f"Global step: {self.global_step:>6}",
            f"Validation Airway Segment Loss: {val_airway_loss:.2f}",
            f"Validation Direction Loss: {val_direction_loss:.2f}",
            f"Validation Airway Accuracy: {val_airway_acc:.3f}",
            f"Validation Direction Accuracy: {val_direction_acc:.3f}",
            f"Train Airway Accuracy: {train_airway_acc:.3f}",
            f"Train Direction Accuracy: {train_direction_acc:.3f}",
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
        Y_batch_airway = to_cuda(Y_batch[0])
        Y_batch_direction = to_cuda(Y_batch[1])

        # Perform the forward pass
        predictions = self.model(X_batch)

        # Reshape 3D to 2D for the loss function
        shape_airway = (self.batch_size * self.num_frames_in_stack, self.num_airway_segment_classes)
        shape_direction = (self.batch_size * self.num_frames_in_stack, self.num_direction_classes)

        Y_batch_airway = Y_batch_airway.reshape(shape_airway)
        Y_batch_direction = Y_batch_direction.reshape(shape_direction)
        predictions_airway = predictions[0].reshape(shape_airway)
        predictions_direction = predictions[1].reshape(shape_direction)

        # Calculate loss for airway segment and direction separately
        airway_loss = self.loss_criterion(predictions_airway, Y_batch_airway)
        direction_loss = self.loss_criterion(predictions_direction, Y_batch_direction)

        # Compute combined loss
        combined_loss = compute_combined_loss(airway_loss, direction_loss)

        # Backpropagation
        combined_loss.backward()

        # Gradient descent step
        self.optimizer.step()

        # Reset all computed gradients to 0
        self.optimizer.zero_grad()

        return airway_loss.detach().cpu().item(), direction_loss.detach().cpu().item(), combined_loss.detach().cpu().item()

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """

        def should_validate_model():
            return self.global_step % self.num_steps_per_val == 0

        for epoch in range(self.epochs):
            self.epoch = epoch
            #print("Epoch: ", epoch)

            # Perform a full pass through all the training samples
            for X_batch, Y_batch in tqdm(self.train_dataloader):
                airway_segment_loss, direction_loss, combined_loss = self.train_step(X_batch, Y_batch)

                # Store training history
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
                train_plot_name, num_airway_segment_classes, num_direction_classes, num_frames_in_stack):
    print("-- TRAINING --")

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
        num_direction_classes,
        num_frames_in_stack,
    )
    trainer.train()

    # Visualize training
    create_plots(trainer, train_plot_path, train_plot_name)

    return trainer
