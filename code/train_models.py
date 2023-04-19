import typing
import time
import collections
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torchmetrics.classification as tm
import torch
from tqdm import tqdm
from utils.neural_nets_utils import to_cuda
from torch.utils.tensorboard import SummaryWriter


def compute_f1_and_loss(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_criterion: torch.nn.modules.loss._Loss,
        num_airway_segment_classes: int,
        num_direction_classes: int,
        alpha_airway: torch.Tensor,
        alpha_direction: torch.Tensor,
        gamma: float,
        use_focal_loss: bool,
        num_frames_in_stack: int,
        batch_size: int):
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
    counter = 0

    # Handle unbalanced dataset with the use of F1 Macro Score
    f1_airway_segment_metric = tm.F1Score(average='macro', task='multilabel', num_classes=num_airway_segment_classes)
    f1_direction_metric = tm.F1Score(average='macro', task='multilabel', num_classes=num_direction_classes)

    with torch.no_grad():
        for (X_batch, (Y_batch_airway, Y_batch_direction)) in dataloader:

            X_batch = to_cuda(X_batch)
            Y_batch_airway = to_cuda(Y_batch_airway)
            Y_batch_direction = to_cuda(Y_batch_direction)

            # Perform the forward pass
            predictions = model(X_batch)
            predictions_airway = predictions[0]
            predictions_direction = predictions[1]

            # Get the prediction probabilities
            predictions_airway_softmax = torch.softmax(predictions_airway, dim=-1)  # [16, 10, 27]
            predictions_direction_softmax = torch.softmax(predictions_direction, dim=-1)  # [16, 10, 2]

            # Decode the one-hot-encoding for predictions
            predictions_airway_decoded = torch.argmax(predictions_airway_softmax, axis=-1).flatten()
            predictions_direction_decoded = torch.argmax(predictions_direction_softmax, axis=-1).flatten()

            # Decode the one-hot-encoding for targets
            targets_airway_decoded = torch.argmax(Y_batch_airway, axis=-1).flatten()
            targets_direction_decoded = torch.argmax(Y_batch_direction, axis=-1).flatten()

            # Compute F1 Score
            f1_airway += f1_airway_segment_metric(predictions_airway_decoded.cpu(), targets_airway_decoded.cpu())
            f1_direction += f1_direction_metric(predictions_direction_decoded.cpu(), targets_direction_decoded.cpu())

            # Reshape 3D to 2D for the loss function
            shape_airway = (batch_size * num_frames_in_stack, num_airway_segment_classes)
            shape_direction = (batch_size * num_frames_in_stack, num_direction_classes)

            Y_batch_airway = Y_batch_airway.reshape(shape_airway)
            Y_batch_direction = Y_batch_direction.reshape(shape_direction)
            predictions_airway = predictions_airway.reshape(shape_airway)
            predictions_direction = predictions_direction.reshape(shape_direction)

            # Compute Loss
            cross_entropy_loss_airway = loss_criterion(predictions_airway.cpu(), Y_batch_airway.float().cpu(), reduction='none')
            cross_entropy_loss_direction = loss_criterion(predictions_direction.cpu(), Y_batch_direction.float().cpu(), reduction='none')

            cross_entropy_loss_airway = cross_entropy_loss_airway.mean()
            cross_entropy_loss_direction = cross_entropy_loss_direction.mean()

            # Calculate focal loss
            if use_focal_loss:
                pt_airway = torch.exp(-cross_entropy_loss_airway)
                pt_direction = torch.exp(-cross_entropy_loss_direction)

                focal_loss_airway = (alpha_airway * (1 - pt_airway) ** gamma * cross_entropy_loss_airway).mean()
                focal_loss_direction = (alpha_direction * (1 - pt_direction) ** gamma * cross_entropy_loss_direction).mean()

                # Summarize over all batches
                loss_airway += focal_loss_airway
                loss_direction += focal_loss_direction

            else:
                # Summarize over all batches
                loss_airway += cross_entropy_loss_airway
                loss_direction += cross_entropy_loss_direction

            counter += 1

    loss_airway = loss_airway / counter
    loss_direction = loss_direction / counter
    f1_airway = f1_airway / counter
    f1_direction = f1_direction / counter

    return loss_airway, loss_direction, f1_airway, f1_direction


def compute_combined_loss(airway_loss, direction_loss):
    # TODO: Trenger en vekt som balanserer ut den andre: ln(2)/ln(27)
    return airway_loss + direction_loss


def compute_combined_accuracy(airway_acc, direction_acc):
    return (airway_acc + direction_acc)/2.0


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
                 num_frames_in_stack: int,
                 model_path: str,
                 model_name: str,
                 use_focal_loss: bool,
                 alpha_airway: torch.Tensor,
                 alpha_direction: torch.Tensor,
                 gamma: float):
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

        self.use_focal_loss = use_focal_loss
        self.alpha_airway = to_cuda(alpha_airway)
        self.alpha_direction = to_cuda(alpha_direction)
        self.gamma = gamma

        # Set loss criterion
        self.loss_criterion = torch.nn.functional.cross_entropy

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
            airway_f1=collections.OrderedDict(),
            direction_f1=collections.OrderedDict(),
            combined_f1=collections.OrderedDict(),
        )
        self.validation_history = dict(
            airway_segment_loss=collections.OrderedDict(),
            direction_loss=collections.OrderedDict(),
            combined_loss=collections.OrderedDict(),
            airway_f1=collections.OrderedDict(),
            direction_f1=collections.OrderedDict(),
            combined_f1=collections.OrderedDict(),
        )
        self.model_dir = pathlib.Path(model_path + model_name)
        self.tensorboard_writer = SummaryWriter(model_path + model_name)

    def validation_step(self):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()
        # Compute train loss and accuracy
        train_airway_loss, train_direction_loss, train_airway_f1, train_direction_f1 = compute_f1_and_loss(
                                                                                        self.train_dataloader, self.model,
                                                                                        self.loss_criterion, self.num_airway_segment_classes,
                                                                                        self.num_direction_classes, self.alpha_airway, self.alpha_direction,
                                                                                        self.gamma, self.use_focal_loss, self.num_frames_in_stack, self.batch_size)
        # Store training accuracy in dictionary
        self.train_history["airway_f1"][self.global_step] = train_airway_f1
        self.train_history["direction_f1"][self.global_step] = train_direction_f1

        train_combined_loss = compute_combined_loss(train_airway_loss, train_direction_loss)

        # Store training accuracy in Tensorboard
        self.tensorboard_writer.add_scalar("train airway loss", train_airway_loss, self.global_step)
        self.tensorboard_writer.add_scalar("train direction loss", train_direction_loss, self.global_step)
        self.tensorboard_writer.add_scalar("train combined loss", train_combined_loss, self.global_step)
        self.tensorboard_writer.add_scalar("train airway f1", train_airway_f1, self.global_step)
        self.tensorboard_writer.add_scalar("train direction f1", train_direction_f1, self.global_step)

        # Compute validation loss and accuracy
        val_airway_loss, val_direction_loss, val_airway_f1, val_direction_f1 = compute_f1_and_loss(
                                                                                self.validation_dataloader,
                                                                                self.model, self.loss_criterion,
                                                                                self.num_airway_segment_classes,
                                                                                self.num_direction_classes, self.alpha_airway,
                                                                                self.alpha_direction, self.gamma, self.use_focal_loss, self.num_frames_in_stack, self.batch_size)
        val_combined_loss = compute_combined_loss(val_airway_loss, val_direction_loss)

        # Store validation loss and f1in dictionary
        self.validation_history["airway_segment_loss"][self.global_step] = val_airway_loss
        self.validation_history["direction_loss"][self.global_step] = val_direction_loss
        self.validation_history["combined_loss"][self.global_step] = val_combined_loss
        self.validation_history["airway_f1"][self.global_step] = val_airway_f1
        self.validation_history["direction_f1"][self.global_step] = val_direction_f1

        # Store validation loss and f1 in Tensorboard
        self.tensorboard_writer.add_scalar("validation airway loss", val_airway_loss, self.global_step)
        self.tensorboard_writer.add_scalar("validation direction loss", val_direction_loss, self.global_step)
        self.tensorboard_writer.add_scalar("validation combined loss", val_combined_loss, self.global_step)
        self.tensorboard_writer.add_scalar("validation airway f1", val_airway_f1, self.global_step)
        self.tensorboard_writer.add_scalar("validation direction f1", val_direction_f1, self.global_step)

        used_time = time.time() - self.start_time
        self.tensorboard_writer.add_scalar("used time", used_time, self.global_step)

        print(f"\nEpoch: {self.epoch:>1} Batches per seconds: {self.global_step / used_time:.2f} Global step: {self.global_step:>6}")
        print(f"Train F1 - Airway: {train_airway_f1:.3f} - Direction: {train_direction_f1:.3f}")
        print(f"Train Loss - Airway: {train_airway_loss:.2f} - Direction: {train_direction_loss:.2f}")
        print(f"Validation F1 - Airway: {val_airway_f1:.3f} - Direction: {val_direction_f1:.3f}")
        print(f"Validation Loss - Airway: {val_airway_loss:.2f} - Direction: {val_direction_loss:.2f}")

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
        predictions_airway = predictions[0]
        predictions_direction = predictions[1]

        # Reshape 3D to 2D for the loss function
        shape_airway = (self.batch_size * self.num_frames_in_stack, self.num_airway_segment_classes)
        shape_direction = (self.batch_size * self.num_frames_in_stack, self.num_direction_classes)

        Y_batch_airway = Y_batch_airway.reshape(shape_airway)
        Y_batch_direction = Y_batch_direction.reshape(shape_direction)
        predictions_airway = predictions_airway.reshape(shape_airway)  # [80, 27]
        predictions_direction = predictions_direction.reshape(shape_direction)  # [80, 2]

        # Calculate loss for airway segment and direction separately
        airway_loss = self.loss_criterion(predictions_airway, Y_batch_airway, reduction="none")  # [16 * 5 = 80 ] --> 3.37
        direction_loss = self.loss_criterion(predictions_direction, Y_batch_direction, reduction='none') # [16 * 5 = 80] --> 0.6999

        airway_loss = airway_loss.mean()
        direction_loss = direction_loss.mean()

        # Calculate focal loss
        if self.use_focal_loss:
            pt_airway = torch.exp(-airway_loss)   # 16 * 5 = 80 eller bare ett tall?
            pt_direction = torch.exp(-direction_loss)  # 16 * 5 = 80 eller bare et tall altså etter mean??

            focal_loss_airway = (self.alpha_airway * (1 - pt_airway) ** self.gamma * airway_loss)
            focal_loss_direction = (self.alpha_direction * (1 - pt_direction) ** self.gamma * direction_loss)

            airway_loss = focal_loss_airway.mean()
            direction_loss = focal_loss_direction.mean()

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
            print("Epoch: ", epoch)

            # Perform a full pass through all the training samples
            for X_batch, Y_batch in self.train_dataloader:
                airway_segment_loss, direction_loss, combined_loss = self.train_step(X_batch, Y_batch)

                # Store training history in dictionary
                self.train_history["airway_segment_loss"][self.global_step] = airway_segment_loss
                self.train_history["direction_loss"][self.global_step] = direction_loss
                self.train_history["combined_loss"][self.global_step] = combined_loss

                self.global_step += 1

            # Compute loss/accuracy for validation set
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
            val_loss = self.validation_history["combined_loss"]
            validation_losses = list(val_loss.values())
            return validation_losses[-1] == min(validation_losses)

        if is_best_model():
            self.model_dir.mkdir(exist_ok=True)
            model_path = self.model_dir.joinpath("best_model.pt")
            model = torch.jit.script(self.model)
            torch.jit.save(model, model_path)

    def load_model(self):
        model_path = self.model_dir.joinpath("best_model.pt")
        # TODO: dette blir torch script model og ikke torch model
        self.model = torch.jit.load(model_path, map_location=torch.device('cpu'))


def plot_loss(loss_dict: dict, label: str = None, color: str = None, npoints_to_average=1, plot_variance=True):
    """
    Args:
        loss_dict: a dictionary where keys are the global step and values are the given loss / accuracy
        label: a string to use as label in plot legend
        npoints_to_average: Number of points to average plot
        color: color of curve in plot
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
    plt.plot(steps,
             mean_loss,
             label=f"{label} (mean over {npoints_to_average} steps)",
             color=color)
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
    plot_loss(trainer.train_history["airway_segment_loss"], label="Training Airway Loss", color='e06377')
    plot_loss(trainer.validation_history["airway_segment_loss"], label="Validation Airway Loss", color='5b9aa0')

    # Direction Loss
    plot_loss(trainer.train_history["direction_loss"], label="Training Direction Loss", color='e06377')
    plot_loss(trainer.validation_history["direction_loss"], label="Validation Direction Loss", color='5b9aa0')

    # Combined loss
    plot_loss(trainer.train_history["combined_loss"], label="Training Combined Loss", color='e06377')
    plot_loss(trainer.validation_history["combined_loss"], label="Combined Validation Loss", color='5b9aa0')

    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    # Airway Segment Accuracy
    plot_loss(trainer.train_history["airway_f1"], label="Training Airway F1", color='e06377')
    plot_loss(trainer.validation_history["airway_f1"], label="Validation Airway F1", color='5b9aa0')

    # Direction Accuracy
    plot_loss(trainer.train_history["direction_f1"], label="Training Direction F1", color='e06377')
    plot_loss(trainer.validation_history["direction_f1"], label="Validation Direction F1", color='5b9aa0')

    # Combined Accuracy
    plot_loss(trainer.train_history["combined_f1"], label="Combined Training F1", color='e06377')
    plot_loss(trainer.validation_history["combined_f1"], label="Combined Validation F1", color='5b9aa0')

    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}.png"))
    plt.show()


def train_model(perform_training, batch_size, learning_rate, early_stop_count, epochs, num_validations,
                neural_net, train_dataloader, validation_dataloader, fps, train_plot_path,
                train_plot_name, num_airway_segment_classes, num_direction_classes, num_frames_in_stack,
                model_path, model_name, use_focal_loss, alpha_airway, alpha_direction, gamma):
    trainer = Trainer(
        batch_size, learning_rate, early_stop_count, epochs, num_validations, neural_net, train_dataloader,
        validation_dataloader, fps, num_airway_segment_classes, num_direction_classes, num_frames_in_stack,
        model_path, model_name, use_focal_loss, alpha_airway, alpha_direction, gamma,
    )
    if perform_training:
        print("-- TRAINING --")
        trainer.train()

        # Visualize training
        create_plots(trainer, train_plot_path, train_plot_name)

    return trainer
