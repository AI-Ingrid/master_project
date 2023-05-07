import typing
import time
import collections
import pathlib
import torchmetrics.classification as tm
import torch
from utils.neural_nets_utils import to_cuda
from torch.utils.tensorboard import SummaryWriter


def compute_f1_and_loss_for_airway(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_criterion: torch.nn.modules.loss._Loss,
        num_airway_segment_classes: int,
        alpha_airway: torch.Tensor,
        gamma: float,
        use_focal_loss: bool,
        num_frames_in_stack: int,
        batch_size: int,
        hidden: tuple(),):
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
    f1_airway = 0
    counter = 0

    # Handle unbalanced dataset with the use of F1 Macro Score
    f1_airway_segment_metric = tm.F1Score(average='macro', task='multilabel', num_classes=num_airway_segment_classes)

    with torch.no_grad():
        for (X_batch, (Y_batch_airway, Y_batch_direction)) in dataloader:

            X_batch = to_cuda(X_batch)
            Y_batch_airway = to_cuda(Y_batch_airway)
            hidden = to_cuda(hidden)

            # Perform the forward pass
            predictions_airway, hidden = model(X_batch, hidden)

            # Detach the hidden state and cell state after every batch
            hidden = model.reset_states()

            # Get the prediction probabilities
            predictions_airway_softmax = torch.softmax(predictions_airway, dim=-1)  # [16, 10, 27]

            # Decode the one-hot-encoding for predictions
            predictions_airway_decoded = torch.argmax(predictions_airway_softmax, axis=-1).flatten()

            # Decode the one-hot-encoding for targets
            targets_airway_decoded = torch.argmax(Y_batch_airway, axis=-1).flatten()

            # Compute F1 Score
            f1_airway += f1_airway_segment_metric(predictions_airway_decoded.cpu(), targets_airway_decoded.cpu())

            # Reshape 3D to 2D for the loss function
            shape_airway = (batch_size * num_frames_in_stack, num_airway_segment_classes)

            Y_batch_airway = Y_batch_airway.reshape(shape_airway)
            predictions_airway = predictions_airway.reshape(shape_airway)

            # Compute Loss
            cross_entropy_loss_airway = loss_criterion(predictions_airway, Y_batch_airway.float(), reduction='none')

            cross_entropy_loss_airway = cross_entropy_loss_airway.mean()

            # Calculate focal loss
            if use_focal_loss:
                pt_airway = torch.exp(-cross_entropy_loss_airway)
                focal_loss_airway = (alpha_airway * (1 - pt_airway) ** gamma * cross_entropy_loss_airway).mean()

                # Summarize over all batches
                loss_airway += focal_loss_airway

            else:
                # Summarize over all batches
                loss_airway += cross_entropy_loss_airway

            counter += 1

    loss_airway = loss_airway / counter
    f1_airway = f1_airway / counter

    return loss_airway, f1_airway, hidden

def compute_f1_and_loss_for_airway_and_direction(
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
        batch_size: int,
        hidden: tuple(),
        ):
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
    f1_airway_segment_metric = tm.F1Score(average='macro', task='multiclass', num_classes=num_airway_segment_classes)
    f1_direction_metric = tm.F1Score(average='macro', task='multiclass', num_classes=num_direction_classes)

    with torch.no_grad():
        for (X_batch, (Y_batch_airway, Y_batch_direction)) in dataloader:
            X_batch = to_cuda(X_batch)
            Y_batch_airway = to_cuda(Y_batch_airway)
            Y_batch_direction = to_cuda(Y_batch_direction)
            hidden = to_cuda(hidden)

            # Perform the forward pass
            predictions_airway, predictions_direction, hidden = model(X_batch, hidden)

            # Detach the hidden state and cell state after every batch
            hidden = model.reset_states()

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
            cross_entropy_loss_airway = loss_criterion(predictions_airway, Y_batch_airway.float(), reduction='none')
            cross_entropy_loss_direction = loss_criterion(predictions_direction, Y_batch_direction.float(), reduction='none')

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

    return loss_airway, loss_direction, f1_airway, f1_direction, hidden


def compute_combined_loss(airway_loss, direction_loss):
    # TODO: Trenger en vekt som balanserer ut den andre: ln(2)/ln(27)
    return airway_loss + direction_loss


class BaselineTrainer:

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 early_stop_count: int,
                 epochs: int,
                 model: torch.nn.Module,
                 train_dataloader: typing.List[torch.utils.data.DataLoader],
                 validation_dataloader: typing.List[torch.utils.data.DataLoader],
                 num_airway_segment_classes: int,
                 num_frames_in_stack: int,
                 model_path: str,
                 model_name: str,
                 use_focal_loss: bool,
                 alpha_airway: torch.Tensor,
                 gamma: float):
        """
            Initialize our trainer class.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_count = early_stop_count
        self.epochs = epochs

        # Initialize the model
        self.model = model

        # Load our dataset
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        # Set variables
        self.num_airway_segment_classes = num_airway_segment_classes
        self.num_frames_in_stack = num_frames_in_stack

        self.use_focal_loss = use_focal_loss
        self.alpha_airway = to_cuda(alpha_airway)
        self.gamma = gamma

        # Set loss criterion
        self.loss_criterion = torch.nn.functional.cross_entropy

        # Transfer model to GPU VRAM, if possible.
        self.model = to_cuda(self.model)

        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          self.learning_rate)

        self.global_step = 0
        self.start_time = time.time()

        # Tracking losses and accuracy
        self.train_history = dict(
            airway_segment_loss=collections.OrderedDict(),
            airway_f1=collections.OrderedDict(),
        )
        self.validation_history = dict(
            airway_segment_loss=collections.OrderedDict(),
            airway_f1=collections.OrderedDict(),
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
        train_airway_loss, train_airway_f1 = compute_f1_and_loss_for_baseline(
            dataloader=self.train_dataloader, model=self.model, loss_criterion=self.loss_criterion,
            num_airway_segment_classes=self.num_airway_segment_classes,
            alpha_airway=self.alpha_airway, gamma=self.gamma,
            use_focal_loss=self.use_focal_loss, num_frames_in_stack=self.num_frames_in_stack,
            batch_size=self.batch_size)

        # Store training accuracy in dictionary
        self.train_history["airway_f1"][self.global_step] = train_airway_f1

        # Store training accuracy in Tensorboard
        self.tensorboard_writer.add_scalar("train airway loss", train_airway_loss, self.global_step)
        self.tensorboard_writer.add_scalar("train airway f1", train_airway_f1, self.global_step)

        # Compute validation loss and accuracy
        val_airway_loss, val_airway_f1 = compute_f1_and_loss_for_baseline(
            dataloader=self.validation_dataloader, model=self.model, loss_criterion=self.loss_criterion,
            num_airway_segment_classes=self.num_airway_segment_classes,
            alpha_airway=self.alpha_airway, gamma=self.gamma,
            use_focal_loss=self.use_focal_loss, num_frames_in_stack=self.num_frames_in_stack,
            batch_size=self.batch_size)

        # Store validation loss and f1in dictionary
        self.validation_history["airway_segment_loss"][self.global_step] = val_airway_loss
        self.validation_history["airway_f1"][self.global_step] = val_airway_f1

        # Store validation loss and f1 in Tensorboard
        self.tensorboard_writer.add_scalar("validation airway loss", val_airway_loss, self.global_step)
        self.tensorboard_writer.add_scalar("validation airway f1", val_airway_f1, self.global_step)

        used_time = time.time() - self.start_time
        self.tensorboard_writer.add_scalar("used time", used_time, self.global_step)

        print(
            f"\nEpoch: {self.epoch:>1} Batches per seconds: {self.global_step / used_time:.2f} Global step: {self.global_step:>6}")
        print(f"Train F1 - Airway: {train_airway_f1:.3f}")
        print(f"Train Loss - Airway: {train_airway_loss:.2f}")
        print(f"Validation F1 - Airway: {val_airway_f1:.3f} ")
        print(f"Validation Loss - Airway: {val_airway_loss:.2f}")

        self.model.train()

    def should_early_stop(self):
        """
            Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        val_loss = self.validation_history["airway_segment_loss"]
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

        # Perform the forward pass
        predictions_airway = self.model(X_batch)

        # Reshape 3D to 2D for the loss function
        shape_airway = (self.batch_size * self.num_frames_in_stack, self.num_airway_segment_classes)

        Y_batch_airway = Y_batch_airway.reshape(shape_airway)
        predictions_airway = predictions_airway.reshape(shape_airway)  # [80, 27]

        # Calculate loss for airway segment and direction separately
        airway_loss = self.loss_criterion(predictions_airway, Y_batch_airway,
                                          reduction="none")  # [16 * 5 = 80 ] --> 3.37

        airway_loss = airway_loss.mean()

        # Calculate focal loss
        if self.use_focal_loss:
            pt_airway = torch.exp(-airway_loss)  # 16 * 5 = 80 eller bare ett tall?

            focal_loss_airway = (self.alpha_airway * (1 - pt_airway) ** self.gamma * airway_loss)

            airway_loss = focal_loss_airway.mean()

        # Backpropagation
        airway_loss.backward()

        # Gradient descent step
        self.optimizer.step()

        # Reset all computed gradients to 0
        self.optimizer.zero_grad()

        return airway_loss

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """

        for epoch in range(self.epochs):
            self.epoch = epoch
            print("Epoch: ", epoch)

            # Perform a full pass through all the training samples
            for X_batch, Y_batch in self.train_dataloader:
                airway_segment_loss = self.train_step(X_batch, Y_batch)

                # Store training history in dictionary
                self.train_history["airway_segment_loss"][self.global_step] = airway_segment_loss

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
            val_loss = self.validation_history["airway_segment_loss"]
            validation_losses = list(val_loss.values())
            return validation_losses[-1] == min(validation_losses)

        if is_best_model():
            self.model_dir.mkdir(exist_ok=True)
            model_path = self.model_dir.joinpath("best_model.pt")
            model = torch.jit.script(self.model)
            torch.jit.save(model, model_path)

    def load_model(self, inference_device):
        model_path = self.model_dir.joinpath("best_model.pt")
        self.model = torch.jit.load(model_path, map_location=torch.device(inference_device))

class NavigationNetTrainer:

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 early_stop_count: int,
                 epochs: int,
                 model: torch.nn.Module,
                 train_dataloader: typing.List[torch.utils.data.DataLoader],
                 validation_dataloader: typing.List[torch.utils.data.DataLoader],
                 num_airway_segment_classes: int,
                 num_direction_classes: int,
                 num_frames_in_stack: int,
                 model_path: str,
                 model_name: str,
                 use_focal_loss: bool,
                 alpha_airway: torch.Tensor,
                 alpha_direction: torch.Tensor,
                 gamma: float,
                 classify_direction: bool,
                 num_LSTM_cells: int,
                 hidden_size: int,
                 ):
        """
            Initialize our trainer class for NAvigationNet.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_count = early_stop_count
        self.epochs = epochs

        # Initialize the model
        self.model = model

        # Load our dataset
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        # Set variables
        self.num_airway_segment_classes = num_airway_segment_classes
        self.classify_direction = classify_direction
        self.num_frames_in_stack = num_frames_in_stack
        self.num_LSTM_cells = num_LSTM_cells
        self.hidden_size = hidden_size

        # Handle an additional classification: Direction
        if self.classify_direction:
            self.num_direction_classes = num_direction_classes

        # Set loss criterion
        self.loss_criterion = torch.nn.functional.cross_entropy
        self.use_focal_loss = use_focal_loss
        self.alpha_airway = to_cuda(alpha_airway)  # For Focal Loss
        self.alpha_direction = to_cuda(alpha_direction)  # For Focal Loss
        self.gamma = gamma  # For Focal Loss

        # Transfer model to GPU VRAM, if possible.
        self.model = to_cuda(self.model)

        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          self.learning_rate)

        # Validate the model everytime we pass through 1/num_validations of the dataset
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

    def validation_step(self, hidden):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        # Freeze model
        self.model.eval()

        # Compute loss and accuracy for airway and direction
        if self.classify_direction:
            train_airway_loss, train_direction_loss, train_airway_f1, train_direction_f1, hidden = \
                compute_f1_and_loss_for_airway_and_direction(dataloader=self.train_dataloader,
                                                             model=self.model,
                                                             loss_criterion=self.loss_criterion,
                                                             num_airway_segment_classes=self.num_airway_segment_classes,
                                                             num_direction_classes=self.num_direction_classes,
                                                             alpha_airway=self.alpha_airway,
                                                             alpha_direction=self.alpha_direction,
                                                             gamma=self.gamma,
                                                             use_focal_loss=self.use_focal_loss,
                                                             num_frames_in_stack=self.num_frames_in_stack,
                                                             batch_size=self.batch_size,
                                                             hidden=hidden)

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

            # Compute validation loss and accuracy for airway and direction
            val_airway_loss, val_direction_loss, val_airway_f1, val_direction_f1, hidden = \
                compute_f1_and_loss_for_airway_and_direction(dataloader=self.validation_dataloader,
                                                             model=self.model,
                                                             loss_criterion=self.loss_criterion,
                                                             num_airway_segment_classes=self.num_airway_segment_classes,
                                                             num_direction_classes=self.num_direction_classes,
                                                             alpha_airway=self.alpha_airway,
                                                             alpha_direction=self.alpha_direction,
                                                             gamma=self.gamma,
                                                             use_focal_loss=self.use_focal_loss,
                                                             num_frames_in_stack=self.num_frames_in_stack,
                                                             batch_size=self.batch_size,
                                                             hidden=hidden)

            val_combined_loss = compute_combined_loss(val_airway_loss, val_direction_loss)

            # Store validation loss and F1 in dictionary
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

        # Compute loss and accuracy for airway
        else:
            train_airway_loss, train_airway_f1, hidden  = compute_f1_and_loss_for_airway(dataloader=self.train_dataloader,
                                                                                         model=self.model,
                                                                                         loss_criterion=self.loss_criterion,
                                                                                         num_airway_segment_classes=self.num_airway_segment_classes,
                                                                                         alpha_airway=self.alpha_airway,
                                                                                         gamma=self.gamma,
                                                                                         use_focal_loss=self.use_focal_loss,
                                                                                         num_frames_in_stack=self.num_frames_in_stack,
                                                                                         batch_size=self.batch_size,
                                                                                         hidden=hidden,)

            # Store training accuracy in dictionary
            self.train_history["airway_f1"][self.global_step] = train_airway_f1

            # Store training accuracy in Tensorboard
            self.tensorboard_writer.add_scalar("train airway loss", train_airway_loss, self.global_step)
            self.tensorboard_writer.add_scalar("train combined loss", train_airway_loss, self.global_step)
            self.tensorboard_writer.add_scalar("train airway f1", train_airway_f1, self.global_step)

            # Compute validation loss and accuracy for airway and direction
            val_airway_loss, val_airway_f1, hidden = compute_f1_and_loss_for_airway(dataloader=self.validation_dataloader,
                                                                                    model=self.model,
                                                                                    loss_criterion=self.loss_criterion,
                                                                                    num_airway_segment_classes=self.num_airway_segment_classes,
                                                                                    alpha_airway=self.alpha_airway,
                                                                                    gamma=self.gamma,
                                                                                    use_focal_loss=self.use_focal_loss,
                                                                                    num_frames_in_stack=self.num_frames_in_stack,
                                                                                    batch_size=self.batch_size,
                                                                                    hidden=hidden)

            # Store validation loss and F1 in dictionary
            self.validation_history["airway_segment_loss"][self.global_step] = val_airway_loss
            self.validation_history["combined_loss"][self.global_step] = val_airway_loss
            self.validation_history["airway_f1"][self.global_step] = val_airway_f1

            # Store validation loss and f1 in Tensorboard
            self.tensorboard_writer.add_scalar("validation airway loss", val_airway_loss, self.global_step)
            self.tensorboard_writer.add_scalar("validation combined loss", val_airway_loss, self.global_step)
            self.tensorboard_writer.add_scalar("validation airway f1", val_airway_f1, self.global_step)

            used_time = time.time() - self.start_time
            self.tensorboard_writer.add_scalar("used time", used_time, self.global_step)

            print(f"\nEpoch: {self.epoch:>1} Batches per seconds: {self.global_step / used_time:.2f} Global step: {self.global_step:>6}")
            print(f"Train F1 - Airway: {train_airway_f1:.3f}")
            print(f"Train Loss - Airway: {train_airway_loss:.2f}")
            print(f"Validation F1 - Airway: {val_airway_f1:.3f}")
            print(f"Validation Loss - Airway: {val_airway_loss:.2f}")

        # Unfreeze model to perform training again
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

    def train_step(self, X_batch, Y_batch, hidden):
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
        hidden = to_cuda(hidden)

        # Forward pass DIRECTION
        if self.classify_direction:
            predictions_airway, predictions_direction, hidden = self.model(X_batch, hidden)

            # Reshape 3D to 2D for the loss function
            shape_direction = (self.batch_size * self.num_frames_in_stack, self.num_direction_classes)
            predictions_direction = predictions_direction.reshape(shape_direction)  # [80, 2]
            Y_batch_direction = Y_batch_direction.reshape(shape_direction)

            # Calculate loss for direction
            direction_loss = self.loss_criterion(predictions_direction, Y_batch_direction, reduction='none')  # [16 * 5 = 80]
            direction_loss = direction_loss.mean()

            # Calculate focal loss
            if self.use_focal_loss:
                pt_direction = torch.exp(-direction_loss)
                focal_loss_direction = (self.alpha_direction * (1 - pt_direction) ** self.gamma * direction_loss)
                direction_loss = focal_loss_direction.mean()
        else:
            predictions_airway, hidden = self.model(X_batch, hidden)
            direction_loss = 0 # When not classifying direction

        # Detach the hidden state and cell state after every batch
        hidden = self.model.reset_states()

        # Reshape 3D to 2D for the loss function
        shape_airway = (self.batch_size * self.num_frames_in_stack, self.num_airway_segment_classes)
        predictions_airway = predictions_airway.reshape(shape_airway)  # [80, 27]
        Y_batch_airway = Y_batch_airway.reshape(shape_airway)

        # Calculate loss for airway segment
        airway_loss = self.loss_criterion(predictions_airway, Y_batch_airway, reduction="none")  # [16 * 5 = 80 ]
        airway_loss = airway_loss.mean()

        # Calculate focal loss
        if self.use_focal_loss:
            pt_airway = torch.exp(-airway_loss)
            focal_loss_airway = (self.alpha_airway * (1 - pt_airway) ** self.gamma * airway_loss)
            airway_loss = focal_loss_airway.mean()

        # Compute combined loss
        combined_loss = compute_combined_loss(airway_loss, direction_loss)

        # Backpropagation
        combined_loss.backward()

        # Gradient descent step
        self.optimizer.step()

        # Reset all computed gradients to 0
        self.optimizer.zero_grad()

        return airway_loss, direction_loss, combined_loss, hidden

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        for epoch in range(self.epochs):
            self.epoch = epoch
            print("Epoch: ", epoch)

            # Initialize hidden state and cell state at the beginning of every epoch
            hidden_state = torch.zeros(self.num_LSTM_cells, self.batch_size, self.hidden_size)  # [num_layers,batch,hidden_size or H_out]
            cell_state = torch.zeros(self.num_LSTM_cells, self.batch_size, self.hidden_size)  # [num_layers,batch,hidden_size]

            # Perform a full pass through all the training samples
            for X_batch, Y_batch in self.train_dataloader:
                airway_segment_loss, direction_loss, combined_loss, (hidden_state, cell_state) = self.train_step(X_batch, Y_batch, (hidden_state, cell_state))

                # Store training history in dictionary
                self.train_history["airway_segment_loss"][self.global_step] = airway_segment_loss
                self.train_history["direction_loss"][self.global_step] = direction_loss
                self.train_history["combined_loss"][self.global_step] = combined_loss

                # Increase global step
                self.global_step += 1

            # Compute loss/accuracy for validation set
            self.validation_step((hidden_state, cell_state))
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

    def load_model(self, inference_device):
        model_path = self.model_dir.joinpath("best_model.pt")
        self.model = torch.jit.load(model_path, map_location=torch.device(inference_device))


def train_model(perform_training, batch_size, learning_rate, early_stop_count, epochs, neural_net, train_dataloader,
                validation_dataloader, num_airway_segment_classes, num_direction_classes, num_frames_in_stack,
                model_path, model_name, use_focal_loss, alpha_airway, alpha_direction, gamma, model_type,
                classify_direction, num_LSTM_cells, num_memory_nodes):

    # Initialize a BaselineTrainer
    if model_type == 'baseline':
        print("Trainer: Baseline")
        trainer = BaselineTrainer(batch_size=batch_size,
                                  learning_rate=learning_rate,
                                  early_stop_count=early_stop_count,
                                  epochs=epochs,
                                  model=neural_net,
                                  train_dataloader=train_dataloader,
                                  validation_dataloader=validation_dataloader,
                                  num_airway_segment_classes=num_airway_segment_classes,
                                  num_frames_in_stack=num_frames_in_stack,
                                  model_path=model_path,
                                  model_name=model_name,
                                  use_focal_loss=use_focal_loss,
                                  alpha_airway=alpha_airway,
                                  gamma=gamma,
                                  )

    # Initialize a NavigationNetTrainer
    else:
        print("Trainer: NavigationNet")
        trainer = NavigationNetTrainer(batch_size=batch_size,
                                       learning_rate=learning_rate,
                                       early_stop_count=early_stop_count,
                                       epochs=epochs,
                                       model=neural_net,
                                       train_dataloader=train_dataloader,
                                       validation_dataloader=validation_dataloader,
                                       num_airway_segment_classes=num_airway_segment_classes,
                                       num_frames_in_stack=num_frames_in_stack,
                                       model_path=model_path,
                                       model_name=model_name,
                                       use_focal_loss=use_focal_loss,
                                       alpha_airway=alpha_airway,
                                       alpha_direction=alpha_direction,
                                       gamma=gamma,
                                       num_direction_classes=num_direction_classes,
                                       classify_direction=classify_direction,
                                       num_LSTM_cells=num_LSTM_cells,
                                       hidden_size=num_memory_nodes,
                                       )

    if perform_training:
        print("-- TRAINING --")

        # Perform training
        trainer.train()

    return trainer
