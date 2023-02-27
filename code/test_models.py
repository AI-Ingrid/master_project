import copy
import torch
from utils.data_utils import plot_dataset_distribution
from train_models import compute_f1_and_loss
from torch.nn import CrossEntropyLoss
from utils.test_utils import plot_confusion_metrics, plot_predictions_test_set, compute_f1_score
from config import load_best_model, confusion_metrics_path, test_plot_path, get_data_dist, get_loss_and_accuracy, \
    get_confusion_metrics, get_f1_score, get_testset_pred, num_airway_segment_classes, num_direction_classes
import pandas as pd
import numpy as np
import pathlib
from utils.neural_nets_utils import to_cuda

def data_distribution(train, validation, test):
    if get_data_dist:
        # Visualize data sets
        plot_dataset_distribution(train, validation=None, test=None)
        plot_dataset_distribution(train, validation=validation, test=None)
        plot_dataset_distribution(train, validation=None, test=test)


def loss_and_accuracy(train, validation, test, neural_net):
    if get_loss_and_accuracy:
        print("---- TRAINING ----")
        compute_f1_and_loss(train, neural_net, CrossEntropyLoss(), num_airway_segment_classes, num_direction_classes)
        print("---- VALIDATION ----")
        compute_f1_and_loss(validation, neural_net, CrossEntropyLoss(), num_airway_segment_classes, num_direction_classes)
        print("---- TEST ----")
        compute_f1_and_loss(test, neural_net, CrossEntropyLoss(), num_airway_segment_classes, num_direction_classes)


def get_testset_predictions(model, test_dataset, test_slide_ratio, num_frames):
    all_predictions = []
    all_targets = []

    # Go through every video in test data set
    for video_frames, (airway_labels, direction_labels) in test_dataset:
        video_predictions = []

        # Handle stack edge case to make sure every stack has length num_frames
        extended_video_frames = copy.deepcopy(video_frames)
        num_left_over_frames = len(video_frames) % (num_frames * test_slide_ratio)

        if num_left_over_frames != 0:
            # Copy last frames to get equal stack length
            additional_frames = [video_frames[-1]] * ((num_frames * test_slide_ratio) - num_left_over_frames)
            additional_frames = torch.stack(additional_frames, dim=0)
            extended_video_frames = torch.cat([video_frames, additional_frames], dim=0)

        # Go through the frames with a given test slide ratio and number of frames in a stack
        for i in range(0, len(extended_video_frames), num_frames * test_slide_ratio):

            # Create a stack containing a given number of frames with a given slide ratio between the frames
            stack = extended_video_frames[i:i + (num_frames * test_slide_ratio): test_slide_ratio]

            # Reshape stack to 5D setting batch size to 1
            stack_shape = stack.shape
            stack_5D = stack.reshape(1, stack_shape[0], stack_shape[1], stack_shape[2], stack_shape[3])

            # Send Tensor to GPU
            stack_5D = to_cuda(stack_5D)

            # Send stack into the model and get predictions
            predictions_airway, predictions_direction = model(stack_5D)

            # Softmax
            probabilities_airway = torch.softmax(predictions_airway, dim=1)
            probabilities_direction = torch.softmax(predictions_direction, dim=1)

            # Free memory
            del predictions_airway, predictions_direction

            # Cast to numpy.cpu
            probabilities_airway = probabilities_airway.detach().cpu().numpy()
            probabilities_direction = probabilities_direction.detach().cpu().numpy()

            # Argmax i numpy her???

            # Store predictions in the current video predictions list
            video_predictions.append((probabilities_airway, probabilities_direction))

        # Convert video predictions to np array
        video_predictions = video_predictions.numpy()

        # Fill every frame with a prediction (interpolate)
        

        # Remove extended stack


        # Store video predictions in all predictions list
        all_predictions.append(video_predictions)

        # Store all targets in a all targets list
        all_targets.append((airway_labels, direction_labels))

    return all_predictions, all_targets


def test_model(trainer, test_dataset, test_slide_ratio, num_frames):
    print("-- TESTING --")
    checkpoint_dir = pathlib.Path(f"checkpoints_baseline_{10}")
    model_path = checkpoint_dir.joinpath("best_model.pth")
    model = torch.load(model_path)  # , map_location=torch.device('cpu'))
    model.eval()

    if not load_best_model:
        # Load neural net model
        trainer.load_model()

        # Set to inference mode (freeze model)
        trainer.model.eval()

    predictions, targets = get_testset_predictions(model, test_dataset, test_slide_ratio, num_frames)

