import copy
import torch
from utils.data_utils import plot_dataset_distribution
from sklearn.metrics import f1_score, precision_recall_fscore_support
from config import load_best_model, get_data_dist
import numpy as np
import pathlib
import scipy
from utils.neural_nets_utils import to_cuda


def get_data_distribution(train, validation, test):
    if get_data_dist:
        # Visualize data sets
        plot_dataset_distribution(train, validation=None, test=None)
        plot_dataset_distribution(train, validation=validation, test=None)
        plot_dataset_distribution(train, validation=None, test=test)


def get_metrics(predictions, targets, num_airway_classes, num_direction_classes):
    f1_score_airway = 0
    f1_score_direction = 0
    precision_airway = 0
    precision_direction = 0
    recall_airway = 0
    recall_direction = 0

    for index, video in enumerate(predictions):
        f1_score_airway += f1_score(targets[index][0], video[0], average='macro')
        f1_score_direction += f1_score(targets[index][1], video[1], average='macro')

        temp_precision_airway, temp_recall_airway, _, _ = precision_recall_fscore_support(
                                                targets[index][0], video[0], average="weighted", labels=list(range(num_airway_classes)))
        temp_precision_direction, temp_recall_direction, _, _ = precision_recall_fscore_support(
                                                targets[index][1], video[1], average="macro", labels=list(range(num_direction_classes)))

        # Summarize the metrics one by one
        precision_airway += temp_precision_airway
        recall_airway += temp_recall_airway

        precision_direction += temp_precision_direction
        recall_direction += temp_recall_direction

    # Get the average for the metrics (every video is equally important)
    average_f1_score_airway = round(f1_score_airway/len(predictions), 3)
    average_f1_score_direction = round(f1_score_direction/len(predictions), 3)

    average_precision_airway = round(precision_airway/len(predictions), 3)
    average_precision_direction = round(precision_direction/len(predictions), 3)

    average_recall_airway = round(recall_airway/len(predictions), 3)
    average_recall_direction = round(recall_direction/len(predictions), 3)

    print("Average F1 Macro Score Airway: ",  average_f1_score_airway)
    print("Average F1 Macro Score Direction: ", average_f1_score_direction)
    print("Average Precision Airway: ", average_precision_airway)
    print("Average Precision Direction: ", average_precision_direction)
    print("Average Recall Airway: ", average_recall_airway)
    print("Average Recall Direction: ", average_recall_direction)


def get_predictions(model, test_dataset, test_slide_ratio, num_frames):
    all_predictions = []
    all_targets = []

    # Go through every video in test data set
    for video_frames, (airway_labels, direction_labels) in test_dataset:
        airway_predictions = []
        direction_predictions = []

        # Handle stack edge case to make sure every stack has length num_frames
        extended_video_frames = copy.deepcopy(video_frames)
        num_left_over_frames = (num_frames * test_slide_ratio) - len(video_frames) % (num_frames * test_slide_ratio)

        if num_left_over_frames != 0:
            # Copy last frames to get equal stack length
            additional_frames = [video_frames[-1]] * (num_left_over_frames)
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
            predictions_airway, predictions_direction = model(stack_5D)  # (1, 10, 27), (1, 10, 2)

            # Remove batch dim
            predictions_airway = torch.squeeze(predictions_airway)  # (10, 27)
            predictions_direction = torch.squeeze(predictions_direction)  # (10, 2)

            # Softmax
            probabilities_airway = torch.softmax(predictions_airway, dim=1).detach().cpu()    # (10, 27)
            probabilities_direction = torch.softmax(predictions_direction, dim=1).detach().cpu()   # (10, 2)

            # Free memory
            del predictions_airway, predictions_direction

            # Argmax
            predictions_airway = np.argmax(probabilities_airway, axis=1)  # (10)
            predictions_direction = np.argmax(probabilities_direction, axis=1)  # (10)

            # Interpolate - Resize
            full_stack_predictions_airway = scipy.ndimage.zoom(predictions_airway, zoom=5, order=0)
            full_stack_predictions_direction = scipy.ndimage.zoom(predictions_direction, zoom=5, order=0)

            # Store stack predictions in the current video predictions list
            airway_predictions += full_stack_predictions_airway.tolist()  #[stack1, stack2, stack3, stack4 ... stack8]
            direction_predictions += full_stack_predictions_direction.tolist()  #[stack1, stack2, stack3, stack4 ... stack8]

        # Store the entire video predictions (without extended frames) in a list with all videos
        all_predictions.append((airway_predictions[:len(video_frames)], direction_predictions[:len(video_frames)]))

        # Argmax on one hot encoded ground truth values
        airway_labels = np.argmax(airway_labels, axis=1)
        direction_labels = np.argmax(direction_labels, axis=1)

        # Store all targets in a all targets list
        all_targets.append((airway_labels.tolist(), direction_labels.tolist()))

    return all_predictions, all_targets


def test_model(trainer, test_dataset, test_slide_ratio, num_frames, num_airway_classes, num_direction_classes):
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

    # Run predictions on testset
    predictions, targets = get_predictions(model, test_dataset, test_slide_ratio, num_frames)

    # Get F1 Macro Score, Precision and Recall
    get_metrics(predictions, targets, num_airway_classes, num_direction_classes)
