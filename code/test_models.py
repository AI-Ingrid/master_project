import copy
import pandas as pd
import torch
from utils.data_utils import plot_dataset_distribution
from sklearn.metrics import f1_score, precision_recall_fscore_support
from config import load_best_model, get_data_dist
import numpy as np
import pathlib
import scipy
from utils.neural_nets_utils import to_cuda
import os
import cv2
from tqdm import tqdm


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


def get_predictions(model, test_dataset, test_slide_ratio, num_frames, data_path):
    # pred: [(airway, direction), (airway, direction), .......]
    # targets: [(airway, direction), (airway, direction), .......]
    all_predictions = []
    all_targets = []

    sequence_list = list(os.listdir(f"{data_path}/datasets/test"))
    test_sequences = [file.split(".")[0] for file in sequence_list if (file.lower().endswith('.csv'))]
    video_counter = 0

    # Go through every video in test data set
    for video_frames, (airway_labels, direction_labels) in tqdm(test_dataset):
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
            predictions_airway, predictions_direction = model(stack_5D)  # (1, 5, 27), (1, 5, 2)

            # Remove batch dim
            predictions_airway = torch.squeeze(predictions_airway)  # (5, 27)
            predictions_direction = torch.squeeze(predictions_direction)  # (5, 2)

            # Softmax
            probabilities_airway = torch.softmax(predictions_airway, dim=1).detach().cpu()    # (5, 27)
            probabilities_direction = torch.softmax(predictions_direction, dim=1).detach().cpu()   # (5, 2)

            # Free memory
            del predictions_airway, predictions_direction

            # Argmax
            predictions_airway = np.argmax(probabilities_airway, axis=1)  # (5)
            predictions_direction = np.argmax(probabilities_direction, axis=1)  # (5)

            # Interpolate - Resize
            full_stack_predictions_airway = scipy.ndimage.zoom(predictions_airway, zoom=(test_slide_ratio), order=0)  # [50]
            full_stack_predictions_direction = scipy.ndimage.zoom(predictions_direction, zoom=(test_slide_ratio), order=0)  # [50]

            # Store stack predictions in the current video predictions list
            airway_predictions += full_stack_predictions_airway.tolist()  # airway_predictions: [[stack1], [stack2], [stack3], [stack4] ... [stack16]]
            direction_predictions += full_stack_predictions_direction.tolist()  # direction_predictions [[stack1], [stack2], [stack3], [stack4] ... [stack16]]

        # Store the entire video predictions (without extended frames) in a list with all videos
        all_predictions.append((airway_predictions[:len(video_frames)], direction_predictions[:len(video_frames)]))

        # Store the entire video predictions (without extended frames) in a csv file
        frame_names = [f"frame_{i}.png" for i in range(len(airway_labels))]

        temp_prediction_dict = {"Frame": frame_names, "Airway Prediction": airway_predictions[:len(video_frames)], "Direction Predictions": direction_predictions[:len(video_frames)]}
        all_predictions_df = pd.DataFrame(temp_prediction_dict)

        directory_path = pathlib.Path(f"{data_path}/test_set_predictions")
        directory_path.mkdir(exist_ok=True)

        all_predictions_df.to_csv(f"{directory_path}/Predictions_Patient_001_{test_sequences[video_counter]}.csv", index=False, mode='a')

        # Argmax on one hot encoded ground truth values
        airway_labels = np.argmax(airway_labels, axis=1)
        direction_labels = np.argmax(direction_labels, axis=1)

        # Store all targets in a all targets list
        all_targets.append((airway_labels.tolist(), direction_labels.tolist()))

        # Store the entire video labels in a csv file
        temp_target_dict = {"Frame": frame_names, "Airway Label": airway_labels, "Direction Label": direction_labels}
        all_predictions_df = pd.DataFrame(temp_target_dict)
        all_predictions_df.to_csv(f"{directory_path}/Labels_Patient_001_{test_sequences[video_counter]}.csv", index=False, mode='a')

        # Increase video counter
        video_counter += 1

    return all_predictions, all_targets


def convert_model_to_onnx(model, num_frames, dimension, model_name):
    dummy_input = torch.randn(1, num_frames, 3, dimension[0], dimension[1])  # Have to use batch size 1 since test set does not use batches
    input = to_cuda(dummy_input)
    torch.onnx.export(model, (input,), f'{model_name}.onnx')


def map_synthetic_frames_and_test_frames(data_path):
    # Get all test sequences
    sequence_list = list(os.listdir(f"{data_path}/datasets/test"))
    test_sequences = [file.split(".")[0] for file in sequence_list if (file.lower().endswith('.csv'))]

    # Create directory for the test frames
    test_directory_path = pathlib.Path(f"{data_path}/test_frames")
    test_directory_path.mkdir(exist_ok=True)

    # Go through every test sequence to store test frames
    for test_sequence in test_sequences:
        # Get frames in current sequence
        test_sequence_df = pd.read_csv(f"{data_path}/datasets/test/{test_sequence}.csv")

        # Create directory for the sequence
        sequence_directory_path = pathlib.Path(f"{test_directory_path}/{test_sequence}")
        sequence_directory_path.mkdir(exist_ok=True)

        frame_count = 0
        # Read every frame and store them in new directory with new name
        for frame_path in test_sequence_df["Frame"]:
            frame = cv2.imread(frame_path)
            cv2.imwrite(f"{sequence_directory_path}/frame_{frame_count}.png", frame)
            frame_count += 1
            

def test_model(trainer, test_dataset, test_slide_ratio, num_frames, num_airway_classes, num_direction_classes, data_path, frame_dimension, convert_to_onnx, model_name):
    print("-- TESTING --")
    # Load neural net model
    if load_best_model:
        trainer.load_model()

        # Set to inference mode -> freeze model
        trainer.model.eval()

    # Convert model to ONNX
    if convert_to_onnx:
        convert_model_to_onnx(trainer.model, num_frames, frame_dimension, model_name)

    # Run predictions on test set
    predictions, targets = get_predictions(trainer.model, test_dataset, test_slide_ratio, num_frames, data_path)

    #map_synthetic_frames_and_test_frames(data_path)

    # Get F1 Macro Score, Precision and Recall
    get_metrics(predictions, targets, num_airway_classes, num_direction_classes)
