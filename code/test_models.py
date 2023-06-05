import copy
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pathlib
import scipy
from utils.neural_nets_utils import to_cuda
import os
import cv2
from tqdm import tqdm
from torch import nn
from utils.test_utils import SoftmaxLayer
from matplotlib import pyplot as plt
import torchmetrics.classification as tm
from train_models import compute_f1_and_loss_for_airway, compute_f1_and_loss_for_airway_and_direction
import shutil
from typing import Tuple


def get_metrics(predictions, targets, num_airway_classes):
    f1_macro_airway = 0
    f1_micro_airway = 0

    f1_macro_airway_metric = tm.F1Score(average='macro', task='multiclass', num_classes=num_airway_classes)
    f1_micro_airway_metric = tm.F1Score(average='micro', task='multiclass', num_classes=num_airway_classes)

    for index, video in enumerate(predictions):
        # Convert to tensors
        airway_targets = np.array(targets[index])
        airway_targets = torch.tensor(airway_targets, dtype=torch.int64)

        # Convert to tensors
        airway_video = np.array(video)
        airway_video = torch.tensor(airway_video, dtype=torch.int64)

        # Convert both predictions and labels from 1-26 to 0-25 because F1Score is zero-indexed
        airway_video -= 1
        airway_targets -= 1

        f1_macro_airway += float(f1_macro_airway_metric(airway_video, airway_targets))
        f1_micro_airway += float(f1_micro_airway_metric(airway_video, airway_targets))

    # Get the average for the metrics (every video is equally important)
    average_f1_macro_airway = round(f1_macro_airway/len(predictions), 3)
    average_f1_micro_airway = round(f1_micro_airway/len(predictions), 3)

    print("Average F1 Macro Score Airway: ",  average_f1_macro_airway)
    print("Average F1 Micro Score Airway: ",  average_f1_micro_airway)


def get_metrics_with_direction(predictions, targets, num_airway_classes, num_direction_classes):
    # Initializing F1 score variables
    f1_macro_airway = 0
    f1_macro_direction = 0

    f1_macro_airway_metric = tm.F1Score(average='macro', task='multiclass', num_classes=num_airway_classes)
    f1_macro_direction_metric = tm.F1Score(average='macro', task='multiclass', num_classes=num_direction_classes)

    for index, video in enumerate(predictions):
        # Convert targets to tensors
        airway_targets = np.array(targets[index][0])
        airway_targets = torch.tensor(airway_targets, dtype=torch.int64)

        direction_targets = np.array(targets[index][1])
        direction_targets = torch.tensor(direction_targets, dtype=torch.int64)

        # Convert video predictions to tensors
        airway_video = np.array(video[0])
        airway_video = torch.tensor(airway_video, dtype=torch.int64)

        direction_video = np.array(video[1])
        direction_video = torch.tensor(direction_video, dtype=torch.int64)

        # Convert both predictions and labels from 1-26 to 0-25 because F1Score is zero-indexed
        airway_video -= 1
        airway_targets -= 1

        f1_macro_airway += float(f1_macro_airway_metric(airway_video, airway_targets))
        f1_macro_direction += float(f1_macro_direction_metric(direction_video, direction_targets))

    # Get the average for the metrics (every video is equally important)
    average_f1_macro_airway = round(f1_macro_airway/len(predictions), 3)
    average_f1_macro_direction = round(f1_macro_direction/len(predictions), 3)

    print("Average F1 Macro Score Airway: ",  average_f1_macro_airway)
    print("Average F1 Macro Score Direction: ", average_f1_macro_direction)


def get_test_set_predictions_for_baseline(model, test_dataset, test_slide_ratio, num_frames_in_test_stack, data_path, model_name):
    # Create path to store csv files with predictions and labels for the given model
    directory_path = pathlib.Path(f"{data_path}/test_set_predictions/{model_name}")
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

    all_predictions = [] # [(airway, direction), (airway, direction), .......]
    all_targets = [] # [(airway, direction), (airway, direction), .......]

    sequence_list = list(os.listdir(f"{data_path}/datasets/test"))
    test_sequences = [file.split(".")[0] for file in sequence_list if (file.lower().endswith('.csv'))]
    video_counter = 0

    # Go through every video in test data set
    for video_frames, (airway_labels, _) in tqdm(test_dataset):
        airway_predictions = []

        # Handle stack edge case to make sure every stack has length num_frames
        extended_video_frames = copy.deepcopy(video_frames)
        num_left_over_frames = (num_frames_in_test_stack * test_slide_ratio) - len(video_frames) % (num_frames_in_test_stack * test_slide_ratio)
        print("Stack size: ", num_frames_in_test_stack, " and num left over frames for video: ", num_left_over_frames)
        if num_left_over_frames != 0:
            # Copy last frames to get equal stack length
            additional_frames = [video_frames[-1]] * (num_left_over_frames)
            additional_frames = torch.stack(additional_frames, dim=0)
            extended_video_frames = torch.cat([video_frames, additional_frames], dim=0)

        # Go through the frames with a given test slide ratio and number of frames in a stack
        for i in range(0, len(extended_video_frames), num_frames_in_test_stack * test_slide_ratio):

            # Create a stack containing a given number of frames with a given slide ratio between the frames
            stack = extended_video_frames[i:i + (num_frames_in_test_stack * test_slide_ratio): test_slide_ratio]

            # Reshape stack to 5D setting batch size to 1
            stack_shape = stack.shape
            stack_5D = stack.reshape(1, stack_shape[0], stack_shape[1], stack_shape[2], stack_shape[3])

            # Send Tensor to GPU
            stack_5D = to_cuda(stack_5D)

            torch.manual_seed(42)

            # Send stack into the model and get predictions
            predictions_airway = model(stack_5D)  # (1, 5, 27)

            # Remove batch dim
            predictions_airway = torch.squeeze(predictions_airway)  # (5, 27)

            # Softmax
            probabilities_airway = torch.softmax(predictions_airway, dim=-1).detach().cpu()    # (5, 27)

            # Free memory
            del predictions_airway

            # Argmax
            predictions_airway = torch.argmax(probabilities_airway, axis=-1)  # (5)

            # Changing the airway predictions values from 0-25 to 1-26 to match the labels
            predictions_airway += 1

            # Interpolate - Resize
            full_stack_predictions_airway = scipy.ndimage.zoom(predictions_airway, zoom=(test_slide_ratio), order=0)  # [50]

            # Store stack predictions in the current video predictions list
            airway_predictions += full_stack_predictions_airway.tolist()  # airway_predictions: [[stack1], [stack2], [stack3], [stack4] ... [stack16]]

        # Store the entire video predictions (without extended frames) in a list with all videos
        all_predictions.append((airway_predictions[:len(video_frames)]))

        # Store the entire video predictions (without extended frames) in a csv file
        frame_names = [f"frame_{i}.png" for i in range(len(airway_labels))]
        temp_prediction_dict = {"Frame": frame_names, "Airway Prediction": airway_predictions[:len(video_frames)]}
        all_predictions_df = pd.DataFrame(temp_prediction_dict)
        all_predictions_df.to_csv(f"{directory_path}/Predictions_Patient_001_{test_sequences[video_counter]}.csv", index=False, mode='a')

        # Store all targets in a all targets list
        all_targets.append(airway_labels.tolist())

        # Store the entire video labels in a csv file
        temp_target_dict = {"Frame": frame_names, "Airway Label": airway_labels}
        all_predictions_df = pd.DataFrame(temp_target_dict)
        all_predictions_df.to_csv(f"{directory_path}/Labels_Patient_001_{test_sequences[video_counter]}.csv", index=False, mode='a')

        # Increase video counter
        video_counter += 1

    return all_predictions, all_targets


def get_test_set_predictions_with_direction(model, test_dataset, test_slide_ratio, num_frames_in_test_stack, data_path, model_name, stateful_testing=True):
    # Create path to store csv files with predictions and labels for the given model
    directory_path = pathlib.Path(f"{data_path}/test_set_predictions/{model_name}")
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

    all_predictions = []  # [(airway, direction), (airway, direction), .......]
    all_targets = []  # [(airway, direction), (airway, direction), .......]

    sequence_list = list(os.listdir(f"{data_path}/datasets/test"))
    test_sequences = [file.split(".")[0] for file in sequence_list if (file.lower().endswith('.csv'))]
    video_counter = 0

    # Go through every video in test data set
    for video_frames, (airway_labels, direction_labels) in tqdm(test_dataset):
        airway_predictions = []
        direction_predictions = []

        # Handle stack edge case to make sure every stack has length num_frames
        extended_video_frames = copy.deepcopy(video_frames)
        num_left_over_frames = (num_frames_in_test_stack * test_slide_ratio) - len(video_frames) % (num_frames_in_test_stack * test_slide_ratio)
        print("Stack size: ", num_frames_in_test_stack, " and num left over frames for video: ", num_left_over_frames)

        if num_left_over_frames != 0:
            # Copy last frames to get equal stack length
            additional_frames = [video_frames[-1]] * (num_left_over_frames)
            additional_frames = torch.stack(additional_frames, dim=0)
            extended_video_frames = torch.cat([video_frames, additional_frames], dim=0)

        # Initialize hidden state and cell state at the beginning of every video
        hidden_state = torch.zeros(model.num_stacked_LSTMs, 1, model.num_memory_nodes)  # OUT: [1, 1, 256]
        cell_state = torch.zeros(model.num_stacked_LSTMs, 1, model.num_memory_nodes)  # OUT: [1, 1, 256]
        hidden = to_cuda((hidden_state, cell_state))

        # Go through the frames with a given test slide ratio and number of frames in a stack
        for i in range(0, len(extended_video_frames), num_frames_in_test_stack * test_slide_ratio):

            # Create a stack containing a given number of frames with a given slide ratio between the frames
            stack = extended_video_frames[i:i + (num_frames_in_test_stack * test_slide_ratio): test_slide_ratio]

            # Reshape stack to 5D setting batch size to 1
            stack_shape = stack.shape
            stack_5D = stack.reshape(1, stack_shape[0], stack_shape[1], stack_shape[2], stack_shape[3])

            # Send Tensor to GPU
            stack_5D = to_cuda(stack_5D)
            torch.manual_seed(42)

            # Send stack into the model and get predictions
            predictions_airway, predictions_direction, hidden = model(stack_5D, hidden) # OUT: [1, 50 or 100, 26] and [1, 50 or 100, 2]

            # Remove batch dim
            predictions_airway = torch.squeeze(predictions_airway)  # OUT: [50 or 100, 26]
            predictions_direction = torch.squeeze(predictions_direction)  # OUT: [50 or 100, 2]

            # STATEFUL: Softmax
            probabilities_airway = torch.softmax(predictions_airway, dim=-1).detach().cpu()  # OUT: [50 or 100, 26]
            probabilities_direction = torch.softmax(predictions_direction, dim=-1).detach().cpu()  # OUT: [50 or 100, 2]

            # Free memory
            del predictions_airway, predictions_direction,

            # Argmax
            predictions_airway = torch.argmax(probabilities_airway, axis=-1)  # OUT: [50 or 100]
            predictions_direction = torch.argmax(probabilities_direction, axis=-1)  # OUT: [50 or 100]

            # Changing the airway predictions values from 0-25 to 1-26 to match the labels
            predictions_airway += 1

            # Interpolate - Resize
            full_stack_predictions_airway = scipy.ndimage.zoom(predictions_airway, zoom=(test_slide_ratio), order=0)  # OUT: [250 or 500]
            full_stack_predictions_direction = scipy.ndimage.zoom(predictions_direction, zoom=(test_slide_ratio), order=0)  # OUT: [250 or 500]

            # Store stack predictions in the current video predictions list
            airway_predictions += full_stack_predictions_airway.tolist()  # airway_predictions: [[stack1], [stack2], [stack3], [stack4] ... [stack16]]
            direction_predictions += full_stack_predictions_direction.tolist()  # direction_predictions [[stack1], [stack2], [stack3], [stack4] ... [stack16]]

            if not stateful_testing:
                # Reset the states
                model.reset_states()

        # Store the entire video predictions (without extended frames) in a list with all videos
        all_predictions.append((airway_predictions[:len(video_frames)], direction_predictions[:len(video_frames)]))

        # Store the entire video predictions (without extended frames) in a csv file
        frame_names = [f"frame_{i}.png" for i in range(len(airway_labels))]
        temp_prediction_dict = {"Frame": frame_names, "Airway Prediction": airway_predictions[:len(video_frames)], "Direction Predictions": direction_predictions[:len(video_frames)]}
        all_predictions_df = pd.DataFrame(temp_prediction_dict)
        all_predictions_df.to_csv(f"{directory_path}/Predictions_Patient_001_{test_sequences[video_counter]}.csv", index=False, mode='a')

        # Store all targets in a all targets list
        all_targets.append((airway_labels.tolist(), direction_labels.tolist()))

        # Store the entire video labels in a csv file
        temp_target_dict = {"Frame": frame_names, "Airway Label": airway_labels, "Direction Label": direction_labels}
        all_predictions_df = pd.DataFrame(temp_target_dict)
        all_predictions_df.to_csv(f"{directory_path}/Labels_Patient_001_{test_sequences[video_counter]}.csv", index=False, mode='a')

        # Increase video counter
        video_counter += 1

        # Reset the states
        model.reset_states()

    return all_predictions, all_targets


def get_test_set_predictions(model, test_dataset, test_slide_ratio, num_frames_in_test_stack, data_path, model_name, stateful_testing=True):
    # Create path to store csv files with predictions and labels for the given model
    directory_path = pathlib.Path(f"{data_path}/test_set_predictions/{model_name}")
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

    all_predictions = []  # [(airway, direction), (airway, direction), .......]
    all_targets = []  # [(airway, direction), (airway, direction), .......]

    sequence_list = list(os.listdir(f"{data_path}/datasets/test"))
    test_sequences = [file.split(".")[0] for file in sequence_list if (file.lower().endswith('.csv'))]
    video_counter = 0

    # Go through every video in test data set
    for video_frames, (airway_labels, _) in tqdm(test_dataset):
        airway_predictions = []

        # Handle stack edge case to make sure every stack has length num_frames
        extended_video_frames = copy.deepcopy(video_frames)
        num_left_over_frames = (num_frames_in_test_stack * test_slide_ratio) - len(video_frames) % (num_frames_in_test_stack * test_slide_ratio)
        print("Stack size: ", num_frames_in_test_stack, " and num left over frames for video: ", num_left_over_frames)

        if num_left_over_frames != 0:
            # Copy last frames to get equal stack length
            additional_frames = [video_frames[-1]] * (num_left_over_frames)
            additional_frames = torch.stack(additional_frames, dim=0)
            extended_video_frames = torch.cat([video_frames, additional_frames], dim=0)

        # Initialize hidden state and cell state at the beginning of every video
        hidden_state = torch.zeros(model.num_stacked_LSTMs, 1, model.num_memory_nodes)  # [num_layers,batch,hidden_size or H_out]
        cell_state = torch.zeros(model.num_stacked_LSTMs, 1, model.num_memory_nodes)  # [num_layers,batch,hidden_size]
        hidden = to_cuda((hidden_state, cell_state))

        # Go through the frames with a given test slide ratio and number of frames in a stack
        for i in range(0, len(extended_video_frames), num_frames_in_test_stack * test_slide_ratio):
            # Create a stack containing a given number of frames with a given slide ratio between the frames
            stack = extended_video_frames[i:i + (num_frames_in_test_stack * test_slide_ratio): test_slide_ratio]

            # Reshape stack to 5D setting batch size to 1
            stack_shape = stack.shape
            stack_5D = stack.reshape(1, stack_shape[0], stack_shape[1], stack_shape[2], stack_shape[3])

            # Send Tensor to GPU
            stack_5D = to_cuda(stack_5D)

            torch.manual_seed(42)

            # Send stack into the model and get predictions
            predictions_airway, hidden = model(stack_5D, hidden)  # out: (1, 5, 27), (1, 5, 2)

            # Remove batch dim
            predictions_airway = torch.squeeze(predictions_airway)  # out: (frames in stack, 27)

            # Softmax
            probabilities_airway = torch.softmax(predictions_airway, dim=-1).detach().cpu()    # (50, 27)

            # Free memory
            del predictions_airway

            # Argmax
            predictions_airway = torch.argmax(probabilities_airway, axis=-1)  # (5)

            # Changing the airway predictions values from 0-25 to 1-26 to match the labels
            predictions_airway += 1

            # Interpolate - Resize
            full_stack_predictions_airway = scipy.ndimage.zoom(predictions_airway, zoom=(test_slide_ratio), order=0)  # [50]

            # Store stack predictions in the current video predictions list
            airway_predictions += full_stack_predictions_airway.tolist()  # airway_predictions: [[stack1], [stack2], [stack3], [stack4] ... [stack16]]

            # Reset the states
            if not stateful_testing:
                model.reset_states()

        # Store the entire video predictions (without extended frames) in a list with all videos
        all_predictions.append((airway_predictions[:len(video_frames)]))

        # Store the entire video predictions (without extended frames) in a csv file
        frame_names = [f"frame_{i}.png" for i in range(len(airway_labels))]
        temp_prediction_dict = {"Frame": frame_names, "Airway Prediction": airway_predictions[:len(video_frames)]}
        all_predictions_df = pd.DataFrame(temp_prediction_dict)
        all_predictions_df.to_csv(f"{directory_path}/Predictions_Patient_001_{test_sequences[video_counter]}.csv", index=False, mode='a')

        # Store all targets in a all targets list
        all_targets.append(airway_labels.tolist())

        # Store the entire video labels in a csv file
        temp_target_dict = {"Frame": frame_names, "Airway Label": airway_labels}
        all_predictions_df = pd.DataFrame(temp_target_dict)
        all_predictions_df.to_csv(f"{directory_path}/Labels_Patient_001_{test_sequences[video_counter]}.csv", index=False, mode='a')

        # Increase video counter
        video_counter += 1

        # Reset the states
        model.reset_states()

    return all_predictions, all_targets


def convert_model_to_onnx_for_baseline(model, num_frames_in_test_stack, dimension, model_name, model_path):
    """ Converts a trained model into an onnx model. """
    # Set path for storing models as torch scripts
    model_directory_path = pathlib.Path(model_path + model_name)

    # Create a softmax layer that will be added at the end of the trained model
    softmax_layer = SoftmaxLayer()

    # Convert the SoftmaxLayer to  a torch script model
    softmax_layer_torchscript = torch.jit.script(softmax_layer)
    softmax_layer_path = model_directory_path.joinpath("softmax_layer.pt")

    # Save SoftmaxLayer as a torch script model
    torch.jit.save(softmax_layer_torchscript, softmax_layer_path)

    # Load the torch script model SoftmaxLayer
    softmax_layer = torch.jit.load(softmax_layer_path, map_location=torch.device('cuda'))

    # Freeze the weights
    softmax_layer.eval()

    # Add the SoftmaxLayer to the trained model in a nn.Module
    class ModelForOnnx(nn.Module):
        def __init__(self):
            super().__init__()
            self.trained_model = model
            self.softmax_layer = softmax_layer

        def forward(self, X):  #[16, 5, 27/2]
            airway_predictions = self.trained_model(X)
            airway_predictions = self.softmax_layer(airway_predictions)
            return airway_predictions

    # Convert the ModelForOnnx (nn.Module) to s torch script
    model_for_onnx = ModelForOnnx()
    model_for_onnx_torchscript = torch.jit.script(model_for_onnx)
    model_for_onnx_path = model_directory_path.joinpath("model_for_onnx.pt")

    # Save LastPred as a torchscript model
    torch.jit.save(model_for_onnx_torchscript, model_for_onnx_path)

    # Load the torchscript model LastPred
    model_for_onnx = torch.jit.load(model_for_onnx_path, map_location=torch.device('cuda'))

    # Freeze the weights
    model_for_onnx.eval()

    # Dummy input
    dummy_input_X = torch.randn(1, num_frames_in_test_stack, 3, dimension[0],
                                dimension[1])  # Have to use batch size 1 since test set does not use batches

    # To cuda
    dummy_input_X_cuda = to_cuda(dummy_input_X)

    torch.onnx.export(model_for_onnx,
                      (dummy_input_X_cuda,),
                      f'{model_path}onnx/{model_name}.onnx',
                      opset_version=11,
                      input_names=['input'],
                      output_names=['airway'],
                      dynamic_axes={
                          'input': {1: 'stack_size'},
                          'airway': {1: 'stack_size'},
                      },
                      verbose=False)
    print("Exported to ONNX")


def convert_model_to_onnx(model, num_frames_in_test_stack, dimension, model_name, model_path):
    """ Converts a trained model into an onnx model. """
    # Set path for storing models as torch scripts
    model_directory_path = pathlib.Path(model_path + model_name)

    # Create a softmax layer that will be added at the end of the trained model
    softmax_layer = SoftmaxLayer()

    # Convert the SoftmaxLayer to  a torch script model
    softmax_layer_torchscript = torch.jit.script(softmax_layer)
    softmax_layer_path = model_directory_path.joinpath("softmax_layer.pt")

    # Save SoftmaxLayer as a torch script model
    torch.jit.save(softmax_layer_torchscript, softmax_layer_path)

    # Load the torch script model SoftmaxLayer
    softmax_layer = torch.jit.load(softmax_layer_path, map_location=torch.device('cuda'))

    # Freeze the weights
    softmax_layer.eval()

    # Add the SoftmaxLayer to the trained model in a nn.Module
    class ModelForOnnx(nn.Module):
        def __init__(self, num_LSTM_cells, batch_size, num_memory_nodes):
            super().__init__()
            self.trained_model = model
            self.softmax_layer = softmax_layer

            # Initialize hidden state and cell state at the beginning of every epoch
            self.hidden_state = to_cuda(torch.zeros(num_LSTM_cells, batch_size, num_memory_nodes))  # [num_layers,batch,hidden_size or H_out]
            self.cell_state = to_cuda(torch.zeros(num_LSTM_cells, batch_size, num_memory_nodes))  # [num_layers,batch,hidden_size]

        def forward(self, X: torch.Tensor):  #[16, 5, 27/2]
            airway, (self.hidden_state, self.cell_state) = self.trained_model(X, (self.hidden_state, self.cell_state))
            airway = self.softmax_layer(airway)
            return airway

    # Convert the ModelForOnnx (nn.Module) to s torch script
    model_for_onnx = ModelForOnnx(model.num_stacked_LSTMs, 1, model.num_memory_nodes)
    model_for_onnx_torchscript = torch.jit.script(model_for_onnx)
    model_for_onnx_path = model_directory_path.joinpath("model_for_onnx.pt")

    # Save LastPred as a torchscript model
    torch.jit.save(model_for_onnx_torchscript, model_for_onnx_path)

    # Load the torchscript model LastPred
    model_for_onnx = torch.jit.load(model_for_onnx_path, map_location=torch.device('cuda'))

    # Freeze the weights
    model_for_onnx.eval()

    # Dummy input
    dummy_input_X = torch.randn(1, num_frames_in_test_stack, 3, dimension[0],
                                dimension[1])  # Have to use batch size 1 since test set does not use batches

    # To cuda
    dummy_input_X_cuda = to_cuda(dummy_input_X)

    torch.onnx.export(model_for_onnx,
                      (dummy_input_X_cuda,),
                      f'{model_path}onnx/{model_name}.onnx',
                      opset_version=11,
                      input_names=['input'],
                      output_names=['airway'],
                      dynamic_axes={
                          'input': {1: 'stack_size'},
                          'airway': {1: 'stack_size'},
                      },
                      verbose=False)
    print("Exported to ONNX")


def convert_model_to_onnx_with_direction(model, num_frames_in_test_stack, dimension, model_name, model_path):
    """ Converts a trained model into an onnx model. """
    # Set path for storing models as torch scripts
    model_directory_path = pathlib.Path(model_path + model_name)

    # Create a softmax layer that will be added at the end of the trained model
    softmax_layer = SoftmaxLayer()

    # Convert the SoftmaxLayer to  a torch script model
    softmax_layer_torchscript = torch.jit.script(softmax_layer)
    softmax_layer_path = model_directory_path.joinpath("softmax_layer.pt")

    # Save SoftmaxLayer as a torch script model
    torch.jit.save(softmax_layer_torchscript, softmax_layer_path)

    # Load the torch script model SoftmaxLayer
    softmax_layer = torch.jit.load(softmax_layer_path, map_location=torch.device('cuda'))

    # Freeze the weights
    softmax_layer.eval()

    # Add the SoftmaxLayer to the trained model in a nn.Module
    class ModelForOnnx(nn.Module):
        def __init__(self, num_LSTM_cells, batch_size, num_memory_nodes):
            super().__init__()
            self.trained_model = model
            self.softmax_layer = softmax_layer

            # Initialize hidden state and cell state at the beginning of every epoch
            self.hidden_state = to_cuda(torch.zeros(num_LSTM_cells, batch_size, num_memory_nodes))  # [num_layers,batch,hidden_size or H_out]
            self.cell_state = to_cuda(torch.zeros(num_LSTM_cells, batch_size, num_memory_nodes))  # [num_layers,batch,hidden_size]

        def forward(self, X: torch.Tensor):
            airway, direction, (self.hidden_state, self.cell_state) = self.trained_model(X, (self.hidden_state, self.cell_state))
            airway = self.softmax_layer(airway)
            direction = self.softmax_layer(direction)
            return airway, direction

    # Convert the ModelForOnnx (nn.Module) to s torch script
    model_for_onnx = ModelForOnnx(model.num_stacked_LSTMs, 1, model.num_memory_nodes)
    model_for_onnx_torchscript = torch.jit.script(model_for_onnx)
    model_for_onnx_path = model_directory_path.joinpath("model_for_onnx.pt")

    # Save LastPred as a torchscript model
    torch.jit.save(model_for_onnx_torchscript, model_for_onnx_path)

    # Load the torchscript model LastPred
    model_for_onnx = torch.jit.load(model_for_onnx_path, map_location=torch.device('cuda'))

    # Freeze the weights
    model_for_onnx.eval()

    # Dummy input
    dummy_input_X = torch.randn(1, num_frames_in_test_stack, 3, dimension[0], dimension[1])  # Have to use batch size 1 since test set does not use batches

    # To cuda
    dummy_input_X_cuda = to_cuda(dummy_input_X)

    torch.onnx.export(model_for_onnx,
                      (dummy_input_X_cuda,),
                      f'{model_path}onnx/{model_name}.onnx',
                      opset_version=11,
                      input_names=['input'],
                      output_names=['airway', 'direction'],
                      dynamic_axes={
                          'input': {1: 'stack_size'},
                          'airway': {1: 'stack_size'},
                          'direction': {1: 'stack_size'},
                      },
                      verbose=False)
    print("Exported to ONNX")


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


def plot_confusion_metrics(predictions, targets, confusion_metrics_plot_path, num_airway_classes, stateful_testing=True):
    plot_directory_path = pathlib.Path(confusion_metrics_plot_path)
    plot_directory_path.mkdir(exist_ok=True)
    if not stateful_testing:
        airway_plot_path = pathlib.Path(confusion_metrics_plot_path + "/stateless_airway_confusion_matrix.png")
    else:
        airway_plot_path = pathlib.Path(confusion_metrics_plot_path + "/stateful_airway_confusion_matrix.png")

    # Predictions og targets er ikke onehote encoda
    # Predictions består av en liste med 9 elementer, hvert element er en video
    # Hvert element har 2 lister, en for airway og en for direction for videoen
    # Samme gjelder targets
    # Så predictions[0] = video 0 og matcher med targets[0]
    all_predictions_airway = []
    all_targets_airway = []

    for i in range(len(predictions)):
        video_predictions = predictions[i]
        video_targets = targets[i]
        all_predictions_airway += video_predictions
        all_targets_airway += video_targets

    confusion_metrics_airway = confusion_matrix(all_targets_airway, all_predictions_airway, labels=list(range(1, num_airway_classes + 1)))

    confusion_metrics_airway = ConfusionMatrixDisplay(confusion_matrix=confusion_metrics_airway,
                                                      display_labels=list(range(1, num_airway_classes + 1)))
    fig, ax = plt.subplots(figsize=(20, 20))

    plt.title(f"Confusion Metrics for Airway Segment Classes")
    confusion_metrics_airway.plot(ax=ax)
    plt.savefig(airway_plot_path)
    plt.show()


def plot_confusion_metrics_with_direction(predictions, targets, confusion_metrics_plot_path, num_airway_classes, num_direction_classes, stateful_testing=True):
    plot_directory_path = pathlib.Path(confusion_metrics_plot_path)
    plot_directory_path.mkdir(exist_ok=True)

    colors = []

    if not stateful_testing:
        airway_plot_path = pathlib.Path(confusion_metrics_plot_path + "/stateless_airway_confusion_matrix.png")
        direction_plot_path = pathlib.Path(confusion_metrics_plot_path + "/stateless_direction_confusion_matrix.png")
    else:
        airway_plot_path = pathlib.Path(confusion_metrics_plot_path + "/stateful_airway_confusion_matrix.png")
        direction_plot_path = pathlib.Path(confusion_metrics_plot_path + "/stateful_direction_confusion_matrix.png")

    # Predictions og targets er ikke onehote encoda
    # Predictions består av en liste med 9 elementer, hvert element er en video
    # Hvert element har 2 lister, en for airway og en for direction for videoen
    # Samme gjelder targets
    # Så predictions[0] = video 0 og matcher med targets[0]
    all_predictions_airway = []
    all_predictions_direction = []
    all_targets_airway = []
    all_targets_direction = []

    for i in range(len(predictions)):
        video_predictions = predictions[i]
        video_targets = targets[i]
        all_predictions_airway += video_predictions[0]
        all_targets_airway += video_targets[0]
        all_predictions_direction += video_predictions[1]
        all_targets_direction += video_targets[1]

    confusion_metrics_airway = confusion_matrix(all_targets_airway, all_predictions_airway, labels=list(range(1, num_airway_classes + 1)))
    confusion_metrics_direction = confusion_matrix(all_targets_direction, all_predictions_direction, labels=list(range(0, num_direction_classes)))

    confusion_metrics_airway = ConfusionMatrixDisplay(confusion_matrix=confusion_metrics_airway,
                                                      display_labels=list(range(1, num_airway_classes + 1)))
    fig, ax = plt.subplots(figsize=(20, 20))

    plt.title(f"Confusion Metrics for Airway Segment Classes")
    confusion_metrics_airway.plot(ax=ax)
    plt.savefig(airway_plot_path)
    plt.show()

    confusion_metrics_direction = ConfusionMatrixDisplay(confusion_matrix=confusion_metrics_direction,
                                                         display_labels=list(range(0, num_direction_classes)))
    fig, ax = plt.subplots(figsize=(20, 20))

    plt.title(f"Confusion Metrics for Direction Classes")
    confusion_metrics_direction.plot(ax=ax)
    plt.savefig(direction_plot_path)
    plt.show()


def test_model(trainer, test_dataset, test_slide_ratio, num_frames_in_test_stack, num_airway_classes, num_direction_classes, data_path,
               frame_dimension, model_name, model_path, test_plot_path, model_type, inference_device, airway_labels, direction_labels,
               local_test_data_path, local_trained_model_path):

    print("-- TESTING --")
    # Load neural net model
    print("Loading best model for: ", model_name)
    trainer.load_model(inference_device)

    # Set to inference mode -> freeze model
    torch.manual_seed(42)
    trainer.model.eval()

    # Test pipeline for Baseline: No LSTM or direction
    if model_type == 'baseline':
        # Convert to onnx
        convert_model_to_onnx_for_baseline(model=trainer.model,
                                           num_frames_in_test_stack=num_frames_in_test_stack,
                                           dimension=frame_dimension,
                                           model_name=model_name,
                                           model_path=model_path)
        # Run predictions on test set
        predictions, targets = get_test_set_predictions_for_baseline(model=trainer.model,
                                                                     test_dataset=test_dataset,
                                                                     test_slide_ratio=test_slide_ratio,
                                                                     num_frames_in_test_stack=num_frames_in_test_stack,
                                                                     data_path=data_path,
                                                                     model_name=model_name)
        # Get F1 Macro Score, Precision and Recall
        get_metrics(predictions=predictions,
                    targets=targets,
                    num_airway_classes=num_airway_classes)

        # Plot Confusion Metrics
        plot_confusion_metrics(predictions=predictions,
                               targets=targets,
                               confusion_metrics_plot_path=test_plot_path,
                               num_airway_classes=num_airway_classes)

    # Test pipeline for Blomst: LSTM
    elif model_type == "blomst":
        convert_model_to_onnx(model=trainer.model,
                              num_frames_in_test_stack=num_frames_in_test_stack,
                              dimension=frame_dimension,
                              model_name=model_name,
                              model_path=model_path)

        # STATEFUL: Run predictions on test set
        print(" -- STATEFUL TESTING --")
        stateful_predictions, targets = get_test_set_predictions(model=trainer.model,
                                                                 test_dataset=test_dataset,
                                                                 test_slide_ratio=test_slide_ratio,
                                                                 num_frames_in_test_stack=num_frames_in_test_stack,
                                                                 data_path=data_path,
                                                                 model_name=model_name,
                                                                 stateful_testing=True)

        # Get F1 Macro Score, Precision and Recall
        get_metrics(predictions=stateful_predictions,
                    targets=targets,
                    num_airway_classes=num_airway_classes)

        # Plot Confusion Metrics
        plot_confusion_metrics(predictions=stateful_predictions,
                               targets=targets,
                               confusion_metrics_plot_path=test_plot_path,
                               num_airway_classes=num_airway_classes,
                               stateful_testing=True)

        # STATELESS: Run predictions on test set
        print(" -- STATELESS TESTING --")
        stateless_predictions, targets = get_test_set_predictions(model=trainer.model,
                                                                  test_dataset=test_dataset,
                                                                  test_slide_ratio=test_slide_ratio,
                                                                  num_frames_in_test_stack=num_frames_in_test_stack,
                                                                  data_path=data_path,
                                                                  model_name=model_name,
                                                                  stateful_testing=False)

        # Get F1 Macro Score, Precision and Recall
        get_metrics(predictions=stateless_predictions,
                    targets=targets,
                    num_airway_classes=num_airway_classes)

        # Plot Confusion Metrics
        plot_confusion_metrics(predictions=stateless_predictions,
                               targets=targets,
                               confusion_metrics_plot_path=test_plot_path,
                               num_airway_classes=num_airway_classes,
                               stateful_testing=False)

    # Test pipeline for Boble or Belle: LSTM and direction
    elif model_type == "boble" or model_type == "belle":
        print(" -- STATEFUL TESTING --")
        # Convert to onnx
        # TODO: Return onnx model og bruk den for testing fordi den støtter dynamics shapes
        convert_model_to_onnx_with_direction(model=trainer.model,
                                             num_frames_in_test_stack=num_frames_in_test_stack,
                                             dimension=frame_dimension,
                                             model_name=model_name,
                                             model_path=model_path)
        # Run predictions on test set
        stateful_predictions, targets = get_test_set_predictions_with_direction(model=trainer.model,
                                                                                test_dataset=test_dataset,
                                                                                test_slide_ratio=test_slide_ratio,
                                                                                num_frames_in_test_stack=num_frames_in_test_stack,
                                                                                data_path=data_path,
                                                                                model_name=model_name,
                                                                                stateful_testing=True)

        # Get F1 Macro Score, Precision and Recall
        get_metrics_with_direction(predictions=stateful_predictions,
                                   targets=targets,
                                   num_airway_classes=num_airway_classes,
                                   num_direction_classes=num_direction_classes)

        # Plot Confusion Metrics
        plot_confusion_metrics_with_direction(predictions=stateful_predictions,
                                              targets=targets,
                                              confusion_metrics_plot_path=test_plot_path,
                                              num_airway_classes=num_airway_classes,
                                              num_direction_classes=num_direction_classes,
                                              stateful_testing=True)

        print(" -- STATELESS TESTING --")
        # Run predictions on test set
        stateless_predictions, targets = get_test_set_predictions_with_direction(model=trainer.model,
                                                                                 test_dataset=test_dataset,
                                                                                 test_slide_ratio=test_slide_ratio,
                                                                                 num_frames_in_test_stack=num_frames_in_test_stack,
                                                                                 data_path=data_path,
                                                                                 model_name=model_name,
                                                                                 stateful_testing=False)

        # Get F1 Macro Score, Precision and Recall
        get_metrics_with_direction(predictions=stateless_predictions,
                                   targets=targets,
                                   num_airway_classes=num_airway_classes,
                                   num_direction_classes=num_direction_classes)

        # Plot Confusion Metrics
        plot_confusion_metrics_with_direction(predictions=stateless_predictions,
                                              targets=targets,
                                              confusion_metrics_plot_path=test_plot_path,
                                              num_airway_classes=num_airway_classes,
                                              num_direction_classes=num_direction_classes,
                                              stateful_testing=False)
        """ 
        # Run inference testing with pyfast
        test_videos = list(os.listdir(local_test_data_path))
        for test_video in test_videos:
            test_video_path = local_test_data_path + f"/{test_video}/frame_#.png"
            run_testing_realtime(airway_labels=airway_labels,
                                 direction_labels=direction_labels,
                                 data_path=test_video_path,
                                 num_frames_in_test_stack=num_frames_in_test_stack,
                                 trained_model_path=local_trained_model_path)
        """