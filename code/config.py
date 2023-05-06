"""
File for setting all parameters for the master project. All changes and variables is set here
"""
from datetime import datetime
import torch
import os

# Preprocessing
fps = 10  # Sampling frequency for video to frames
label_map_dict = {
                1: [1, "Trachea"],
                5: [2, "Right Main Bronchus"],
                4: [3, "Left Main Bronchus"],
                14: [4, "Right/Left Upper Lobe Bronchus"],
                15: [5, "Right Truncus Intermedicus"],
                12: [6, "Left Lower Lobe Bronchus"],
                13: [7,  "Left Upper Lobe Bronchus"],
                49: [8, "Right B1"],
                50: [9,  "Right B2"],
                48: [10,  "Right B3"],
                2: [11,  "Right Middle Lobe Bronchus (parent for B4 og B5)"],
                3: [12, "Right lower Lobe Bronchus (possible called right lower lobe bronchus (1))"],
                11: [13, "Right Lower Lobe Bronchus (2)"],
                39: [14, "Left Main Bronchus"],
                38: [15, "Left B6"],
                42: [16, "Left Upper Division Bronchus"],
                43: [17, "Left Lingular Bronchus"],
                7:  [18, "Right B4"],
                6:  [19, "Right B5"],
                91: [20, "Left B1+B2"],
                90: [21, "Left B3"],
                40: [22, "Left B4"],
                41: [23, "Left B5"],
                82: [24, "Left B8"],
                37: [25, "Left B9"],
                36: [26,  "Left B10"],
            }
convert_videos_to_frames = False  # Convert to frames or not
label_the_frames = False  # Crop, scale and label the frames
relabel_the_frames = False
frame_dimension = (256, 256)  # Dimension of the scaled frames that will be sent into CNN

# Datasets
dataset_type = 'synthetic'  # {'synthetic' 'human'}
use_random_stack_generator = False
test_split = 0.1  # Fraction that split data into test data
validation_split = 0.1  # Fraction that split data into validation data
split_the_data = False  # Split videos into train, test or validation
num_frames_in_stack = 10  # Num frames in a stack that gets sent into RNN
slide_ratio_in_stack = 5  # Ratio of slide between frames in a stack

# Neural nets details
num_airway_segment_classes = 27 # Don't change these
num_direction_classes = 2 # Don't change these
num_features_extracted = 256
num_memory_nodes = 128  # Max = 512
use_stateful_LSTM = False

# Training specifications
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # "0", "1" or "2"
perform_training = True
classify_direction = True
epochs = 1
batch_size = 8
learning_rate = 1e-3 # 1e-4 for å finetune features etter early stopping har kicka inn med Adam i følge andre
early_stop_count = 100
num_stacks = 1024  # Must be divisible by batch size
alpha_airway = torch.Tensor([0.2, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
alpha_direction = torch.Tensor([1, 1])
gamma = 2.0
use_focal_loss = True

# Testing trained model
use_test_dataloader = True
load_best_model = True
convert_to_onnx = True
num_test_frames_in_stack = num_frames_in_stack
test_slide_ratio_in_stack = slide_ratio_in_stack
plot_dataset_stacks = False
inference_device = 'cuda'  # or 'cpu'

# Data paths
root_directory_path = "/mnt/EncryptedPathology/bronchi-navigation/master_project"  # SINTEF PATH
data_path = f"{root_directory_path}/data/{dataset_type}"
videos_path = f"{data_path}/{dataset_type}_videos/"
frames_path = f"{data_path}/{dataset_type}_frames"
label_file_path = f"{data_path}/labeling_info/{dataset_type}_branches_positions_numbers.txt"
names_file_path = f"{data_path}/labeling_info/{dataset_type}_branch_number_name.txt"
csv_videos_path = f"{data_path}/videos_csv/"
relabeled_csv_videos_path = f"{data_path}/relabeled_videos_csv/"
dataset_path = f"{data_path}/datasets/"

# Training and Testing paths
model_type = "blomst"  # {'baseline', 'blomst', 'boble', 'belle'}
date_and_time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
train_plot_path = f"{root_directory_path}/plots/training/"
train_plot_name = f"{date_and_time}_{dataset_type}_{model_type}_fps_{fps}"
test_plot_path = f"{root_directory_path}/plots/testing/"
confusion_metrics_name = f"{date_and_time}_confusion_metrics_{dataset_type}_{model_type}_fps_{fps}"
model_path = f"{root_directory_path}/models/"
model_name = f"{model_type}_{num_frames_in_stack}_{slide_ratio_in_stack}_features_{num_features_extracted}_hidden_{num_memory_nodes}_epochs_{epochs}_focal_loss_{use_focal_loss}_stateful_{use_stateful_LSTM}_direction_{classify_direction}"
test_plot_path = f"{root_directory_path}/plots/testing/{model_name}"
