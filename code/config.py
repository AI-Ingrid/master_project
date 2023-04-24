"""
File for setting all parameters for the master project. All changes and variables is set here
"""
from datetime import datetime
import os
import torch

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

# Data Augmentation
frame_dimension = (384, 384)  # Dimension of the scaled frames that will be sent into CNN

# Datasets
dataset_type = 'synthetic'  # {'synthetic' 'human'}
test_split = 0.1  # Fraction that split data into test data
validation_split = 0.2  # Fraction that split data into validation data
num_frames_in_stack = 5  # Num frames in a stack that gets sent into RNN
slide_ratio_in_stack = 5  # Ratio of slide between frames in a stack
split_the_data = False  # Split videos into train, test or validation
shuffle_dataset = True

# Neural nets details
num_airway_segment_classes = 27
num_direction_classes = 2
hidden_nodes = 128
num_features = 256
num_LSTM_cells = "None"

# Training specifications
perform_training = False
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0" or "1"
epochs = 1
batch_size = 8
#accum_iter = 4  # batch accumulation parameter
learning_rate = 1e-3
early_stop_count = 10
num_stacks = 1024  # Must be divisible by batch size
num_validations = 1000  # Num times for validation our model during training TODO brukes egt denne mer?
alpha_airway = torch.Tensor([0.2, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1])
alpha_direction = torch.Tensor([1, 1])
gamma = 2.0
test_slide_ratio_in_stack = slide_ratio_in_stack
use_focal_loss = True

# Testing trained model
load_best_model = True
get_data_dist = False
convert_to_onnx = True

# Data paths
root_directory_path = "/cluster/home/ingrikol/master"
data_path = f"{root_directory_path}/data/{dataset_type}"
videos_path = f"{data_path}/{dataset_type}_videos/"
frames_path = f"{data_path}/{dataset_type}_frames"
label_file_path = f"{data_path}/labeling_info/{dataset_type}_branches_positions_numbers.txt"
names_file_path = f"{data_path}/labeling_info/{dataset_type}_branch_number_name.txt"
csv_videos_path = f"{data_path}/videos_csv/"
relabeled_csv_videos_path = f"{data_path}/relabeled_videos_csv/"
dataset_path = f"{data_path}/datasets/"

# Training and Testing paths
model_type = "baseline"
date_and_time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
train_plot_path = f"{root_directory_path}/plots/training/"
train_plot_name = f"{date_and_time}_{dataset_type}_{model_type}_fps_{fps}"
test_plot_path = f"{root_directory_path}/plots/testing/"
confusion_metrics_name = f"{date_and_time}_confusion_metrics_{dataset_type}_{model_type}_fps_{fps}"
model_path = f"{root_directory_path}/models/"
model_name = f"baseline_frames_{num_frames_in_stack}_slide_{slide_ratio_in_stack}_features_{num_features}_hidden_nodes_{hidden_nodes}_LSTM_cells_{num_LSTM_cells}_batch_size_{batch_size}_epochs_{epochs}_focal_loss_{use_focal_loss}"
test_plot_path = f"{root_directory_path}/plots/testing/{model_name}"

#tensorboard --logdir="/cluster/home/ingrikol/master/models/frames_5_slide_10_stacks_1024_features_512_LSTM_cells_1_batchsize_16_epochs_2000_focal_loss_False"
#scp ingrikol@idun-login2.hpc.ntnu.no:"/cluster/home/ingrikol/master/models/onnx/frames_5_slide_10_stacks_1024_features_128_LSTM_cells_1_batchsize_16_epochs_2000_focal_loss_False.onnx" ~