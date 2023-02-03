"""
File for setting parameters for the project
"""
import torch

# Preprocessing and datasets
dataset_type = 'synthetic'  # {'synthetic' 'human'}
convert_videos = False  # Convert to frames or not
test_split = 0.2
validation_split = 0.1
frame_dimension = (256, 256)  # Dimension of the cropped frames that will be sent into CNN
fps = 10  # {10, 5}

# Neural nets details
num_classes = 27
train_mean = torch.tensor([0.4678, 0.3913, 0.3522])
test_mean = torch.tensor([0.4654, 0.3889, 0.3505])
train_std = torch.tensor([0.2417, 0.1829, 0.1488])
test_std = torch.tensor([0.2422, 0.1825, 0.1487])
num_bronchus_generations = 4  # {1, 2, 3, 4}
hidden_nodes = 128

# Training specifications
epochs = 20
batch_size = 32
learning_rate = 7e-5
early_stop_count = 5
alpha = 0.25
gamma = 2.0
load_best_model = True

# Testing specifications
get_data_dist = True
get_loss_and_accuracy = True
get_confusion_metrics = True
get_testset_pred = True
get_f1_score = True

# Data paths
root_directory_path = f"/cluster/home/ingrikol/master/data/{dataset_type}"
videos_path = root_directory_path + f"/{dataset_type}_videos/"
frames_path = root_directory_path + f"/{dataset_type}_frames"
label_file_path = root_directory_path + f"/labeling_info/{dataset_type}_branches_positions_numbers.txt"
names_file_path = root_directory_path + f"/labeling_info/{dataset_type}_branch_number_name.txt"
raw_dataset_path = root_directory_path + "/raw_data/"
dataset_path = root_directory_path + f"/datasets/"
train_plot_path = "master/plots/training/"
train_plot_name = f"train_{dataset_type}_fps_{fps}"
test_plot_path = "master/plots/testing/"
confusion_metrics_path = f"master/plots/testing/confusion_metrics_{dataset_type}_fps_{fps}"
