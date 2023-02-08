"""
File for setting all parameters for the master project
"""
# Preprocessing
frame_dimension = (512, 512)  # Dimension of the scaled frames that will be sent into CNN
fps = 20  # Sampling frequency for video to frames

convert_videos = False  # Convert to frames or not
# Data Augmentation


# Datasets
dataset_type = 'synthetic'  # {'synthetic' 'human'}
test_split = 0.2  # Fraction that split data into test data
validation_split = 0.1  # Fraction that split data into validation data
num_frames_in_stack = 30  # Num frames in a stack that gets sent into RNN
slide_ratio_in_stack = 5  # Ratio of slide between frames in a stack TODO: Vurdere en variabel her som feks random mellom 3- 7
num_stacks = 1000

data_is_split = True  # True means data is already  split into train, test or validation, false means perform split


# Neural nets details
num_classes = 27
hidden_nodes = 128

# Training specifications
epochs = 20
batch_size = 32
learning_rate = 7e-5
early_stop_count = 5
alpha = 0.25
gamma = 2.0

load_best_model = True

# Testing trained model
get_data_dist = True
get_loss_and_accuracy = True
get_confusion_metrics = True
get_testset_pred = True
get_f1_score = True

# Data paths
#root_directory_path = f"/cluster/home/ingrikol/master/data/{dataset_type}"    #  IDUN
root_directory_path = "/Users/ikolderu/PycharmProjects/master/data/synthetic"  # LOCAL
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
