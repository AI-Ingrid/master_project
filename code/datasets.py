from preprocess import *
from utils.data_utils import create_stack_dict, plot_stack
from torch.utils.data import Dataset, DataLoader
from pathlib import PurePath
import torch
import cv2
from torchvision import transforms


def separate_dataframe(video_df, transform, num_airway_segment_classes, num_direction_classes, is_test_dataset=False):
    frame_paths = video_df["Frame"]
    airway_labels = video_df["Airway_Segment"]
    direction_labels = video_df["Direction"]

    frames = []
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        # Transform frame
        frame = transform(frame)
        frames.append(np.asarray(frame))

    # Convert python list to array
    new_frames = np.array(frames)

    # Convert to Tensor
    frames = torch.tensor(new_frames)  # [num_frames=5, channels=3, width=384, height=384]

    # Not one-hot encode the labels for the test set
    if is_test_dataset:
        airway_labels = torch.tensor(airway_labels.values)  # Values from 1- 26
        direction_labels = torch.tensor(direction_labels.values)

    else:
        # Shift all ground truth values one value lower, go from labels 1-26 -> 0-25 (for one-hot-encoding)
        airway_labels = np.array(list(airway_labels.values))
        airway_labels = airway_labels - 1

        # One-hot encode the labels for training and validation dataset
        airway_labels = torch.nn.functional.one_hot(torch.tensor(airway_labels),
                                                    num_classes=num_airway_segment_classes)
        direction_labels = torch.nn.functional.one_hot(torch.tensor(direction_labels.values),
                                                       num_classes=num_direction_classes)
    return frames, airway_labels.float(), direction_labels.float()

class RandomGeneratorDataset(Dataset):
    """ Dataset class for the lung airway net that generates
    random stacks consisting of 'num_frames_in_stack' frames with
    a 'slide_ratio_in_stack' ratio between each frame for a random video """

    def __init__(self, file_list, num_stacks, num_frames, slide_ratio, num_airway_classes, num_direction_classes,
                 transform):
        self.file_list = file_list
        self.num_stacks = num_stacks
        self.num_frames = num_frames
        self.slide_ratio = slide_ratio
        self.num_airway_classes = num_airway_classes
        self.num_direction_classes = num_direction_classes
        self.transform = transform

    def __len__(self):
        return self.num_stacks

    def __getitem__(self, item):
        # Get random video
        video = np.random.choice(self.file_list, size=1)[0]

        # Read the video file
        video_df = pd.read_csv(video)

        # Sample frames from video with a given slide ratio
        possible_frames = video_df[::self.slide_ratio]
        possible_start_frames = possible_frames[:-self.num_frames]

        # Get a random start frame for the stack
        start_frame_indices = np.array(range(len(possible_start_frames)))
        start_index = np.random.choice(start_frame_indices)

        # Create a stack from given start frame index
        stack_df = possible_frames.iloc[start_index: (start_index + self.num_frames)]

        # Separate the stack's data into frames, airway labels and direction labels
        frames, airway_labels, direction_labels = separate_dataframe(stack_df, self.transform, self.num_airway_classes,
                                                                     self.num_direction_classes)

        return frames, [airway_labels, direction_labels]


class StackGeneratorDataset(Dataset):
    """ Dataset class for the lung airway net that generates
       chronological stack sequences for videos consisting of 'num_frames_in_stack' frames with
       a 'slide_ratio_in_stack' ratio between each frame for a random video """

    def __init__(self, file_list, stack_size, slide_ratio, num_airway_classes, num_direction_classes, transform):
        self.file_list = file_list
        self.stack_size = stack_size
        self.slide_ratio = slide_ratio
        self.num_airway_classes = num_airway_classes
        self.num_direction_classes = num_direction_classes
        self.transform = transform
        self.total_stacks, self.stack_dict, self.overlap_dict = create_stack_dict(self.file_list, self.stack_size,
                                                                                  self.slide_ratio)
    def __len__(self):
        return self.total_stacks

    def __getitem__(self, item):
        # Fetch video and the stack number for that video
        video_path = self.stack_dict[item]["video_path"]
        local_stack_number = self.stack_dict[item]["local_stack_number"]

        # Read the video file
        video_df = pd.read_csv(video_path)

        # Get the stack indices
        start_index = local_stack_number * (self.stack_size * self.slide_ratio)
        end_index = start_index + (self.stack_size * self.slide_ratio) - 1

        # Handle stack edge case to make sure every stack has length num_frames
        num_left_over_frames = (self.stack_size * self.slide_ratio) - len(video_df) % (
                    self.stack_size * self.slide_ratio)
        last_frame = video_df.tail(1)

        for i in range(num_left_over_frames):
            video_df = video_df.append(last_frame, ignore_index=True)

        stack_df = video_df[start_index: end_index: self.slide_ratio]

        # Separate the stack's data into frames, airway labels and direction labels
        frames, airway_labels, direction_labels = separate_dataframe(stack_df, self.transform, self.num_airway_classes,
                                                                     self.num_direction_classes)
        return frames, [airway_labels, direction_labels]


class TestDataSet(Dataset):
    def __init__(self, file_list, transform, num_airway_classes, num_direction_classes):
        self.file_list = file_list
        self.transform = transform
        self.num_airway_classes = num_airway_classes
        self.num_direction_classes = num_direction_classes

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        # Get random video
        video = self.file_list[item]

        # Read the video file
        video_df = pd.read_csv(video)

        # Separate the stack's data into frames, airway labels and direction labels
        frames, airway_labels, direction_labels = separate_dataframe(video_df, self.transform, self.num_airway_classes,
                                                                     self.num_direction_classes, is_test_dataset=True)
        return frames, [airway_labels, direction_labels]


def move_location(videos, new_location, raw_dataset_path):
    for video in videos:
        temp_dataframe = pd.read_csv(raw_dataset_path + "/" + video)
        new_path = PurePath(new_location, video)
        temp_dataframe.to_csv(new_path, index=False)


def split_data(validation_split, test_split, raw_dataset_path, dataset_path, split_the_data):
    """
    Splits the given raw_data into datasets/train, datasets/test or datasets/validation.
    A video is represented as a csv file with all its frames and belonging labels
     """
    new_validation_path = dataset_path + "validation"
    new_test_path = dataset_path + "test"
    new_train_path = dataset_path + "train"

    # Split the videos into training, testing and validation videos
    if split_the_data:
        # Create a list of all videos given
        raw_data = os.listdir(raw_dataset_path)

        # Remove .DS_Store
        videos = list(filter(lambda x: not x.startswith("."), raw_data))

        # Get num videos for train, test and validation
        num_videos = len(videos)
        num_validation_videos = int(validation_split * num_videos)
        num_test_videos = int(test_split * num_videos)

        # Get random videos for the validation set
        validation_videos_paths = [str(np.random.choice(videos, replace=False)) for _ in range(num_validation_videos)]

        # Get all videos NOT in the validation set
        temp_videos_paths = [index for index in videos if index not in validation_videos_paths]

        # Get random videos for the test set
        test_videos_paths = [str(np.random.choice(temp_videos_paths, replace=False)) for _ in
                             range(num_test_videos)]

        # Get the remaining videos for training (NOT in test -> NOT in validation)
        train_videos_paths = [index for index in temp_videos_paths if index not in test_videos_paths]

        # Move videos to correct folder: train, test or validation
        move_location(validation_videos_paths, new_validation_path + "/", raw_dataset_path)
        move_location(test_videos_paths, new_test_path + "/", raw_dataset_path)
        move_location(train_videos_paths, new_train_path + "/", raw_dataset_path)

    # Store the absolute paths
    validation_videos_paths = [os.path.join(new_validation_path, file) for file in
                               list(filter(lambda x: not x.startswith("."), os.listdir(new_validation_path)))]
    test_videos_paths = [os.path.join(new_test_path, file) for file in
                         list(filter(lambda x: not x.startswith("."), os.listdir(new_test_path)))]
    train_videos_paths = [os.path.join(new_train_path, file) for file in
                          list(filter(lambda x: not x.startswith("."), os.listdir(new_train_path)))]

    # Return a list with filenames for each folder: train, test and validation
    return train_videos_paths, test_videos_paths, validation_videos_paths


def create_datasets_and_dataloaders(validation_split, test_split, raw_dataset_path, dataset_path,
                                    num_stacks, num_frames_in_stack, slide_ratio_in_stack,
                                    test_slide_ratio_in_stack, batch_size,
                                    split_the_data, num_airway_segment_classes, num_direction_classes,
                                    frame_dimension, use_test_dataloader, plot_dataset_stacks,
                                    use_random_stack_generator):
    print("-- DATASETS --")
    # Split data in Train, Test and Validation
    train_csv_files, test_csv_files, validation_csv_files = split_data(validation_split=validation_split,
                                                                    test_split=test_split,
                                                                    raw_dataset_path=raw_dataset_path,
                                                                    dataset_path=dataset_path,
                                                                    split_the_data=split_the_data)

    # Set up transforming details for the dataset to output
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(frame_dimension),
                                    transforms.ToTensor(),])

    # Use datasets with random stacks from random videos - enables sliding window
    if use_random_stack_generator:
        train_dataset = RandomGeneratorDataset(file_list=train_csv_files,
                                               num_stacks=num_stacks,
                                               num_frames=num_frames_in_stack,
                                               slide_ratio=slide_ratio_in_stack,
                                               num_airway_classes=num_airway_segment_classes,
                                               num_direction_classes=num_direction_classes,
                                               transform=transform,)

        validation_dataset = RandomGeneratorDataset(file_list=train_csv_files,
                                                    num_stacks=num_stacks,
                                                    num_frames=num_frames_in_stack,
                                                    slide_ratio=slide_ratio_in_stack,
                                                    num_airway_classes=num_airway_segment_classes,
                                                    num_direction_classes=num_direction_classes,
                                                    transform=transform,)

    # Use dataset with given stacks - no sliding window
    else:
        train_dataset = StackGeneratorDataset(file_list=train_csv_files,
                                              stack_size=num_frames_in_stack,
                                              slide_ratio=slide_ratio_in_stack,
                                              num_airway_classes=num_airway_segment_classes,
                                              num_direction_classes=num_direction_classes,
                                              transform=transform)

        validation_dataset = StackGeneratorDataset(file_list=validation_csv_files,
                                                   stack_size=num_frames_in_stack,
                                                   slide_ratio=slide_ratio_in_stack,
                                                   num_airway_classes=num_airway_segment_classes,
                                                   num_direction_classes=num_direction_classes,
                                                   transform=transform)

    # Create dataloaders for train and validation datasets
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)


    validation_dataloader = DataLoader(dataset=validation_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=True)

    # Create Test Dataset
    if use_test_dataloader:
        test_dataloader = TestDataSet(file_list=test_csv_files,
                                      transform=transform,
                                      num_airway_classes=num_airway_segment_classes,
                                      num_direction_classes=num_direction_classes)

    else:
        test_dataset = StackGeneratorDataset(file_list=test_csv_files,
                                             stack_size=num_frames_in_stack,
                                             slide_ratio=test_slide_ratio_in_stack,
                                             num_airway_classes=num_airway_segment_classes,
                                             num_direction_classes=num_direction_classes,
                                             transform=transform)

        test_dataloader = DataLoader(dataset=test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     drop_last=True)

    # Plot images of stacks from the datasets
    if plot_dataset_stacks:
        plot_stack(train_dataloader)
        plot_stack(validation_dataloader)
        plot_stack(test_dataloader)

    return train_dataloader, validation_dataloader, test_dataloader
