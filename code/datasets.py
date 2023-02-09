from preprocess import *
from torch.utils.data import Dataset, DataLoader
from pathlib import PurePath
import torch
from skimage import io


class RandomGeneratorDataset(Dataset):
    """ Dataset class for the lung airway net that generates
    random stacks consisting of 'num_frames_in_stack' frames with
    a 'slide_ratio_in_stack' ratio between each frame for a random video """
    def __init__(self, file_list, num_stacks, num_frames, slide_ratio, num_airway_segment_classes, num_direction_classes):
        self.file_list = file_list
        self.num_stacks = num_stacks
        self.num_frames = num_frames
        self.slide_ratio = slide_ratio
        self.num_airway_segment_classes = num_airway_segment_classes
        self.num_direction_classes = num_direction_classes
        self.label_map_dict = {
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
                43: [17, "Left Singular Bronchus"],
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

    def __len__(self):
        return self.num_stacks

    def __getitem__(self, item):
        # Get random video
        video = np.random.choice(self.file_list, size=1)[0]
        video_df = pd.read_csv(video)

        # Get all possible stacks in the video given the parameter conditions
        video_length = len(video_df)
        last_stack_start = video_length - (self.num_frames - 1) * self.slide_ratio
        possible_stack_starts = list(range(0, last_stack_start, self.slide_ratio))

        # Get random start of a stack
        stack_start = np.random.choice(possible_stack_starts, size=1)[0]

        # Get the stack's data
        stack_df = video_df.iloc[stack_start: stack_start + (self.num_frames * self.slide_ratio): self.slide_ratio]

        # Relabel the dataset
        stack_df = self.relabel_airway_segments(stack_df)

        # Distort the stack's data into frames, airway labels and direction labels
        frame_paths = stack_df["Frame"]
        airway_labels = stack_df["Airway_Segment"]
        direction_labels = stack_df["Direction"]

        # Transform
        frames = []
        for frame_path in frame_paths:
            frames.append(io.imread(frame_path))

        frames = torch.tensor(frames)
        airway_labels = torch.nn.functional.one_hot(torch.tensor(airway_labels.values), self.num_airway_segment_classes)
        direction_labels = torch.nn.functional.one_hot(torch.tensor(direction_labels.values), self.num_direction_classes)

        return frames, [airway_labels, direction_labels]

    def perform_mapping(self, label):
        # Label is >= num_generations we are looking at
        if label in self.label_map_dict.keys():
            new_label = self.label_map_dict[label][0]
        # Label is > num_generations we are looking at
        else:
            new_label = 0
        return new_label

    def relabel_airway_segments(self, stack):
        """
        Method that relabels the dataset based on a mapping created after the numbers of generations entered.
        Method also deleted frames that has a larger generation than the one entered
        """
        # Change label by using the given mapping system
        # TODO: Use .iloc, SettingWithCopyWarning
        stack["Airway_Segment"] = stack["Airway_Segment"].apply(lambda label: self.perform_mapping(label))

        # Remove frames not belonging to the generation chosen (labeled with 0)
        stack = stack[(stack.Airway_Segment != 0)]

        return stack


def move_location(videos, new_location, raw_dataset_path):
    for video in videos:
        temp_dataframe = pd.read_csv(raw_dataset_path + "/" + video)
        new_path = PurePath(new_location, video)
        temp_dataframe.to_csv(new_path, index=False)


def split_data(validation_split, test_split, raw_dataset_path, dataset_path, data_is_split):
    """
    Splits the given raw_data into datasets/train, datasets/test or datasets/validation.
    A video is represented as a csv file with all its frames and belonging labels
     """
    new_validation_path = dataset_path + "validation"
    new_test_path = dataset_path + "test"
    new_train_path = dataset_path + "train"

    # Data is not split
    if not data_is_split:
        # Create a list of all videos given
        raw_data = os.listdir(raw_dataset_path)
        videos = list(filter(lambda x: not x.startswith("."), raw_data))

        # Get num videos for train, test and validation
        num_videos = len(videos)
        num_validation_videos = int(validation_split * num_videos)
        num_test_videos = int(test_split * num_videos)

        # Get random videos for the validation set
        validation_videos_paths = [str(np.random.choice(videos, replace=False)) for i in range(num_validation_videos)]

        # Get all videos NOT in the validation set
        temp_videos_paths = [index for index in videos if index not in validation_videos_paths]

        # Get random videos for the test set
        test_videos_paths = [str(np.random.choice(temp_videos_paths, replace=False)) for i in
                                    range(num_test_videos)]

        # Get the remaining videos for training (NOT in test -> NOT in validation)
        train_videos_paths = [index for index in temp_videos_paths if index not in test_videos_paths]

        # Move videos to correct folder: train, test or validation
        move_location(validation_videos_paths, new_validation_path + "/", raw_dataset_path)
        move_location(test_videos_paths, new_test_path + "/", raw_dataset_path)
        move_location(train_videos_paths, new_train_path + "/", raw_dataset_path)

    # Data is already split
    validation_videos_paths = [os.path.join(new_validation_path, file) for file in list(os.listdir(new_validation_path))]
    test_videos_paths = [os.path.join(new_test_path, file) for file in list(os.listdir(new_test_path))]
    train_videos_paths = [os.path.join(new_train_path, file) for file in list(os.listdir(new_train_path))]

    # Return a list with filenames for each folder: train, test and validation
    return train_videos_paths, test_videos_paths, validation_videos_paths


def create_datasets_and_dataloaders(validation_split, test_split, raw_dataset_path, dataset_path,
                                    num_stacks, num_frames_in_stack, slide_ratio_in_stack, batch_size,
                                    shuffle_dataset, data_is_split, num_airway_segment_classes, num_direction_classes):
    # Split data in Train, Test and Validation
    train, test, validation = split_data(validation_split=validation_split, test_split=test_split,
                                         raw_dataset_path=raw_dataset_path, dataset_path=dataset_path,
                                         data_is_split=data_is_split)

    # Create Train Dataset and DataLoader
    train_dataset = RandomGeneratorDataset(file_list=train, num_stacks=num_stacks,
                                           num_frames=num_frames_in_stack, slide_ratio=slide_ratio_in_stack,
                                           num_airway_segment_classes=num_airway_segment_classes,
                                           num_direction_classes=num_direction_classes)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle_dataset)

    # Create Test Dataset and DataLoader
    test_dataset = RandomGeneratorDataset(file_list=test, num_stacks=num_stacks,
                                          num_frames=num_frames_in_stack, slide_ratio=slide_ratio_in_stack,
                                          num_airway_segment_classes=num_airway_segment_classes,
                                          num_direction_classes=num_direction_classes)

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle_dataset)

    # Create Validation DataSet and DataLoader
    validation_dataset = RandomGeneratorDataset(file_list=validation, num_stacks=num_stacks,
                                          num_frames=num_frames_in_stack, slide_ratio=slide_ratio_in_stack,
                                          num_airway_segment_classes=num_airway_segment_classes,
                                          num_direction_classes=num_direction_classes)

    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=shuffle_dataset)

    return train_dataloader, test_dataloader, validation_dataloader
