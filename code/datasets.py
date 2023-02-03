import torch
from preprocess import *
from torch.utils.data import Dataset, sampler
from skimage import io
import random
from parameters import dataset_path, test_split, root_directory_path, \
    num_bronchus_generations, batch_size, validation_split, dataset_type
from torchvision import transforms
from pathlib import PurePath


class BronchusDataset(Dataset):
    """ The dataset class """
    def __init__(self, csv_file, root_directory, num_bronchus_generations, transform=None):
        # CSV file containing 2 columns: frame_path and label
        self.labeled_frames = pd.read_csv(csv_file, index_col=False)
        self.root_directory = root_directory
        self.num_generations = num_bronchus_generations
        self.transform = transform

        # Mapping from original number of labels to X labels given self.num_generations
        self.label_mapping = self.get_label_mapping()
        self.keys = list(self.label_mapping.keys())

    def __len__(self):
        return len(self.labeled_frames)

    def __getitem__(self, index):
        """
        Enables the fetching of values with dataset[index] in the dataset
        """
        if torch.is_tensor(index):
            index = index.tolist()

       # Fetch columns from the csv for the given index
        frame_name = self.labeled_frames.iloc[index, 0]

        # Frame
        frame = io.imread(frame_name)

        if self.transform:
            frame = self.transform(frame)

        # Get original airway label
        original_airway_label = self.labeled_frames.iloc[index, 1]
        num_classes = len(list(self.label_mapping.keys()))+1
        airway_label = torch.nn.functional.one_hot(torch.tensor(original_airway_label), num_classes=num_classes)

        # Get direction label
        direction_label = self.labeled_frames.iloc[index, 2]
        direction_label = torch.nn.functional.one_hot(torch.tensor(direction_label), num_classes=2)

        return frame, airway_label.float(), direction_label.float()

    def perform_mapping(self, label):
        # Label is >= num_generations
        if label in self.keys:
            new_label = self.label_mapping[label]
        # Label is > num_generations
        else:
            new_label = 0
        return new_label

    def relabel_dataset(self):
        """
        Method that relabels the dataset based on a mapping created after the numbers of generations entered.
        Method also deleted frames that has a larger generation than the one entered
        """
        # Change label by using the given mapping system
        self.labeled_frames["Airway_Segment"] = self.labeled_frames["Airway_Segment"].apply(lambda label: self.perform_mapping(label))

        # Remove frames not belonging to the generation chosen (labeled with 0)
        self.labeled_frames = self.labeled_frames[(self.labeled_frames.Label != 0)]

    def get_label_mapping(self):
        """
        Return a dictionary {"old originally label": "new label reducing class numbers"}
        """

        if self.num_generations == 1:
            label_mapping = {
                1: 1,  # Trachea
                5: 2,  # Right Main Bronchus
                4: 3,  # Left Main Bronchus
            }

        elif self.num_generations == 2:
            label_mapping = {
                1:  1,   # Trachea
                5:  2,   # Right Main Bronchus
                4:  3,   # Left Main Bronchus
                14: 4,  # Right/Left Upper Lobe Bronchus
                15: 5,  # Right Truncus Intermedicus
                12: 6,  # Left Lower Lobe Bronchus
                13: 7   # Left Upper Lobe Bronchus
            }

        elif self.num_generations == 3:
            label_mapping = {
                1:   1,   # Trachea
                5:   2,   # Right Main Bronchus
                4:   3,   # Left Main Bronchus
                14:  4,   # Right/Left Upper Lobe Bronchus
                15:  5,   # Right Truncus Intermedicus
                12:  6,   # Left Lower Lobe Bronchus
                13:  7,   # Left Upper Lobe Bronchus
                49:  8,   # Right B1
                50:  9,   # Right B2
                48:  10,  # Right B3
                2:   11,  # Right Middle Lobe Bronchus (parent for B4 og B5)
                3:   12,  # Right lower Lobe Bronchus (possible called right lower lobe bronchus (1))
                11:  13,  # Right Lower Lobe Bronchus (2)
                39:  14,  # Left Main Bronchus
                38:  15,  # Left B6
                42:  16,  # Left Upper Division Bronchus
                43:  17,  # Left Lingular Bronchus (or singular?)
            }

        elif self.num_generations == 4:
            label_mapping = {
                1: 1,  # Trachea
                5: 2,  # Right Main Bronchus
                4: 3,  # Left Main Bronchus
                14: 4,  # Right/Left Upper Lobe Bronchus
                15: 5,  # Right Truncus Intermedicus
                12: 6,  # Left Lower Lobe Bronchus
                13: 7,  # Left Upper Lobe Bronchus
                49: 8,  # Right B1
                50: 9,  # Right B2
                48: 10,  # Right B3
                2: 11,  # Right Middle Lobe Bronchus (parent for B4 og B5)
                3: 12,  # Right lower Lobe Bronchus (possible called right lower lobe bronchus (1))
                11: 13,  # Right Lower Lobe Bronchus (2)
                39: 14,  # Left Main Bronchus
                38: 15,  # Left B6
                42: 16,  # Left Upper Division Bronchus
                43: 17,  # Left Lingular Bronchus (or singular?)
                7:  18,  # Right B4
                6:  19,  # Right B5
                91: 20,  # Left B1+B2
                90: 21,  # Left B3
                40: 22,  # Left B4
                41: 23,  # Left B5
                82: 24,  # Left B8
                37: 25,  # Left B9
                36: 26,  # Left B10
            }

        else:
            label_mapping = None
            print("Did not find the number of bronchus generations")

        return label_mapping

    def get_num_classes(self):
        return len(list(self.label_mapping.keys()))+1  # Num defined classes + one class:0 for all undefined classes


def move_location(videos, new_location):
    for video in videos:
        temp_dataframe = pd.read_csv(raw_dataset_path + "/" + video)
        new_path = PurePath(new_location, video)
        temp_dataframe.to_csv(new_path, index=False)


def split_data(validation_split, test_split, raw_dataset_path, dataset_path):
    """
    Splits the given raw_data into datasets/train, datasets/test or datasets/validation.
    A video is represented as a csv file with all its frames and belonging labels
     """
    # Create a list of all videos given
    raw_data = os.listdir(raw_dataset_path)
    videos = list(filter(lambda x: not x.startswith("."), raw_data))

    # Get num videos for train, test and validation
    num_videos = len(videos)
    num_validation_videos = int(validation_split * num_videos)
    num_test_videos = int(test_split * num_videos)

    # Get random videos for the validation set
    validation_videos = [str(np.random.choice(videos, replace=False)) for i in range(num_validation_videos)]

    # Get all videos NOT in the validation set
    temp_videos = [index for index in videos if index not in validation_videos]

    # Get random videos for the test set
    test_videos = [str(np.random.choice(temp_videos, replace=False)) for i in
                                range(num_test_videos)]

    # Get the remaining videos for training (NOT in test -> NOT in validation)
    train_videos = [index for index in temp_videos if index not in test_videos]

    print("Validation")
    move_location(validation_videos, dataset_path + "validation/")
    print("Test")
    move_location(test_videos, dataset_path + "test/")
    print("Train")
    move_location(train_videos, dataset_path + "train/")


def create_datasets_and_dataloaders():
    # Split data in Train, test and validation
    split_data(validation_split, test_split, raw_dataset_path, dataset_path)


create_datasets_and_dataloaders()
