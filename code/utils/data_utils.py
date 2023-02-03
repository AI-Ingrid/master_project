import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import torch
from tqdm import tqdm
import numpy as np
import re


def get_class_distribution_for_batch(y_batch, count):
    for label in y_batch:
        decoded_label = np.argmax(label)
        if not count:
            count[int(decoded_label)] = 1
        else:
            labels = list(count.keys())
            if decoded_label not in labels:
                count[int(decoded_label)] = 1

            else:
                count[int(decoded_label)] += 1
    return count


def get_class_distribution(dataloader):
    count = {}
    for x_batch, y_batch in dataloader:
        count = get_class_distribution_for_batch(y_batch, count)
    return count


def plot_dataset_distribution(fps, train, validation=None, test=None):
    plt.figure()
    train_dist = get_class_distribution(train)
    x_axis = list(train_dist.keys())
    type = "None"

    # Train
    if validation is None and test is None:
        y_train = list(train_dist.values())
        plt.bar(x_axis, y_train, label="Train")
        type = "train"

    # Validation
    elif validation is not None and test is None:
        validation_dist = get_class_distribution(validation)
        y_validation = list(validation_dist.values())
        while len(y_validation) != 26:
            y_validation.append(0)
        plt.bar(x_axis, y_validation, label="Validation")
        type = "validation"

    # Test
    elif validation is None and test is not None:
        test_dist = get_class_distribution(test)
        y_test = list(test_dist.values())
        while len(y_test) != 26:
            y_test.append(0)
        plt.bar(x_axis, y_test, label="Test")
        type = "test"

    print("Plotting: ", type)
    plt.title(f'Distribution in SegmentDetNet with {fps} FPS for the {type} data set')
    plt.xlabel('Labels')
    plt.ylabel('Number of examples')
    plt.legend()
    plt.savefig(f"6_{type}_distribution_{fps}_fps.png")


def compute_mean_std(dataset, dataloader, frame_dim):
    # placeholders
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for datatype in dataloader:
        for X_batch, Y_batch in tqdm(datatype):
            # psum += X_batch.sum(axis=[0, 2, 3])
            psum += X_batch.sum(axis=[0, 1, 2, 3])
            # psum_sq += (X_batch ** 2).sum(axis=[0, 2, 3])
            psum_sq += (X_batch ** 2).sum(axis=[0, 1, 2, 3])

    # pixel count
    count = len(dataset) * frame_dim[0] * frame_dim[1]

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    # output
    print('mean: ' + str(total_mean))
    print('std:  ' + str(total_std))
    return total_mean, total_std


def get_label_name(label):
    label_names = {
        1: "Trachea",
        2: "Right Main Bronchus",
        3: "Left Main Bronchus",
        4: "Right/Left Upper Lobe Bronchus",
        5: "Right Truncus Intermedicus",
        6: "Left Lower Lobe Bronchus",
        7: "Left Upper Lobe Bronchus",
        8: "Right B1",
        9: "Right B2",
        10: "Right B3",
        11: "Right Middle Lobe Bronchus 2",
        12: "Right Lower Lobe Bronchus 1",
        13: "Right Lower Lobe Bronchus 2",
        14: "Left Main Bronchus",
        15: "Left B6",
        26: "Left Upper Division Bronchus",
        27: "Left Singular Bronchus",
    }
    if label not in list(label_names.keys()):
        name = label
    else:
        name = label_names[label]
    return name


def get_trim_start_end_frames(trim_time, fps, nb_frames):
    """

    Parameters
    ----------
    trim_time : VideoTrimmingLimits
        The start (t1) and end (t2) times to trim the video
    fps : int/float
        The frame rate (frames per second) of the video
    nb_frames : int
        The total number of frames in the video sequence
    """

    start_frame = round(trim_time.t1 * fps)
    end_frame = round(trim_time.t2 * fps)
    if not (0 <= start_frame < nb_frames and 0 < end_frame <= nb_frames):
        raise ValueError(f"Error: start_frame or end_frame is outside of range [0, nb_frames]")
    if start_frame >= end_frame:
        raise ValueError(f"Error: start_frame {start_frame} >= end_frame {end_frame}")
    return start_frame, end_frame


class VideoTrimmingLimits(object):
    """
    Simple class to keep two time points that belong together.
    Example: Use for specifying start/end frames of parts of a video
    """

    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.__class__.__name__}( t1={self.t1}, t2={self.t2} )"


def find_first_free_idx(arr):
    """
    Find the next unused integer in a sequence (e.g. 003 if indices 001 and 002
    are already used)

    Parameters
    ----------
    arr : array-like
        An array containing 'used' indices in a sequence
    """

    # Prepend the array with -1 and 0 so np.diff works
    arr = np.concatenate((np.array([-1, 0], dtype=np.int), np.array(arr, dtype=np.int)))
    # Compute differences between consecutive values
    where_sequence_breaks = np.where(np.diff(arr) > 1)[0]

    # If there ary any "holes" return the id of the first one
    if len(where_sequence_breaks) > 0:
        return where_sequence_breaks[0]
    # Otherwise return the integer succeeding the last element.
    else:
        return arr[-1] + 1


def find_next_folder_nbr(dataset_dir):
    """
    In a folder of numbered folders (e.g. Patient_001,... or Sequence_001,...),
    find the next integer that has not yet been used for a folder.

    Example: If the folders in the path-like object 'dataset_dir' are labelled
        'Patient_001', 'Patient_002', and 'Patient_001', the function will
        return 4 (the next integer in the sequence).

    Parameters
    ----------
    dataset_dir : str, or path-like
        Path to a folder containing numbered subfolders
    """
    # List subfolders in the folder
    subfolders = [p for p in os.listdir(dataset_dir) if not p.startswith('.')]
    if len(subfolders) == 0:
        return 1

    # Find highest subfolder number in folder
    subfolders = [int(folder.split('_')[-1]) for folder in subfolders]
    subfolders.sort()
    next_free_integer = find_first_free_idx(subfolders)
    return next_free_integer


# The function: atoi and natural_keys are taken from Stackoverflow in order to sort a
# string that contains integers the correct way
def atoi(text):
    """
    Code from Stackoverflow: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    """
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    Code from Stackoverflow: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]
