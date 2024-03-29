import copy
import warnings
# TODO: Handle this
warnings.filterwarnings("ignore")

import os
import math
import imageio
import cv2
import pandas as pd
import numpy as np
import matplotlib
import pathlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.spatial.distance import cdist
from utils.data_utils import get_trim_start_end_frames, VideoTrimmingLimits, find_next_folder_nbr, natural_keys

matplotlib.use('Agg')   # Use 'Agg' backend to avoid memory overload


def convert_video_to_frames(input_data_path, output_data_path, fps):
    """
    Code build upon work from author: Ingrid Tveten (SINTEF)
    """
    output_path = pathlib.Path(output_data_path)
    output_path.mkdir(exist_ok=True)

    extract_frame_interval = 1  # Extract every x frames

    # Hard coded for now
    nb_patients = 1
    files_list = os.listdir(input_data_path)

    # Create list of videos
    video_list = [fn for fn in files_list if
                (fn.lower().endswith('.avi') or fn.lower().endswith('.mpg') or fn.lower().endswith('.mp4'))]

    # Create list of tuples with video, belonging timestamps and positions
    label_list = []
    video_list.sort()
    for video_file in video_list:
        video_name = video_file.split(".")[0]
        timestamps_index = files_list.index(video_name + "_timestamps.txt")
        position_index = files_list.index(video_name + "_positions.txt")
        label_list.append((video_file, files_list[timestamps_index], files_list[position_index]))

    for p in tqdm(range(nb_patients), 'Patient'):
        # Create ./Patient_XX directory
        next_patient_nbr = find_next_folder_nbr(dataset_dir=output_data_path)
        patient_dir = os.path.join(output_data_path, f'Patient_{next_patient_nbr:03d}')
        try:
            os.makedirs(patient_dir, exist_ok=False)
        except OSError as exc:
            print(f"OSError: Patient folder {patient_dir} probably already exists")
            exit(-1)

        videos_for_patient = [fn for fn in label_list]

        # Generate sequences
        for (video_fn, timestamp, position) in tqdm(videos_for_patient, 'Sequences'):

            # Create ./Patient_XX/Sequence_XX directory
            seq_nbr = find_next_folder_nbr(patient_dir)
            seq_dir = os.path.join(patient_dir, f'Sequence_{seq_nbr:03d}')
            try:
                os.makedirs(seq_dir, exist_ok=False)
            except OSError as exc:
                print(f"OSError: Sequence folder {seq_dir} probably already exists")
                exit(-1)

            # Save the timestamps belonging to the video in the correct Sequence folder
            read_timestamp_file = pd.read_csv(input_data_path + "/" + timestamp)
            read_timestamp_file.to_csv(seq_dir + "/" + timestamp, index=None)

            # Save the positions belonging to the video in the correct Sequence folder
            read_positions_file = pd.read_csv(input_data_path + "/" + position)
            read_positions_file.to_csv(seq_dir + "/" + position, index=None)

            # Get full path to video file and read video data
            video_path = os.path.join(input_data_path, video_fn)
            vid_reader = imageio.get_reader(video_path)

            # TODO: Bruk np.array isteden
            metadata = vid_reader.get_meta_data()
            FPS = fps
            metadata['fps'] = FPS
            duration = metadata['duration']
            nb_frames = math.floor(FPS * metadata['duration'])
            print("Their duration: ", duration)
            print("Number of frames: ", nb_frames)

            # TODO: Kutte ut trimminga
            trim_time = VideoTrimmingLimits(t1=0., t2=duration)
            start_frame, end_frame = get_trim_start_end_frames(trim_time, FPS, nb_frames)

            # Loop through the frames of the video
            # TODO: Tar lang tid fordi jeg lager en figur for hvert frame, bedre å lagre numpy array for frame
            for frnb, fr in enumerate(tqdm(range(start_frame, end_frame, int(extract_frame_interval)), 'Frames')):
                arr = np.asarray(vid_reader.get_data(fr))   # Array: [H, W, 3]

                # Display figure and image
                figure_size = (metadata['size'][0] / 100, metadata['size'][1] / 100)
                fig = plt.figure(figsize=figure_size)
                plt.imshow(arr, aspect='auto')

                # Adjust layout to avoid margins, axis ticks, etc. Save and close.
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(seq_dir, f'frame_{frnb:d}.png'))
                plt.close(fig)

            # Close reader before moving (video) files
            vid_reader.close()


def get_positions_from_video(path_to_position_file):
    """ Help function that reads positions from a position file and returns an array with tuples
    representing a position (x_pos, y_pos, z_pos). The position file is complex and uses 3 lines to
    express one position. So this function extracts only the necessary variables
    x, y and z from three different lines and stores it as a tuple in an array
    """
    # Find position file for video
    position_file = open(path_to_position_file)

    x_pos = None
    y_pos = None
    z_pos = None
    positions = []

    # Go through all lines in position file
    for line in position_file:
        # Find position value from line in position file
        new_pos = line.split(" ")[3]
        # Set x_pos
        if x_pos is None:
            x_pos = new_pos

        # Set y_pos
        elif y_pos is None:
            y_pos = new_pos

        # Set z_pos
        elif z_pos is None:
            z_pos = new_pos
            # Add total position (x_pos, y_pos, z_pos) to list of positions
            positions.append((float(x_pos), float(y_pos), float(z_pos)))

            # Reset position
            x_pos = None
            y_pos = None
            z_pos = None

    position_file.close()
    return positions


def get_possible_positions_and_its_airways(label_file_path):
    # Create a function for reading branches_position_numbers.txt file
    positions_labels_file = open(label_file_path)
    lines = positions_labels_file.readlines()

    is_first_line = True
    positions_and_labels = {}

    for line in lines:
        # Handle columns names in first line
        if is_first_line:
            is_first_line = False
            continue

        # Handle last empty line
        elif line == "\n":
            break

        # Fetch only the necessary info from the line
        label = int(line.split(";")[0])
        position = line.split(";")[4]

        x_pos = float(position.split(",")[0])
        y_pos = float(position.split(",")[1])
        z_pos = float(position.split(",")[2])

        # Add position (x_pos, y_pos, z_pos) to dictionary with the belonging label as value
        positions_and_labels[(x_pos, y_pos, z_pos)] = label

    positions_labels_file.close()
    return positions_and_labels


def match_frames_with_positions_and_timestamps(fps, positions, path_to_timestamp_file, frames, positions_and_labels, dataframe, is_forward):
    # Get frame sampling ratio in ms
    frame_sampling_ratio = 1/fps * 1000

    # Get timestamps
    timestamp_file = open(path_to_timestamp_file)
    timestamp_list = np.loadtxt(timestamp_file, delimiter=" ", dtype='int')
    timestamp = 0

    # Check for errors
    if len(timestamp_list) != len(positions):
        print("Num timestamps and num positions does not match")

    #
    possible_positions = positions_and_labels.keys()
    possible_positions_2D_array = map(np.array, possible_positions)
    possible_positions_2D_array = np.array(list(possible_positions_2D_array))

    # Find label for every frame
    for frame_index, frame in enumerate(frames):
        # Find nearest timestamp
        timestamp_array = np.asarray(timestamp_list)
        nearest_timestamp_index = (np.abs(timestamp_array - timestamp)).argmin()
        nearest_timestamp = timestamp_array[nearest_timestamp_index]
        #print("Nearest timestamp: ", nearest_timestamp)

        # Get position from the nearest timestamp
        nearest_position = positions[nearest_timestamp_index]
        nearest_position_list = [float(nearest_position[0]), float(nearest_position[1]), float(nearest_position[2])]
        #print("Nearest position: ", nearest_position)

        # Match the nearest position with a possible position to get a label
        best_match_index = cdist([nearest_position_list], possible_positions_2D_array).argmin()
        best_match_position = possible_positions_2D_array[best_match_index]

        # Find label
        best_match_position_tuple = (best_match_position[0], best_match_position[1], best_match_position[2])
        airway_segment = positions_and_labels[best_match_position_tuple]

        # Add the frame and label to dataframe
        labeled_frame = pd.DataFrame({
            'Frame': [frame],
            'Airway_Segment': airway_segment,
            'Direction': is_forward,
        }).reset_index(drop=True)

        # Store labeled frame in dataframe
        new_dataframe = pd.concat([labeled_frame, dataframe.loc[:]]).reset_index(drop=True)
        dataframe = new_dataframe

        # Set timestamp to its next value for the next iteration in the loop
        timestamp += frame_sampling_ratio
    return new_dataframe


def add_labels(path_to_timestamp_file, path_to_position_file, frames, dataframe, video_count, label_file_path, fps):
    """A help function for crop_scale_and_label_frames that labels the frames in a specific
    sequence given from the path_to_frames parameter
    """
    # Sort the list of frames from the video
    frames.sort(key=natural_keys)

    # Keep track of the direction: forward=1, backward=0
    is_forward = 1

    # "Even"-videos plays backward while "odd"-videos plays forward
    if video_count % 2 == 0:
        # Backward
        # TODO: Something is wrong with the ordering
        frames.sort(key=natural_keys, reverse=True)
        is_forward = 0

    # Get all positions in current video
    positions = get_positions_from_video(path_to_position_file)

    # Get all possible positions and its airway segments
    positions_and_airways = get_possible_positions_and_its_airways(label_file_path)

    # Match positions from video with possible positions to label the positions in video
    dataframe = match_frames_with_positions_and_timestamps(fps, positions, path_to_timestamp_file, frames, positions_and_airways, dataframe, is_forward)

    return dataframe


def crop_and_label_the_frames(path_to_patients, raw_dataset_path, label_file_path, fps):
    """ Crops the frames such that only the frame from the virtual video of the airway
    is stored and passed into the network. Overwrites the frames by storing the cropped
    frame as the frame """

    # Crop specifications from SINTEF Navigation device
    x_start = 538
    y_start = 107
    x_end = 1364
    y_end = 1015

    # Create a list of all patients and remove hidden files like .DS_Store
    patient_list = os.listdir(path_to_patients)
    patient_list = list(filter(lambda x: not x.startswith("."), patient_list))

    # Go through all patients and their sequences to get every frame
    for patient in tqdm(patient_list):
        print("Going through patient: ", patient)
        path_to_sequences = path_to_patients + "/" + patient

        # Create a list of all sequences and remove hidden files like .DS_Store
        sequences_list = os.listdir(path_to_sequences)
        sequences_list = list(filter(lambda x: not x.startswith("."), sequences_list))

        # Count the number of videos
        video_count = 1

        # Go through all video sequences for current patient
        for sequence in sequences_list:
            path_to_frames = path_to_sequences + "/" + sequence

            # Create a dataframe for each video sequence
            dataframe = pd.DataFrame(columns=["Frame", "Airway_Segment", "Direction"])

            # Create a list of all frames and remove hidden files like .DS_Store
            file_list = os.listdir(path_to_frames)
            file_list = list(filter(lambda x: not x.startswith("."), file_list))

            # Save labeling information for current file
            path_to_timestamp_file = ""
            path_to_position_file = ""
            frame_list = []

            # Go through every frame in a video sequence
            for file in file_list:
                # Check for file being a frame
                if file.endswith(".png"):
                    path_to_frame = path_to_frames + "/" + file
                    frame = cv2.imread(path_to_frame)

                    # Crop the frame
                    frame_cropped = frame[y_start:y_end, x_start:x_end]

                    # Save the new frame
                    cv2.imwrite(path_to_frame, frame_cropped)
                    frame_list.append(path_to_frame)

                # Check for file being the timestamp.txt file
                elif file.endswith("timestamps.txt"):
                    path_to_timestamp_file = path_to_frames + "/" + file

                # Check for file being the position.txt file
                elif file.endswith("positions.txt"):
                    path_to_position_file = path_to_frames + "/" + file

            # Add labels to the frames in current sequence
            dataframe = add_labels(path_to_timestamp_file, path_to_position_file, frame_list, dataframe, video_count,
                                   label_file_path, fps)
            video_count += 1

            # Convert dataframe into CSV file in order to store the video sequence as a file
            path = raw_dataset_path + f"{sequence}.csv"
            dataframe.to_csv(path, index=False)


def perform_mapping(label_map_dict, label):
    # Label is >= num_generations we are looking at
    if label in label_map_dict.keys():
        new_label = label_map_dict[label][0]
    # Label is > num_generations we are looking at
    else:
        new_label = 0
    return new_label


def relabel_frames(labeled_videos_path, label_map_dict, relabeled_videos_path):
    """
    Method that relabels the dataset based on a mapping created after the numbers of generations entered.
    Method also deleted frames that has a larger generation than the one entered
    """
    # Go through all csv files and avoid .DS_Store
    for video in list(os.listdir(labeled_videos_path)):
        if not video.startswith("."):
            video_df = pd.read_csv(labeled_videos_path + video)
            # Create a new dataframe to store the relabeled information
            temp_df = copy.deepcopy(video_df)

            # Change label by using the given mapping system
            temp_df["Airway_Segment"] = video_df["Airway_Segment"].apply(lambda label: perform_mapping(label_map_dict=label_map_dict, label=label))

            # Remove frames not belonging to the generation chosen (labeled with 0)
            new_video_df = copy.deepcopy(temp_df[(temp_df.Airway_Segment != 0)])
            new_video_df.to_csv(relabeled_videos_path + video, index=False)


def preprocess(convert_videos_to_frames, label_the_frames, videos_path, frames_path, fps, raw_dataset_path,
               label_file_path, label_map_dict, relabel_the_frames, relabeled_csv_videos_path):
    print("-- PREPROCESS --")
    # Create frames (png) from videos (mp4)
    if convert_videos_to_frames:
        convert_video_to_frames(input_data_path=videos_path, output_data_path=frames_path, fps=fps)

    # Label the frames with the original labels and crop them
    if label_the_frames:
        crop_and_label_the_frames(path_to_patients=frames_path, raw_dataset_path=raw_dataset_path,
                                  label_file_path=label_file_path, fps=fps)

    # Relabel each frame to label between 0-26 and remove all frames with label 0
    if relabel_the_frames:
        relabel_frames(labeled_videos_path=raw_dataset_path, label_map_dict=label_map_dict, relabeled_videos_path=relabeled_csv_videos_path)
