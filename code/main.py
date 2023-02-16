from random import seed
from config import *
from preprocess import preprocess
from datasets import create_datasets_and_dataloaders
from neural_nets import create_neural_net
from train_models import train_model
from test_models import test_model


def main():
    """ The function running the entire pipeline of the project """
    seed(0)

    # Preprocess the data from videos to frames with labels
    preprocess(convert_videos_to_frames=convert_videos_to_frames, label_the_frames=label_the_frames, videos_path=videos_path,
               frames_path=frames_path, fps=fps, raw_dataset_path=csv_videos_path, label_file_path=label_file_path,
               label_map_dict=label_map_dict, relabel_the_frames=relabel_the_frames, relabeled_csv_videos_path=relabeled_csv_videos_path)

    # Create datasets and dataloaders
    train_dataloader, test_dataloader, validation_dataloader = create_datasets_and_dataloaders(
        validation_split=validation_split, test_split=test_split,
        raw_dataset_path=relabeled_csv_videos_path, dataset_path=dataset_path,
        num_stacks=num_stacks, num_frames_in_stack=num_frames_in_stack,
        slide_ratio_in_stack=slide_ratio_in_stack, batch_size=batch_size,
        shuffle_dataset=shuffle_dataset, split_the_data=split_the_data,
        num_airway_segment_classes=num_airway_segment_classes, num_direction_classes=num_direction_classes,
        frame_dimension=frame_dimension
    )
    """
    # Create neural network
    neural_net = create_neural_net(hidden_nodes, num_airway_segment_classes, num_direction_classes)

    # Train model
    trainer = train_model(batch_size, learning_rate, early_stop_count, epochs, num_validations, neural_net,
                          train_dataloader, validation_dataloader, fps, train_plot_path, train_plot_name,
                          num_airway_segment_classes, num_direction_classes)

    # Test model
    test_model(trainer, train_dataloader, test_dataloader, neural_net)
    """

if __name__ == "__main__":
    main()
