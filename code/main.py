from config import *
from preprocess import preprocess
from datasets import create_datasets_and_dataloaders
from neural_nets import create_neural_net
from train_models import train_model
from test_models import test_model


def main():
    """ The function running the entire pipeline of the project """
    # Preprocess the data from videos to frames with labels
    preprocess(convert_videos_to_frames=convert_videos_to_frames,
               label_the_frames=label_the_frames, videos_path=videos_path,
               frames_path=frames_path, fps=fps, raw_dataset_path=csv_videos_path,
               label_file_path=label_file_path,label_map_dict=label_map_dict,
               relabel_the_frames=relabel_the_frames, relabeled_csv_videos_path=relabeled_csv_videos_path)

    # Create datasets and dataloaders
    train_dataloader, validation_dataloader, test_dataset = create_datasets_and_dataloaders(
        validation_split=validation_split, test_split=test_split,
        raw_dataset_path=relabeled_csv_videos_path, dataset_path=dataset_path,
        num_stacks=num_stacks, num_frames_in_stack=num_frames_in_stack,
        slide_ratio_in_stack=slide_ratio_in_stack, batch_size=batch_size,
        shuffle_dataset=shuffle_dataset, split_the_data=split_the_data,
        num_airway_segment_classes=num_airway_segment_classes, num_direction_classes=num_direction_classes,
        frame_dimension=frame_dimension
    )

    # Create neural network
    neural_net = create_neural_net(num_memory_nodes=num_memory_nodes, num_features_extracted=num_features_extracted,
                                   model_type=model_type, num_frames_in_stack=num_frames_in_stack,
                                   num_airway_segment_classes=num_airway_segment_classes,
                                   num_direction_classes=num_direction_classes, frame_dimension=frame_dimension,
                                   batch_size=batch_size, use_stateful_LSTM=use_stateful_LSTM)

    # Train model
    trainer = train_model(perform_training=perform_training, batch_size=batch_size, learning_rate=learning_rate,
                          early_stop_count=early_stop_count, epochs=epochs, num_validations=num_validations,
                          neural_net=neural_net, train_dataloader=train_dataloader,
                          validation_dataloader=validation_dataloader, fps=fps, num_airway_segment_classes=num_airway_segment_classes,
                          num_direction_classes=num_direction_classes, num_frames_in_stack=num_frames_in_stack,
                          model_path=model_path, model_name=model_name, use_focal_loss=use_focal_loss, alpha_airway=alpha_airway,
                          alpha_direction=alpha_direction, gamma=gamma, model_type=model_type)

    # Test model
    test_model(trainer=trainer, test_dataset=test_dataset, test_slide_ratio=test_slide_ratio_in_stack,
               num_frames=num_frames_in_stack, num_airway_classes=num_airway_segment_classes,
               num_direction_classes=num_direction_classes, data_path=data_path, frame_dimension=frame_dimension,
               convert_to_onnx=convert_to_onnx, model_name=model_name, model_path=model_path, test_plot_path=test_plot_path,
               model_type=model_type)


if __name__ == "__main__":
    main()
