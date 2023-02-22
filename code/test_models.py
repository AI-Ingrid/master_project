from utils.data_utils import plot_dataset_distribution
from train_models import compute_f1_and_loss
from torch.nn import CrossEntropyLoss
from utils.test_utils import plot_confusion_metrics, plot_predictions_test_set, compute_f1_score
from config import load_best_model, confusion_metrics_path, test_plot_path, get_data_dist, get_loss_and_accuracy, \
    get_confusion_metrics, get_f1_score, get_testset_pred, num_airway_segment_classes, num_direction_classes
import pandas as pd
import numpy as np


def data_distribution(train, validation, test):
    if get_data_dist:
        # Visualize data sets
        plot_dataset_distribution(train, validation=None, test=None)
        plot_dataset_distribution(train, validation=validation, test=None)
        plot_dataset_distribution(train, validation=None, test=test)


def loss_and_accuracy(train, validation, test, neural_net):
    if get_loss_and_accuracy:
        print("---- TRAINING ----")
        compute_f1_and_loss(train, neural_net, CrossEntropyLoss(), num_airway_segment_classes, num_direction_classes)
        print("---- VALIDATION ----")
        compute_f1_and_loss(validation, neural_net, CrossEntropyLoss(), num_airway_segment_classes, num_direction_classes)
        print("---- TEST ----")
        compute_f1_and_loss(test, neural_net, CrossEntropyLoss(), num_airway_segment_classes, num_direction_classes)


def visualize_predictions_on_videos(model, test_dataset, test_slide_ratio, num_frames):
    predictions = []
    for frames, (airway_labels, direction_labels) in test_dataset:
        for i in range(0, len(frames), num_frames * test_slide_ratio):
            stack = frames[i:i + num_frames * test_slide_ratio: test_slide_ratio]
            predictions_airway, predictions_direction = model(stack)


def test_model(trainer, test_dataset, test_slide_ratio, num_frames):
    print("-- TESTING --")
    if load_best_model:
        # Load neural net model
        trainer.load_model()
        # Set to inference mode (freeze model)
        trainer.model.eval()

    visualize_predictions_on_videos(trainer.model, test_dataset, test_slide_ratio, num_frames)
