from parameters import load_best_model, confusion_metrics_path, test_plot_path, get_data_dist, get_loss_and_accuracy, get_confusion_metrics, get_f1_score, get_testset_pred
from utils.data_utils import plot_dataset_distribution
from train_models import compute_loss_and_accuracy
from torch.nn import CrossEntropyLoss
from utils.test_utils import plot_confusion_metrics, plot_predictions_test_set, compute_f1_score


def data_distribution(train, validation, test):
    if get_data_dist:
        # Visualize data sets
        plot_dataset_distribution(train, validation=None, test=None)
        plot_dataset_distribution(train, validation=validation, test=None)
        plot_dataset_distribution(train, validation=None, test=test)


def loss_and_accuracy(train, validation, test, neural_net):
    if get_loss_and_accuracy:
        print("---- TRAINING ----")
        compute_loss_and_accuracy(train, neural_net, CrossEntropyLoss())
        print("---- VALIDATION ----")
        compute_loss_and_accuracy(validation, neural_net, CrossEntropyLoss())
        print("---- TEST ----")
        compute_loss_and_accuracy(test, neural_net, CrossEntropyLoss())


def test_model(trainer, train_dataloader, test_dataloader, neural_net):
    print("TESTING")
    if load_best_model:
        # Load neural net model
        trainer.load_best_model()

    # Split the datasets in train, test and validation
    train, validation = train_dataloader
    test = test_dataloader

    # Distribution
    data_distribution(train, validation, test)

    # Loss and accuracy
    loss_and_accuracy(train, validation, test, neural_net)

    # Confusion metrics
    print("plotting confusion metrics")
    plot_confusion_metrics(test, trainer, confusion_metrics_path, get_confusion_metrics)

    print("plotting test images")
    # Plot test images with predicted and original label on it
    plot_predictions_test_set(test, trainer, test_plot_path, get_testset_pred)

    # F1 score
    print("computing f1 score..")
    compute_f1_score(test, trainer, get_f1_score)
