from matplotlib import pyplot as plt
from pathlib import Path
from utils.neural_nets_utils import to_cuda, decode_one_hot_encoded_labels
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import numpy as np
from utils.data_utils import get_label_name


def plot_confusion_metrics(test_set, trainer, path, get_confusion_metrics, network_type):
    if get_confusion_metrics:
        all_predicted_labels = []
        all_original_labels = []

        plot_path = Path(path)
        plot_path.mkdir(exist_ok=True)

        for X_batch, Y_batch in test_set:
            X_batch_cuda = to_cuda(X_batch)

            # Perform the forward pass
            predictions = trainer.model(X_batch_cuda)

            predicted_labels = decode_one_hot_encoded_labels(predictions)
            original_labels = decode_one_hot_encoded_labels(Y_batch)

            all_predicted_labels += predicted_labels
            all_original_labels += original_labels

        if network_type == 'direction_det_net':
            classes = list(range(0, 2))
        else:
            classes = list(range(1, 28))

        cm = confusion_matrix(all_original_labels, all_predicted_labels, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        fig, ax = plt.subplots(figsize=(20, 20))
        plt.title(f"Confusion Metrics for {network_type}")
        disp.plot(ax=ax)
        plt.savefig(plot_path.joinpath(f"confusion_matrix.png"))
        plt.show()


def plot_predictions_test_set(test_set, trainer, path, network_type, get_testset_pred):
    if get_testset_pred:
        # Store images with predicted and true label on it
        batch_num = 0
        for X_batch, Y_batch in test_set:
            print("Batch num: ", batch_num)
            X_batch_cuda = to_cuda(X_batch)

            # Perform the forward pass
            predictions = trainer.model(X_batch_cuda)

            predictions = predictions.cpu()
            # DirectionDetNet
            if network_type == 'direction_det_net':
                label_names = {1: "Forward",
                               0: "Backward"}
                # Find predicted label
                for batch_index, batch in enumerate(predictions.detach().numpy()):
                    predicted_label = int(np.argmax(batch))
                    predicted_name_label = label_names[predicted_label]
                    original_label = int(np.argmax(Y_batch[batch_index]))
                    original_name_label = label_names[original_label]

                    name = f"batch_{batch_num}_index_{batch_index}"
                    print("Predicted label: ", predicted_name_label, " Original label: ", original_name_label)

                    # Create plots
                    plot_path = Path(path)
                    plot_path.mkdir(exist_ok=True)
                    fig = plt.figure(figsize=(35, 12), constrained_layout=True)
                    images = X_batch[batch_index]
                    fig.suptitle(f"Predicted Label: {predicted_name_label} \n Original Label: {original_name_label}")

                    # Image 1
                    plt.subplot(1, 5, 1)
                    image_1 = images[0].numpy()
                    plt.title("Frame 1")
                    plt.imshow(image_1)

                    # Image 2
                    plt.subplot(1, 5, 2)
                    image_2 = images[1].numpy()
                    plt.title("Frame 2")
                    plt.imshow(image_2)

                    # Image 3
                    plt.subplot(1, 5, 3)
                    image_3 = images[2].numpy()
                    plt.title("Frame 3")
                    plt.imshow(image_3)

                    # Image 4
                    plt.subplot(1, 5, 4)
                    image_4 = images[3].numpy()
                    plt.title("Frame 4")
                    plt.imshow(image_4)

                    # Image 5
                    plt.subplot(1, 5, 5)
                    image_5 = images[4].numpy()
                    plt.title("Frame 5")
                    plt.imshow(image_5)

                    plt.savefig(plot_path.joinpath(f"{name}.png"))
                    print("Saving figure..")

                batch_num += 1
                if batch_num == 10:
                    break

            # SegmentDetNet
            else:
                for batch_index, batch in enumerate(predictions.detach().numpy()):
                    # Find predicted label
                    predicted_label = int(np.argmax(batch))
                    original_label = int(np.argmax(Y_batch[batch_index]))

                    # Get label names
                    original_name_label = get_label_name(original_label)
                    predicted_name_label = get_label_name(predicted_label)

                    name = f"batch_{batch_num}_index_{batch_index}"
                    print("Predicted label: ", str(predicted_label), " Original label: ", str(original_label))

                    # Create plots
                    plot_path = Path(path)
                    plot_path.mkdir(exist_ok=True)
                    plt.figure(figsize=(8, 8), constrained_layout=True)
                    image = X_batch[batch_index]
                    image = image.permute(1, 2, 0).numpy()

                    # Predicted label and Original label image
                    plt.subplot(1, 2, 1)
                    plt.title(f"Predicted Label: {predicted_label} : {predicted_name_label} \n Original Label: {original_label}: {original_name_label}")
                    plt.imshow(image)
                    plt.savefig(plot_path.joinpath(f"{name}.png"))
                    print("Figure saved..")

                batch_num += 1
                if batch_num == 10:
                    break


def compute_f1_score(test_set, trainer, get_f1_score):
    if get_f1_score:
        batch_num = 0
        f1_macro_score = 0
        f1_micro_score = 0
        f1_weighted_score = 0

        for X_batch, Y_batch in test_set:
            X_batch_cuda = to_cuda(X_batch)
            # Perform the forward pass
            predictions = trainer.model(X_batch_cuda)

            predicted_labels = decode_one_hot_encoded_labels(predictions)
            Y_batch_1d = decode_one_hot_encoded_labels(Y_batch)

            f1_macro_score += f1_score(Y_batch_1d, predicted_labels, average='macro')
            f1_micro_score += f1_score(Y_batch_1d, predicted_labels, average='micro')
            f1_weighted_score += f1_score(Y_batch_1d, predicted_labels, average='weighted')

            batch_num += 1

        print("F1 macro score: ", f1_macro_score/batch_num)
        print("F1 micro score: ", f1_micro_score/batch_num)
        print("F1 weighted score: ", f1_weighted_score/batch_num)
