import torch
import numpy as np
import pathlib
import random
import torch.nn as nn
import torch.nn.functional as F


# Allow torch/cudnn to optimize/analyze the input/output shape of convolutions
# To optimize forward/backward pass.
# This will increase model throughput for fixed input shape to the network
torch.backends.cudnn.benchmark = True

# Cudnn is not deterministic by default. Set this to True if you want
# to be sure to reproduce your results
torch.backends.cudnn.deterministic = True


def focal_loss(predictions, labels, num_classes, alpha, gamma):
    # Predictions = [batch_size * num_frames = 160, num_classes = 27 eller 2]
    # Labels = [batch_size * num_frames = 160, num_classes = 27 eller 2]

    cross_entropy_loss = torch.nn.functional.cross_entropy(predictions, labels, reduction='none')  # important to add reduction='none' to keep per-batch-item loss
    pt = torch.exp(-cross_entropy_loss)
    loss = (alpha * (1 - pt) ** gamma * cross_entropy_loss).mean()


    return loss


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def to_cuda(elements):
    """
    Transfers every object in elements to GPU VRAM if available.
    elements can be a object or list/tuple of objects
    """
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.cuda() for x in elements]
        return elements.cuda()
    return elements


def save_checkpoint(state_dict: dict,
                    filepath: pathlib.Path,
                    is_best: bool,
                    max_keep: int = 1):
    """
    Saves state_dict to filepath. Deletes old checkpoints as time passes.
    If is_best is toggled, saves a checkpoint to best.ckpt
    """
    filepath.parent.mkdir(exist_ok=True, parents=True)
    list_path = filepath.parent.joinpath("latest_checkpoint")
    torch.save(state_dict, filepath)
    if is_best:
        torch.save(state_dict, filepath.parent.joinpath("best.ckpt"))
    previous_checkpoints = get_previous_checkpoints(filepath.parent)
    if filepath.name not in previous_checkpoints:
        previous_checkpoints = [filepath.name] + previous_checkpoints
    if len(previous_checkpoints) > max_keep:
        for ckpt in previous_checkpoints[max_keep:]:
            path = filepath.parent.joinpath(ckpt)
            if path.exists():
                path.unlink()
    previous_checkpoints = previous_checkpoints[:max_keep]
    with open(list_path, 'w') as fp:
        fp.write("\n".join(previous_checkpoints))


def get_previous_checkpoints(directory: pathlib.Path) -> list:
    assert directory.is_dir()
    list_path = directory.joinpath("latest_checkpoint")
    list_path.touch(exist_ok=True)
    with open(list_path) as fp:
        ckpt_list = fp.readlines()
    return [_.strip() for _ in ckpt_list]


def load_best_checkpoint(directory: pathlib.Path):
    filepath = directory.joinpath("best.ckpt")
    if not filepath.is_file():
        return None
    #map_location = torch.device('cpu')
    return torch.load(directory.joinpath("best.ckpt"))


def decode_one_hot_encoded_labels(one_hot_encoded_labels):
    encoded_labels = one_hot_encoded_labels.cpu()
    decoded_labels = []

    for batch_index, batch in enumerate(encoded_labels.detach().numpy()):
        decoded_label = np.argmax(batch)
        decoded_labels.append(decoded_label)
    decoded_labels = torch.tensor(decoded_labels)
    return decoded_labels


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






