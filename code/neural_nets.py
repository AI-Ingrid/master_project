import torch
from torch import nn
from pytorch_pretrained_vit import ViT


class TimeDistributer(nn.Module):
    def __init__(self, module):
        super(TimeDistributer, self).__init__()
        self.module = module

    def forward(self, X):
        """
        Reshape 5 dim input into 4 dim input by merging columns batch_size and time_steps,
        such that:
        In: [batch_dim, time_steps, height, width, channels]
        Temp: [batch_dim * time_steps, height, width, channels]
        Then reshaping back into 5 dim
        """
        # TODO: CROP AND SCALE FRAMES TO (384, 384) fordi ViT-en trenger det
        # [32, 30, 256, 256, 3]
        X = X.transpose(2, 3)  # [32, 30, 256, 256, 3]
        X = X.transpose(2, 4)  # [32, 30, 3, 256, 256]
        org_shape = tuple(X.shape)

        # Reshape to 4 dim
        X_reshaped = X.reshape((torch.prod(torch.tensor(org_shape[:2])),) + org_shape[2:])  # [960, 3, 256, 256]
        output = self.module(X_reshaped.float())

        # Reshape back to 5 dim
        output_reshaped = output.reshape(org_shape[:2] + (output.shape[-1],))  # [32, 30, 3, 256, 256] ??
        return output_reshaped


class NavigationNet(nn.Module):
    def __init__(self, hidden_nodes, num_airway_segment_classes, num_direction_classes):
        super().__init__()
        self.hidden_nodes = hidden_nodes
        self.num_airway_segment_classes = num_airway_segment_classes
        self.num_direction_classes = num_direction_classes

        # INPUT: Sequence of 'num_frames_in_stack' frames
        # Encoder: ViT + TimeDistributer
        self.vision_transformer = ViT('B_16_imagenet1k', pretrained=True)  # OUT: [batch_size, 1000]
        self.time_distributer = TimeDistributer(self.vision_transformer)   # OUT: # [32, 30, 3, 256 -> 384, 256 -> 384] ???

        # Decoder: RNN + classifier
        # RNN: LSTM
        self.RNN = nn.LSTM(1000, self.hidden_nodes, 1, batch_first=True)  # OUT: ?

        # Classifier Direction
        self.direction_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(640, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_direction_classes),
        )

        # Classifier Airway
        self.airway_classifier = nn.Sequential(
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.num_airway_segment_classes),
        )

    def forward(self, X):
        # INPUT: Sequence of 10 frames
        # Timedistributer + ViT
        X = self.time_distributer(X)

        # RNN
        X = self.RNN(X)

        # Classifiers (Direction and Airway)
        airway = self.airway_classifier(X)
        direction = self.direction_classifier(X)

        return airway, direction


def create_neural_net(hidden_nodes, num_airway_segment_classes, num_direction_classes):
    print("NEURAL NET")
    neural_net = NavigationNet(hidden_nodes, num_airway_segment_classes, num_direction_classes)
    return neural_net
