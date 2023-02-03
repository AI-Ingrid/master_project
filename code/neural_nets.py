import torch
from torch import nn
from parameters import num_classes, hidden_nodes
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
        Out: [batch_dim * time_steps, height, width, channels]
        Then reshaping back into 5 dim
        """
        X = X.transpose(2, 3)
        X = X.transpose(2, 4)
        org_shape = tuple(X.shape)

        # Reshape to 4 dim
        X_reshaped = X.reshape((torch.prod(torch.tensor(org_shape[:2])),) + org_shape[2:])
        output = self.module(X_reshaped.float())

        # Reshape back to 5 dim
        output_reshaped = output.reshape(org_shape[:2] + (output.shape[-1],))
        return output_reshaped


class NavigationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_nodes = hidden_nodes

        # INPUT: Sequence of 10 frames
        # Encoder: ViT + TimeDistributer
        self.vision_transformer = ViT('B_16_imagenet1k', pretrained=True)  # OUT: 1000
        self.time_distributer = TimeDistributer(self.vision_transformer)   # OUT: ?

        # Decoder: RNN + classifier
        # RNN: LSTM
        self.RNN = nn.LSTM(1000, self.hidden_nodes, 1, batch_first=True)  # OUT: ?

        # Classifier Direction
        self.direction_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(640, 64),
            nn.ReLU(),

            nn.Linear(64, 2),
        )

        # Classifier Airway
        self.airway_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        # OUTPUT: Direction at the end of the sequence, Segment at the end of the sequence

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


def create_neural_net():
    neural_net = NavigationNet()
    return neural_net