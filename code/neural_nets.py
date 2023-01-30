import torchvision
from torch import nn
import torch
from parameters import batch_size, hidden_nodes, network_type, num_classes


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
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


class DirectionDetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_nodes = hidden_nodes

        # Feature extractor
        self.feature_extractor = torchvision.models.resnet18(pretrained=True)

        # Reshaping 5D to 4D
        self.time_distributed = TimeDistributed(self.feature_extractor)

        # Recurrent Neural Network
        self.RNN = nn.LSTM(1000, self.hidden_nodes, 1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(640, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        # Handle training for certain layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.feature_extractor.layer3.parameters():
            param.requires_grad = True
        for param in self.feature_extractor.layer4.parameters():
            param.requires_grad = True
        for param in self.feature_extractor.fc.parameters():
            param.requires_grad = True

    def forward(self, X):
        X = self.time_distributed(X)
        X = self.RNN(X)[0]
        # Reshape from 3D to 2D
        original_shape = tuple(X.shape)
        X_reshaped = X.reshape((batch_size, 5*self.hidden_nodes))
        X = self.classifier(X_reshaped)
        return X


class SegmentDetNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True, progress=True)
        self.model.fc = nn.Linear(512, 256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        # Only train the last layers of the network
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer3.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x


def create_neural_net():
    if network_type == "segment_det_net":
        neural_net = SegmentDetNet(num_classes)

    elif network_type == "direction_det_net":
        neural_net = DirectionDetNet()

    else:
        print("Neural network type not set")
        neural_net = None

    return neural_net