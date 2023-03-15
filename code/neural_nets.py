import torch
import torchvision.models
from torch import nn
from pytorch_pretrained_vit import ViT


class NavigationNet(nn.Module):
    def __init__(self, hidden_nodes, num_frames_in_stack, num_airway_segment_classes, num_direction_classes):
        super().__init__()
        self.hidden_nodes = hidden_nodes
        self.num_frames_in_stack = num_frames_in_stack
        self.num_airway_segment_classes = num_airway_segment_classes
        self.num_direction_classes = num_direction_classes

        # Feature extractor: Resnet18 or ViT
        self.feature_extractor = torchvision.models.resnet18('ResNet18_Weights')
        self.feature_extractor.conv1 = nn.Conv2d(self.num_frames_in_stack * 3, 64, kernel_size=7, stride=2, padding=3,
                                                 bias=False)
        self.feature_extractor.avgpool = nn.Sequential()
        self.feature_extractor.fc = nn.Sequential()

        print(self.feature_extractor)
        # LSTM
        self.LSTM = nn.LSTM(input_size=1000, hidden_size=self.hidden_nodes, num_layers=3, batch_first=True)

        # Classifier Direction
        self.direction_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.hidden_nodes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_direction_classes),
        )

        # Classifier Airway
        self.airway_classifier = nn.Sequential(
            nn.Linear(self.hidden_nodes, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_airway_segment_classes),
        )

        # Handle training for certain layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        for param in self.LSTM.parameters():
            param.requires_grad = True
        for param in self.airway_classifier.parameters():
            param.requires_grad = True
        for param in self.direction_classifier.parameters():
            param.requires_grad = True

    def forward(self, X):
        X = X.transpose(2, 3)  # [16, 5, 384, 384, 3]
        X = X.transpose(2, 4)  # [16, 5, 3, 384, 384]
        org_shape = tuple(X.shape)

        # Reshape to 4 dim
        X = X.reshape((org_shape[0], org_shape[1] * org_shape[2], org_shape[3], org_shape[4])).float()  # [16, 3*5, 384, 384]

        # Feature extractor
        X = self.feature_extractor(X)  # [batch_sizer=16, features=1000]
        print("SHAPE X: ", X.shape)

        # LSTM
        X, _, _ = self.LSTM(X)

        # Classifiers (Direction and Airway)
        airway = self.airway_classifier(X)          # [batch_size, num_frames_in_stack, 27]
        direction = self.direction_classifier(X)    # [batch_size, num_frames_in_stack, 2]

        return airway, direction


def create_neural_net(hidden_nodes, num_frames_in_stack, num_airway_segment_classes, num_direction_classes):
    print("-- NEURAL NET --")
    neural_net = NavigationNet(hidden_nodes, num_frames_in_stack, num_airway_segment_classes, num_direction_classes)
    return neural_net
