import torch
import torchvision.models
from torch import nn


class TimeDistributed(nn.Module):
    def __init__(self, feature_extractor, org_shape):
        super(TimeDistributed, self).__init__()
        self.feature_extractor = feature_extractor
        self.org_shape = org_shape

    def forward(self, X):
        """
        Reshape 5 dim input ([batch_dim, time_steps, height, width, channels]) into 4 dim
        ([batch_dim * time_steps, height, width, channels]) by merging columns batch_size and time_steps,
        such that the feature extractor can handle the input. When features are returned in a 2D output, the output
        is reshaped to 3D ([batch_size=16, num_frames=5, features=128])
        """
        # Reshape to 4 dim
        X_reshaped = X.contiguous().view((-1,) + self.org_shape[2:])  # [8 * 10, 3, 384, 384]
        output = self.feature_extractor(X_reshaped.float())  # [8 * 10, 128]

        # Reshape to 3D
        output_reshaped = output.contiguous().view((-1, self.org_shape[1]) + (output.shape[-1],))  # [8, 10, 128]
        return output_reshaped


class NavigationNet(nn.Module):
    def __init__(self, hidden_nodes, num_features, num_LSTM_cells, num_frames_in_stack, num_airway_segment_classes, num_direction_classes, frame_dimension, batch_size):
        super().__init__()
        self.hidden_nodes = hidden_nodes
        self.num_features = num_features
        self.num_LSTM_cells = num_LSTM_cells
        self.num_frames_in_stack = num_frames_in_stack
        self.num_airway_segment_classes = num_airway_segment_classes
        self.num_direction_classes = num_direction_classes
        self.batch_size = batch_size
        self.shape = tuple((batch_size, num_frames_in_stack, 3, frame_dimension[0], frame_dimension[1]))

        # Feature extractor: Resnet18 or ViT
        self.feature_extractor = torchvision.models.resnet18('IMAGENET1K_V1')
        self.feature_extractor.fc = nn.Linear(512, self.num_features, bias=True)
        self.time_distributed = TimeDistributed(self.feature_extractor, self.shape)

        # LSTM
        #self.LSTM = nn.LSTM(input_size=self.num_features, hidden_size=self.hidden_nodes, num_layers=self.num_LSTM_cells, batch_first=True)

        # Classifier Direction
        self.direction_classifier = nn.Sequential(
            #nn.Linear(self.hidden_nodes, 32),
            nn.Linear(self.num_features, 32), # For baseline without LSTM
            nn.ReLU(),
            nn.Linear(32, self.num_direction_classes),
        )

        # Classifier Airway
        self.airway_classifier = nn.Sequential(
            #nn.Linear(self.hidden_nodes, 32),
            nn.Linear(self.num_features, 32),  # For baseline without LSTM
            nn.ReLU(),
            nn.Linear(32, self.num_airway_segment_classes),
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
        #for param in self.LSTM.parameters():
            #param.requires_grad = True
        for param in self.airway_classifier.parameters():
            param.requires_grad = True
        for param in self.direction_classifier.parameters():
            param.requires_grad = True

    def forward(self, X):
        # Feature extractor
        X = self.time_distributed(X)  # [batch_sizer=16, features=128]

        # LSTM
        #X, _ = self.LSTM(X)  # [16, 5, 64]

        # Classifiers (Direction and Airway)
        airway = self.airway_classifier(X)          # [batch_size=16, num_frames_in_stack=5, 27]
        direction = self.direction_classifier(X)    # [batch_size=16, num_frames_in_stack=5, 2]

        return airway, direction


def create_neural_net(hidden_nodes, num_features, num_LSTM_cells, num_frames_in_stack, num_airway_segment_classes, num_direction_classes, frame_dimension, batch_size):
    print("-- NEURAL NET --")
    neural_net = NavigationNet(hidden_nodes=hidden_nodes, num_features=num_features, num_LSTM_cells=num_LSTM_cells,
                               num_frames_in_stack=num_frames_in_stack, num_airway_segment_classes=num_airway_segment_classes,
                               num_direction_classes=num_direction_classes, frame_dimension=frame_dimension, batch_size=batch_size)
    return neural_net
