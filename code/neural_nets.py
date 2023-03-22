import torch
import torchvision.models
from torch import nn
from pytorch_pretrained_vit import ViT


class TimeDistributer(nn.Module):
    def __init__(self, feature_extractor, org_shape):
        super(TimeDistributer, self).__init__()
        self.feature_extractor = feature_extractor
        self.org_shape = org_shape

    def forward(self, X):
        """
        Reshape 5 dim input into 4 dim input by merging columns batch_size and time_steps,
        such that:
        In: [batch_dim, time_steps, height, width, channels]
        Temp: [batch_dim * time_steps, height, width, channels]
        Then reshaping back into 5 dim
        """
        # Reshape to 4 dim
        X_reshaped = X.contiguous().view((-1,) + self.org_shape[2:])  # (samples * timesteps, input_size)
        #X_reshaped = X.reshape((self.org_shape[0] * self.org_shape[1],) + self.org_shape[2:])  # [80, 3, 384, 384]
        output = self.feature_extractor(X_reshaped.float())  # [80, 128]

        # Reshape to 3D
        output_reshaped = output.contiguous().view((-1, self.org_shape[1]) + (output.shape[-1],))
        #output_reshaped = output.reshape(self.org_shape[:2] + (output.shape[-1],))  # [batch_size=16, num_frames=5, features=128]
        return output_reshaped


class NavigationNet(nn.Module):
    def __init__(self, hidden_nodes, num_frames_in_stack, num_airway_segment_classes, num_direction_classes, frame_dimension, batch_size):
        super().__init__()
        self.hidden_nodes = hidden_nodes
        self.num_frames_in_stack = num_frames_in_stack
        self.num_airway_segment_classes = num_airway_segment_classes
        self.num_direction_classes = num_direction_classes
        self.batch_size = batch_size
        self.shape = tuple((batch_size, num_frames_in_stack, 3, frame_dimension[0], frame_dimension[1]))

        # Feature extractor: Resnet18 or ViT
        self.feature_extractor = torchvision.models.resnet18('ResNet18_Weights')
        self.feature_extractor.fc = nn.Linear(512, 128, bias=True)
        self.time_distributer = TimeDistributer(self.feature_extractor, self.shape)

        # LSTM
        self.LSTM = nn.LSTM(input_size=128, hidden_size=self.hidden_nodes, num_layers=3, batch_first=True)

        # Classifier Direction
        self.direction_classifier = nn.Sequential(
            nn.Linear(self.hidden_nodes, 32),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(32, self.num_direction_classes),
        )

        # Classifier Airway
        self.airway_classifier = nn.Sequential(
            nn.Linear(self.hidden_nodes, 32),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(32, self.num_airway_segment_classes),
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
        # Feature extractor
        X = self.time_distributer(X)  # [batch_sizer=16, features=128]

        # LSTM
        X, _ = self.LSTM(X)  # [16, 5, 64]

        # Classifiers (Direction and Airway)
        airway = self.airway_classifier(X)          # [batch_size=16, num_frames_in_stack=5, 27]
        direction = self.direction_classifier(X)    # [batch_size=16, num_frames_in_stack=5, 2]

        return airway, direction


class FrankyNet(nn.Module):
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

        # Reshape to 4D
        X = X.reshape(
            (org_shape[0], org_shape[1] * org_shape[2], org_shape[3], org_shape[4])).float()  # [16, 3*5, 384, 384]
        print("1. reshape: ", X.shape)


        # Feature extractor
        X = self.feature_extractor(X)  # [16, 1000]
        print("After CNN: ", X.shape)

        # Reshape to 3D
        X = X.reshape(
            (1, org_shape[0], 1000)).float()  # [16, 3*5, 384, 384]
        print("2. reshape: ", X.shape)

        # LSTM
        X, _ = self.LSTM(X)  # [1, 16, 64]

        print("After LSTM: ", X.shape)

        # Classifiers (Direction and Airway)
        airway = self.airway_classifier(X)  # [batch_size, num_frames_in_stack, 27]
        direction = self.direction_classifier(X)  # [batch_size, num_frames_in_stack, 2]

        return airway, direction


def create_neural_net(hidden_nodes, num_frames_in_stack, num_airway_segment_classes, num_direction_classes, frame_dimension, batch_size):
    print("-- NEURAL NET --")
    neural_net = NavigationNet(hidden_nodes, num_frames_in_stack, num_airway_segment_classes, num_direction_classes, frame_dimension, batch_size)

    #neural_net.eval()
    # Create dummy input for the model. It will be used to run the model inside export function.
    #dummy_input = torch.randn(16, 5, 384, 384, 3)
    # Call the export function
    #torch.onnx.export(neural_net, (dummy_input,), 'model.onnx')

    return neural_net
