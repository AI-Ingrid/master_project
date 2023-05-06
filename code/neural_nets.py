import torchvision.models
from torch import nn
import torch
import torch.nn.functional as F

class TimeDistributed(nn.Module):
    def __init__(self, feature_extractor, org_shape):
        super(TimeDistributed, self).__init__()
        self.feature_extractor = feature_extractor
        self.org_shape = org_shape

    def forward(self, X):
        """
        Reshape 5 dim input ([batch_dim, time_steps, channels, height, width]) into 4 dim
        ([batch_dim * time_steps, channels, height, width]) by merging columns batch_size and time_steps,
        such that the feature extractor can handle the input. When features are returned in a 2D output, the output
        is reshaped to 3D ([batch_size=16, num_frames=5, features=128])
        """
        assert X.shape[1:] == self.org_shape[1:], f"Wrong input shape to network. Should be [batch_size, {self.org_shape[1]}, {self.org_shape[2]}, {self.org_shape[3]}, {self.org_shape[4]}] but got {X.shape}"

        # Reshape to 4 dim
        X_reshaped = X.contiguous().view((-1,) + self.org_shape[2:])  # [batch_size * num_frames_in_stack, RGB, height, width]
        output = self.feature_extractor(X_reshaped.float())  # [batch_size * num_frames_in_stack, num_features]

        # Reshape to 3D
        output_reshaped = output.contiguous().view((-1, self.org_shape[1]) + (output.shape[-1],))  # [8, 10, 128]
        return output_reshaped

class AdaptiveAvgPooling(nn.Module):
    def __init__(self):
        super(AdaptiveAvgPooling, self).__init__()

    def forward(self, x):
        return torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

class Baseline(nn.Module):
    def __init__(self, num_features, num_frames_in_stack, num_airway_segment_classes, batch_size, frame_dimension):
        super().__init__()
        self.num_features = num_features
        self.num_frames_in_stack = num_frames_in_stack
        self.num_airway_segment_classes = num_airway_segment_classes
        self.shape = tuple((batch_size, num_frames_in_stack, 3, frame_dimension[0], frame_dimension[1]))

        # Feature extractor: resnet18
        self.feature_extractor = torchvision.models.alexnet(weights='IMAGENET1K_V1').features # 1000
        self.feature_extractor.flatten = AdaptiveAvgPooling()

        # Handle temporal axis: TimeDistributed
        self.time_distributed = TimeDistributed(self.feature_extractor, self.shape)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_airway_segment_classes),
        )

        # Handle training for certain layers
        for param in self.time_distributed.parameters():
            param.requires_grad = False
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, X):
        # Feature extractor
        X = self.time_distributed(X)  # [batch_size, num_features]

        # Classifier
        predictions = self.classifier(X)  # [batch_size, num_frames_in_stack, airway_classes]

        return predictions

class LSTM(nn.Module):
    def __init__(self, num_features, num_memory_nodes, num_stacked_LSTMs):
        super().__init__()
        self.input_size = num_features
        self.num_memory_nodes = num_memory_nodes
        self.num_stacked_LSTMs = num_stacked_LSTMs
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.num_memory_nodes, num_layers=self.num_stacked_LSTMs, batch_first=True)

        for param in self.lstm.parameters():
            param.requires_grad = True

    def forward(self, X):
        X, _ = self.lstm(X)
        return X

class StatefulLSTM(nn.Module):
    def __init__(self, num_features, num_memory_nodes, num_stacked_LSTMs):
        super(StatefulLSTM, self).__init__()
        self.input_size = num_features
        self.num_memory_nodes = num_memory_nodes
        self.num_stacked_LSTMs = num_stacked_LSTMs
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.num_memory_nodes, num_layers=self.num_stacked_LSTMs, batch_first=True)

        for param in self.lstm.parameters():
                param.requires_grad = True

    def forward(self, X, hidden_state, current_state):
        X, (hidden_state, current_state) = self.lstm(X, (hidden_state, current_state))
        return X, (hidden_state, current_state)

class NoDirectionNavigationNet(nn.Module):
    def __init__(self, num_memory_nodes, num_features, num_stacked_LSTMs, use_stateful_LSTM, num_frames_in_stack,
                 num_airway_segment_classes, batch_size, frame_dimension):
        super().__init__()
        self.num_memory_nodes = num_memory_nodes
        self.num_features = num_features
        self.num_stacked_LSTMs = num_stacked_LSTMs
        self.use_stateful_LSTM = use_stateful_LSTM
        self.num_frames_in_stack = num_frames_in_stack
        self.num_airway_segment_classes = num_airway_segment_classes
        self.shape = tuple((batch_size, num_frames_in_stack, 3, frame_dimension[0], frame_dimension[1]))

        # Feature extractor: Alexnet
        self.feature_extractor = torchvision.models.alexnet(weights='IMAGENET1K_V1').features # 1000
        self.feature_extractor.flatten = AdaptiveAvgPooling()

        # Handle temporal axis: TimeDistributed
        self.time_distributed = TimeDistributed(self.feature_extractor, self.shape)

        # Recurrent Neural Network: LSTM
        if self.use_stateful_LSTM:
            print(" - Using Stateful LSTM -")
            self.lstm = StatefulLSTM(self.num_features, self.num_memory_nodes, self.num_stacked_LSTMs)
        else:
            self.lstm = LSTM(self.num_features, self.num_memory_nodes, self.num_stacked_LSTMs)

        # Classifier Airway
        self.airway_classifier = nn.Sequential(
            nn.Linear(self.num_memory_nodes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_airway_segment_classes),
        )

        # Handle training for certain layers
        for param in self.time_distributed.parameters():
            param.requires_grad = False
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.lstm.parameters():
            param.requires_grad = True
        for param in self.airway_classifier.parameters():
            param.requires_grad = True

    def forward(self, X):
        # Feature extractor
        X = self.time_distributed(X)  # [batch_size, features]

        # Handle Temporal data
        X = self.lstm(X)  # [batch_size, num_frames_in_stack, output_size]

        # Classifier
        airway = self.airway_classifier(X)   # [batch_size, num_frames_in_stack, airway_classes]

        return airway

class DirectionNavigationNet(nn.Module):
    def __init__(self, num_memory_nodes, num_features, num_stacked_LSTMs, use_stateful_LSTM, num_frames_in_stack,
                 num_airway_segment_classes, num_direction_classes, batch_size, frame_dimension):
        super().__init__()
        self.num_memory_nodes = num_memory_nodes
        self.num_features = num_features
        self.num_stacked_LSTMs = num_stacked_LSTMs
        self.use_stateful_LSTM = use_stateful_LSTM
        self.num_frames_in_stack = num_frames_in_stack
        self.num_airway_segment_classes = num_airway_segment_classes
        self.num_direction_classes = num_direction_classes
        self.shape = tuple((batch_size, num_frames_in_stack, 3, frame_dimension[0], frame_dimension[1]))

        # Feature extractor: Alexnet
        self.feature_extractor = torchvision.models.alexnet(weights='IMAGENET1K_V1').features # 1000
        self.feature_extractor.flatten = AdaptiveAvgPooling()

        # Handle temporal axis: TimeDistributed
        self.time_distributed = TimeDistributed(self.feature_extractor, self.shape)

        # Recurrent Neural Network: LSTM
        if self.use_stateful_LSTM:
            print(" - Using Stateful LSTM -")
            self.lstm = StatefulLSTM(self.num_features, self.num_memory_nodes, self.num_stacked_LSTMs)
        else:
            self.lstm = LSTM(self.num_features, self.num_memory_nodes, self.num_stacked_LSTMs)

        # Classifier Airway
        self.airway_classifier = nn.Sequential(
            nn.Linear(self.num_memory_nodes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_airway_segment_classes),
        )

        # Classifier Direction
        self.direction_classifier = nn.Sequential(
            nn.Linear(self.num_memory_nodes, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_direction_classes),
        )

        # Handle training for certain layers
        for param in self.time_distributed.parameters():
            param.requires_grad = False
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.lstm.parameters():
            param.requires_grad = True
        for param in self.airway_classifier.parameters():
            param.requires_grad = True
        for param in self.direction_classifier.parameters():
                param.requires_grad = True

    def forward(self, X):
        # Feature extractor
        X = self.time_distributed(X)  # [batch_size, features]

        # Handle Temporal data
        X = self.lstm(X)  # [batch_size, num_frames_in_stack, output_size]

        # Classifier
        airway = self.airway_classifier(X)   # [batch_size, num_frames_in_stack, airway_classes]

        # Handle an additional classification: Direction
        direction = self.direction_classifier(X)    # [batch_size, num_frames_in_stack, direction_classes]

        return airway, direction

def create_neural_net(num_memory_nodes, num_features_extracted, model_type, num_frames_in_stack, num_airway_segment_classes,
                      num_direction_classes, frame_dimension, batch_size, use_stateful_LSTM, classify_direction):

    print("-- NEURAL NET --")
    print(f"Model: {model_type}")

    # Baseline -> No LSTM and no classifying of direction
    if model_type == 'baseline':
        neural_net = Baseline(num_features=num_features_extracted,
                              num_frames_in_stack=num_frames_in_stack,
                              num_airway_segment_classes=num_airway_segment_classes,
                              batch_size=batch_size,
                              frame_dimension=frame_dimension,)

    # Blomst, Boble or Belle -> with LSTMs
    else:
        # Blomst
        if classify_direction:
            neural_net = DirectionNavigationNet(num_memory_nodes=num_memory_nodes,
                                                num_features=num_features_extracted,
                                                num_stacked_LSTMs=1,
                                                use_stateful_LSTM=use_stateful_LSTM,
                                                num_frames_in_stack=num_frames_in_stack,
                                                num_airway_segment_classes=num_airway_segment_classes,
                                                num_direction_classes=num_direction_classes,
                                                batch_size=batch_size,
                                                frame_dimension=frame_dimension,
                                                )
        # Boble or Belle
        else:
            neural_net = NoDirectionNavigationNet(num_memory_nodes=num_memory_nodes,
                                                  num_features=num_features_extracted,
                                                  num_stacked_LSTMs=1,
                                                  use_stateful_LSTM=use_stateful_LSTM,
                                                  num_frames_in_stack=num_frames_in_stack,
                                                  num_airway_segment_classes=num_airway_segment_classes,
                                                  batch_size=batch_size,
                                                  frame_dimension=frame_dimension,
                                                  )


    return neural_net
