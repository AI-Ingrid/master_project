import fast
import numpy as np


# *********** FILE STREAMING ***********
class CustomImagesToSequence(fast.PythonProcessObject):
    """
    Given an image stream as input, the CustomImagesToSequence loads an image
    stack with `sequence_size` images.
    """

    def __init__(self, sequence_size=50):
        super().__init__()
        self.createInputPort(0)  # Input:  Frame number
        self.createOutputPort(0)  # Output: Image stack (TODO: ndarray or fast.Image?)

        self._sequence_size = sequence_size
        self.queue = []  # Shape: (n_frames, *self.image_shape)

    @property
    def sequence_size(self):
        return self._sequence_size

    def execute(self):
        new_image = self.getInputData(0)

        if len(self.queue) > 0:
            self.queue = self.queue[1:]
        while len(self.queue) < self.sequence_size:
            self.queue.append(new_image)

        self.addOutputData(0, fast.Sequence.create(self.queue))


# *********** CLASSIFICATION HANDLING ***********
class HiddenState(fast.PythonProcessObject):

    def __init__(self):
        super().__init__()
        self.createInputPort(0)
        self.createInputPort(1)
        self.createOutputPort(0)
        self.createOutputPort(1)

        self.is_initialized = False

    def execute(self):
        if self.is_initialized:
            input1 = self.getInputData(0)
            input2 = self.getInputData(1)
        else:
            input1 = fast.Image.createFromArray(np.zeros(shape=(1, 1 , 128)))
            input2 = fast.Image.createFromArray(np.zeros(shape=(1, 1 , 128)))
            self.is_initialized = True

        self.addOutputData(0, input1)
        self.addOutputData(1, input2)


class StackClassificationToText(fast.PythonProcessObject):
    """

    Parameters
    ----------
    labels : dict
        Dictionary mapping integer label to label text (e.g. {0: 'A', 1: 'B'})

    Input port
    ----------
    classifications : list/tuple with shape (n_frames in stack, 2)
        Classification model output. For each frame in stack, gives
        classification labels (airway, direction).

    Output port
    -----------
    classification for last frame : fast.Text
        Contains information on airway and direction classification for last frame in stack
    """

    def __init__(self, labels=None, name=None):
        super().__init__()
        self.createInputPort(0)
        self.createOutputPort(0)

        if labels is not None:
            self.labels = labels
        if name is not None:
            self.name = name

    def execute(self):

        classification = self.getInputData(0)
        classification_arr = np.asarray(classification)

        # Output: Return only classification (airway, direction) for last frame in stack
        prediction = classification_arr[:, -1]
        pred = np.argmax(prediction)

        output_text = f'{self.name + ": " if hasattr(self, "name") else "":>16}'
        if self.labels is None:
            output_text += f'{pred:<4}\n'
        else:
            output_text += f'{self.labels[pred]:<40}\n'
        self.addOutputData(0, fast.Text.create(output_text))


class TextOutputMergerDirection(fast.PythonProcessObject):

    def __init__(self):
        super().__init__()
        self.createInputPort(0)
        self.createInputPort(1)
        self.createOutputPort(0)

    def execute(self):

        airway = self.getInputData(0)
        direction = self.getInputData(1)

        combined = airway.getText() + direction.getText()
        self.addOutputData(0, fast.Text.create(combined, color=fast.Color.White()))


class TextOutputMerger(fast.PythonProcessObject):

    def __init__(self):
        super().__init__()
        self.createInputPort(0)
        self.createOutputPort(0)

    def execute(self):

        airway = self.getInputData(0)

        combined = airway.getText()
        self.addOutputData(0, fast.Text.create(combined, color=fast.Color.White()))


# *********** VISUALIZATION WINDOW ***********
class VideoTrackingClassificationWindowDirection(object):
    is_running = False

    airway_labels = {
        0: 'Trachea',
        1: 'Right Main Bronchus',
        2: 'Left Main Bronchus',
        3: 'Right/Left Upper Lobe Bronchus',
        4: 'Right Truncus Intermedicus',
        5: 'Left Lower Lobe Bronchus',
        6: 'Left Upper Lobe Bronchus',
        7: 'Right B1',
        8: 'Right B2',
        9: 'Right B3',
        10: 'Right Middle Lobe Bronchus (parent for B4 og B5)',
        11: 'Right lower Lobe Bronchus (1)',
        12: 'Right Lower Lobe Bronchus (2)',
        13: 'Left Main Bronchus',
        14: 'Left B6',
        15: 'Left Upper Division Bronchus',
        16: 'Left Lingular Bronchus',
        17: 'Right B4',
        18: 'Right B5',
        19: 'Left B1+B2',
        20: 'Left B3',
        21: 'Left B4',
        22: 'Left B5',
        23: 'Left B8',
        24: 'Left B9',
        25: 'Left B10',
    }
    direction_labels = {
        1: 'forward',
        0: 'backward'
    }

    def __init__(self, data_path, classification_model_path, sequence_size=5, framerate=-1):
        # Get image stream and convert to stack
        self.streamer = fast.ImageFileStreamer.create(data_path, loop=True, framerate=framerate)
        # self.image_to_sequence = CustomImagesToSequence.create(sequence_size=sequence_size)
        self.image_to_sequence = fast.ImagesToSequence.create(sequenceSize=sequence_size)
        self.image_to_sequence.connect(self.streamer)

        # Neural network model
        self.inputNode = fast.NeuralNetworkNode('input', fast.NodeType_TENSOR, fast.TensorShape((1, sequence_size, 3, 256, 256)), 0)     # INPUT NODE
        self.inputHiddenState = fast.NeuralNetworkNode('hidden_state_in', fast.NodeType_TENSOR, fast.TensorShape((1, 1, 256)), 1)     # INPUT NODE
        self.inputCellState = fast.NeuralNetworkNode('cell_state_in', fast.NodeType_TENSOR, fast.TensorShape((1, 1, 256)), 2)     # INPUT NODE

        self.outputAirway = fast.NeuralNetworkNode('airway', fast.NodeType_TENSOR, fast.TensorShape((1, sequence_size, 26)), 0)          # AIRWAY
        self.outputDirection = fast.NeuralNetworkNode('direction', fast.NodeType_TENSOR, fast.TensorShape((1, sequence_size, 2)), 1)    # DIRECTION
        self.outputHiddenState = fast.NeuralNetworkNode('hidden_state_out', fast.NodeType_TENSOR, fast.TensorShape((1, 1, 256)), 2)  # INPUT NODE
        self.outputCellState = fast.NeuralNetworkNode('cell_state_out', fast.NodeType_TENSOR, fast.TensorShape((1, 1, 256)), 3)  # INPUT NODE

        self.classification_model = fast.NeuralNetwork.create(classification_model_path, scaleFactor=1. / 255.,)

        # for hidden state/stateful network
        self.classification_model.addTemporalState(
            inputNodeName='hidden_state_in',
            outputNodeName='hidden_state_out',
        )
        # for hidden state/stateful network
        self.classification_model.addTemporalState(
            inputNodeName='cell_state_in',
            outputNodeName='cell_state_out',
        )

        self.classification_model.connect(0, self.image_to_sequence)

        # Classification (neural network output) to Text
        self.airway_classification_to_text = StackClassificationToText.create(name='Airway', labels=self.airway_labels)
        self.direction_classification_to_text = StackClassificationToText.create(name='Direction', labels=self.direction_labels)
        self.airway_classification_to_text.connect(0, self.classification_model, 0)
        self.direction_classification_to_text.connect(0, self.classification_model, 1)

        self.text_merger = TextOutputMergerDirection.create()
        self.text_merger.connect(0, self.airway_classification_to_text)
        self.text_merger.connect(1, self.direction_classification_to_text)

        # Renderers
        self.image_renderer = fast.ImageRenderer.create().connect(self.streamer)

        self.classification_renderer = fast.TextRenderer.create(fontSize=48)
        self.classification_renderer.connect(self.text_merger)

        # Set up video window
        self.window = fast.DualViewWindow2D.create(
            width=800,
            height=900,
            bgcolor=fast.Color.Black(),
            verticalMode=True  # defaults to False
        )

        self.window.connectTop([self.classification_renderer])
        self.window.addRendererToTopView(self.classification_renderer)
        self.window.connectBottom([self.image_renderer])
        self.window.addRendererToBottomView(self.image_renderer)

        # Set up playback widget
        self.widget = fast.PlaybackWidget(streamer=self.streamer)
        self.window.connect(self.widget)

    def run(self):
        self.window.run()


class VideoTrackingClassificationWindow(object):
    is_running = False

    airway_labels = {
        0: 'Trachea',
        1: 'Right Main Bronchus',
        2: 'Left Main Bronchus',
        3: 'Right/Left Upper Lobe Bronchus',
        4: 'Right Truncus Intermedicus',
        5: 'Left Lower Lobe Bronchus',
        6: 'Left Upper Lobe Bronchus',
        7: 'Right B1',
        8: 'Right B2',
        9: 'Right B3',
        10: 'Right Middle Lobe Bronchus (parent for B4 og B5)',
        11: 'Right lower Lobe Bronchus (1)',
        12: 'Right Lower Lobe Bronchus (2)',
        13: 'Left Main Bronchus',
        14: 'Left B6',
        15: 'Left Upper Division Bronchus',
        16: 'Left Lingular Bronchus',
        17: 'Right B4',
        18: 'Right B5',
        19: 'Left B1+B2',
        20: 'Left B3',
        21: 'Left B4',
        22: 'Left B5',
        23: 'Left B8',
        24: 'Left B9',
        25: 'Left B10',
    }

    def __init__(self, data_path, classification_model_path, sequence_size=5, framerate=-1):
        # Get image stream and convert to stack
        self.streamer = fast.ImageFileStreamer.create(data_path, loop=True, framerate=framerate)
        # self.image_to_sequence = CustomImagesToSequence.create(sequence_size=sequence_size)
        self.image_to_sequence = fast.ImagesToSequence.create(sequenceSize=sequence_size)
        self.image_to_sequence.connect(self.streamer)

        # Neural network model
        self.inputNode = fast.NeuralNetworkNode('input', fast.NodeType_TENSOR,
                                                fast.TensorShape((1, sequence_size, 3, 256, 256)), 0)  # INPUT NODE
        self.inputHiddenState = fast.NeuralNetworkNode('hidden_state_in', fast.NodeType_TENSOR,
                                                       fast.TensorShape((1, 1, 256)), 1)  # INPUT NODE
        self.inputCellState = fast.NeuralNetworkNode('cell_state_in', fast.NodeType_TENSOR,
                                                     fast.TensorShape((1, 1, 256)), 2)  # INPUT NODE

        self.outputAirway = fast.NeuralNetworkNode('airway', fast.NodeType_TENSOR,
                                                   fast.TensorShape((1, sequence_size, 26)), 0)  # AIRWAY

        self.outputHiddenState = fast.NeuralNetworkNode('hidden_state_out', fast.NodeType_TENSOR,
                                                        fast.TensorShape((1, 1, 256)), 2)  # INPUT NODE
        self.outputCellState = fast.NeuralNetworkNode('cell_state_out', fast.NodeType_TENSOR,
                                                      fast.TensorShape((1, 1, 256)), 3)  # INPUT NODE

        self.classification_model = fast.NeuralNetwork.create(classification_model_path, scaleFactor=1. / 255., )

        # for hidden state/stateful network
        self.classification_model.addTemporalState(
            inputNodeName='hidden_state_in',
            outputNodeName='hidden_state_out',
        )
        # for hidden state/stateful network
        self.classification_model.addTemporalState(
            inputNodeName='cell_state_in',
            outputNodeName='cell_state_out',
        )

        self.classification_model.connect(0, self.image_to_sequence)

        # Classification (neural network output) to Text
        self.airway_classification_to_text = StackClassificationToText.create(name='Airway', labels=self.airway_labels)
        self.airway_classification_to_text.connect(0, self.classification_model, 0)

        self.text_merger = TextOutputMerger.create()
        self.text_merger.connect(0, self.airway_classification_to_text)

        # Renderers
        self.image_renderer = fast.ImageRenderer.create().connect(self.streamer)

        self.classification_renderer = fast.TextRenderer.create(fontSize=48)
        self.classification_renderer.connect(self.text_merger)

        # Set up video window
        self.window = fast.DualViewWindow2D.create(
            width=800,
            height=900,
            bgcolor=fast.Color.Black(),
            verticalMode=True  # defaults to False
        )

        self.window.connectTop([self.classification_renderer])
        self.window.addRendererToTopView(self.classification_renderer)
        self.window.connectBottom([self.image_renderer])
        self.window.addRendererToBottomView(self.image_renderer)

        # Set up playback widget
        self.widget = fast.PlaybackWidget(streamer=self.streamer)
        self.window.connect(self.widget)

    def run(self):
        self.window.run()


def run_realtime_evaluation_in_FAST(model_type):

    fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)
    data_path = "/Users/ikolderu/PycharmProjects/master/data/synthetic/synthetic_frames2/Patient_001/Sequence_020/frame_#.png"

    if model_type == 'blomst':
        model_path = "/Users/ikolderu/blomst_50_5_features_256_hidden_256_epochs_5000_focal_loss_True_stateful_False_direction_False.onnx"
        fast_classification = VideoTrackingClassificationWindow(
            data_path=data_path,
            classification_model_path=model_path,
            sequence_size=2,
            framerate=10
        )
        fast_classification.window.run()

    elif model_type == 'boble':
        model_path = "/Users/ikolderu/PycharmProjects/master/onnx_models/boble_50_5_features_256_hidden_256_epochs_5000_focal_loss_True_stateful_False_direction_True.onnx"
        fast_classification = VideoTrackingClassificationWindowDirection(
            data_path=data_path,
            classification_model_path=model_path,
            sequence_size=2,
            framerate=10
        )
        fast_classification.window.run()

    elif model_type == 'belle':
        model_path= " "



run_realtime_evaluation_in_FAST("blomst")

