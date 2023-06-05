import fast
import numpy as np


fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)


class CustomImagesToSequence(fast.PythonProcessObject):
    """ Given an image stream as input, the CustomImagesToSequence loads an image stack with `sequence_size` images. """
    def __init__(self, sequence_size=50):
        super().__init__()
        self.createInputPort(0) # Input: Frame number
        self.createOutputPort(0) # Output: Image stack

        self._sequence_size = sequence_size
        self.queue = [] # Shape: (n_frames, *self.image_shape)
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


class StackClassificationToText(fast.PythonProcessObject):
    """
    Parameters
    ----------
    labels : dict
        Dictionary mapping integer label to label text (e.g. {0: 'A', 1: 'B'})
    Input port
    ----------
    classifications : list/tuple with shape (n_frames in stack, 2)
    Classification model output. For each frame in stack, gives classification labels (airway, direction).
    Output port
    -----------
    classification for last frame : fast.Text Contains information on airway and direction classification for last frame in stack
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

        output_text = f'{self.name if hasattr(self, "name") else "":<16}'

        if self.labels is None:
            output_text += f'{pred:<4}\n'

        else:
            output_text += f'{pred:<4}{self.labels[pred]:<40}\n'

        self.addOutputData(0, fast.Text.create(output_text))


class TextOutputMerger(fast.PythonProcessObject):
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


class ImageAndClassificationWindow(object):
    is_running = False

    def __init__(self, data_path, classification_model_path, airway_labels, direction_labels, stack_size, framerate=-1):

        # Get image stream and convert to stack
        self.streamer = fast.ImageFileStreamer.create(data_path, loop=True, framerate=framerate)
        self.image_to_sequence = CustomImagesToSequence.create(sequence_size=stack_size)
        self.image_to_sequence.connect(self.streamer)

        # Neural network model
        self.classification_model = fast.NeuralNetwork.create(classification_model_path, scaleFactor=1. / 255.)

        self.classification_model.connect(self.image_to_sequence)

        # Classification (neural network output) to Text
        self.airway_classification_to_text = StackClassificationToText.create(name='Airway', labels=airway_labels)
        self.direction_classification_to_text = StackClassificationToText.create(name='Direction', labels=direction_labels)
        self.text_merger = TextOutputMerger.create()
        self.airway_classification_to_text.connect(0, self.classification_model, 0)
        self.direction_classification_to_text.connect(0, self.classification_model, 1)
        self.text_merger.connect(0, self.airway_classification_to_text)
        self.text_merger.connect(1, self.direction_classification_to_text)

        # Renderers
        self.image_renderer = fast.ImageRenderer.create().connect(self.streamer)
        self.classification_renderer = fast.TextRenderer.create(fontSize=48)
        self.classification_renderer.connect(self.text_merger)

        # Set up video window
        self.window = fast.DualViewWindow2D.create(
            width=1000,
            height=1200,
            bgcolor=fast.Color.Black(),
            verticalMode=True # defaults to False
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


def run_testing_realtime(data_path, trained_model_path, airway_labels, direction_labels, num_frames_in_test_stack):

    fast_classification = ImageAndClassificationWindow(
        data_path=data_path,
        classification_model_path=trained_model_path,
        airway_labels=airway_labels,
        direction_labels=direction_labels,
        stack_size=num_frames_in_test_stack,
        framerate=5,
    )
    fast_classification.window.run()
