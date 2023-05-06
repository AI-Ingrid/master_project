import logging
import fast
import numpy as np

from logger import init_logger, close_logger

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

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
        # logger.info(f'Prediction: {prediction}, type = {type(prediction)}')
        pred = np.argmax(prediction)
        # logger.info(f'Argmax prediction: {pred}')

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
        logger.info('\nClassification\n' + combined)
        self.addOutputData(0, fast.Text.create(combined, color=fast.Color.White()))
class ImageAndClassificationWindow(object):
    is_running = False
    def __init__(self, data_path, classification_model_path, framerate=-1):
        # Get image stream and convert to stack
        self.streamer = fast.ImageFileStreamer.create(data_path, loop=True, framerate=framerate)
        self.image_to_sequence = CustomImagesToSequence.create(sequence_size=50)
        self.image_to_sequence.connect(self.streamer)

        # Neural network model
        self.classification_model = fast.NeuralNetwork.create(classification_model_path, scaleFactor=1. / 255.)
        self.classification_model.connect(self.image_to_sequence)

        for nodeStr, nodePtr in self.classification_model.getInputNodes().items():
            logger.info(f'{"INPUT":<10}\t{nodeStr:<10}\t{nodePtr.shape.toString()}')
        for nodeStr, nodePtr in self.classification_model.getOutputNodes().items():
            logger.info(f'{"OUTPUT":<10}\t{nodeStr:<10}\t{nodePtr.shape.toString()}')

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

if __name__ == '__main__':
    init_logger('application')
    data_path = "C:/data/Lung/Bronkoskopivideo_syntetisk/Frames/Patient_001/Sequence_001/frame_#.png"
    model_path = "C:/data/Lung/Bronkoskopivideo_syntetisk/blomst_3.onnx"

    airway_labels = { 0: 'Other/unknown', 1: 'Trachea', 2: 'Right Main Bronchus', 3: 'Left Main Bronchus',
                      4: 'Right/Left Upper Lobe Bronchus', 5: 'Right Truncus Intermedicus', 6: 'Left Lower Lobe Bronchus',
                      7: 'Left Upper Lobe Bronchus', 8: 'Right B1', 9: 'Right B2', 10: 'Right B3',
                      11: 'Right Middle Lobe Bronchus (parent for B4 og B5)', 12: 'Right lower Lobe Bronchus (possible called right lower lobe bronchus (1))',
                      13: 'Right Lower Lobe Bronchus (2)', 14: 'Left Main Bronchus', 15: 'Left B6', 16: 'Left Upper Division Bronchus',
                      17: 'Left Lingular Bronchus', 18: 'Right B4', 19: 'Right B5', 20: 'Left B1+B2', 21: 'Left B3', 22: 'Left B4',
                      23: 'Left B5', 24: 'Left B8', 25: 'Left B9', 26: 'Left B10', }

    direction_labels = {
        1: 'forward',
        0: 'backward' }

    fast_classification = ImageAndClassificationWindow(
        data_path=data_path,
        classification_model_path=model_path,
        framerate=5
    )
    fast_classification.window.run()

    close_logger()

