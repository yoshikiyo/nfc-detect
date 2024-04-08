"""This module defines classifier class."""

from .config import ClassifierConfig
from .model import create_embedding_model
from .padding import RepeatPadding
import numpy as np
import joblib

class Classifier(object):
    """Base class for classifiers"""

    def __init__(self):
        raise NotImplementedError


class EmbeddingClassifier(Classifier):
    """Classifier based on embeddings."""

    def __init__(self, classifier_config: ClassifierConfig):
        self.cfg = classifier_config
        self.cls_model = joblib.load(self.cfg.classifier_model_path)
        self.embedding_model = create_embedding_model(self.cfg.embedding_config)
        if self.cfg.padding_method == 'repeat':
            frame_size = self.cfg.embedding_config.frame_size
            sampling_rate = self.cfg.embedding_config.sampling_rate
            self.padding = RepeatPadding(output_size=int(frame_size * sampling_rate), sr=sampling_rate)

    def predict(self, input: np.ndarray) -> np.ndarray:
        # Split into frames

        # Compute last permissible position
        sr = self.cfg.embedding_config.sampling_rate
        window = int(self.cfg.crop_size * sr)

        stride = window
        num_frames = np.ceil(input.size / window).astype(int)
        last_pos = (num_frames - 1) * stride
        # Pad zeros at the end
        if last_pos + window > input.size:
            input = np.append(input, np.zeros(last_pos + window - input.size))
        print(f'input.size={input.size}, stride={stride}, num_frames={num_frames}, last_pos={last_pos}')

        strides = (input.itemsize * stride, input.itemsize)
        frames = np.lib.stride_tricks.as_strided(input, shape=(num_frames, window), strides=strides)

        # Apply padding
        frames = [self.padding.pad(frame) for frame in list(frames)]
        
        #print(list(frames))

        # Extract features
        embeddings = self.embedding_model.embed(frames)

        # Apply classifier to embeddings and extract probability
        prob = [prediction[1] for prediction in self.cls_model.predict_proba(embeddings)]
        
        return prob