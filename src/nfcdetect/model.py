from .config import EmbeddingConfig
import numpy as np
import sys
import tensorflow as tf
import tensorflow_hub as hub

#PKG_PATH = '/home/yoshi/projects'
#if PKG_PATH not in sys.path:
#    sys.path.append(PKG_PATH)
#import birdnet
import birdnetlib



class AudioEmbeddingModel(object):
    '''Base class of audio embedding models.'''
    def get_dimension(self):
        raise NotImplementedError

    def get_sampling_rate(self):
        raise NotImplementedError
    
    def inference(self, raw_audio):
        raise NotImplementedError


class BirdVocalizationModel(AudioEmbeddingModel):
    MODEL_URL = 'https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/4'
    DIMENSION = 1280
    SR = 32000
    WINDOW = 5 * 32000

    # Minimum proportion of non-padded data in the frame required.
    # Used to determine the number of frames to extract from the recording.
    MIN_COVERAGE = 0.5

    def __init__(self, stride=1.0):
        """
            stride: Stride of sliding window in seconds to extract frames
                    from the recording.
        """
        self.model = hub.load(self.MODEL_URL)
        self.stride_in_sec = stride
    
    def get_dimension(self):
        return self.DIMENSION
    
    def get_sampling_rate(self):
        return self.SR
    
    def get_stride_in_sec(self):
        return self.stride_in_sec
    
    def inference(self, array):
        def stride_array(arr):
            stride = self.stride_in_sec * self.get_sampling_rate()
            strides = (arr.itemsize * stride, arr.itemsize)
            return np.lib.stride_tricks.as_strided(
                arr, shape=(int(np.floor((arr.size - window) / stride)) + 1, window), strides=strides)
        
        # Pad array
        # Compute last permissible position
        stride = int(self.stride_in_sec * self.get_sampling_rate())
        max_pos = array.size - np.ceil(self.WINDOW * self.MIN_COVERAGE).astype(int)
        num_frames = np.floor(max_pos / stride).astype(int) + 1
        last_pos = (num_frames - 1) * stride
        # Pad zeros at the end
        if last_pos + self.WINDOW > array.size:
            array = np.append(array, np.zeros(last_pos + self.WINDOW - array.size))
        
        strides = (array.itemsize * stride, array.itemsize)
        frames = np.lib.stride_tricks.as_strided(array, shape=(num_frames, self.WINDOW), strides=strides)

        _, embeddings = self.model.infer_tf(frames)
        return embeddings


class BirdNetModel(AudioEmbeddingModel):
    """Wrapper class of BirdNet model."""

    def __init__(self, stride=1.0):
        """
            stride: Stride of sliding window in seconds to extract frames
                    from the recording.
        """
        self.stride_in_sec = stride
        self.analyzer = birdnetlib.analyzer.Analyzer()
        
        # Hack needed to make _return_embeddings() actually return embeddings instead of logits
        self.analyzer.output_layer_index -= 1
    
    def embed(self, frames):
        return self.analyzer._return_embeddings(frames)


def create_embedding_model(config: EmbeddingConfig) -> AudioEmbeddingModel:
    if config.name == 'birdnet':
        return BirdNetModel()
    elif config.name == 'perch':
        return BirdVocalizationModel()
