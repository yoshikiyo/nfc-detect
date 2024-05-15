from .config import EmbeddingConfig

import birdnetlib
import numpy as np
import sys
import tensorflow as tf
import tensorflow_hub as hub


class AudioEmbeddingModel(object):
    '''Base class of audio embedding models.'''
    def embed(self, frames):
        raise NotImplementedError


class BirdVocalizationModel(AudioEmbeddingModel):
    MODEL_URL = 'https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/4'

    def __init__(self):
        self.model = hub.load(self.MODEL_URL)
    
    def embed(self, frames):
        _, embeddings = self.model.infer_tf(frames)
        return embeddings


class BirdNetModel(AudioEmbeddingModel):
    """Wrapper class of BirdNet model."""

    def __init__(self):
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
    else:
        raise ValueError(f'Uknown model type: {config.name}')
