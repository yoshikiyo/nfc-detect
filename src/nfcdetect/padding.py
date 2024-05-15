"""This module defines various padding methods to be used with audio embedding models."""

import numpy as np

from .config import ClassifierConfig

class PaddingMethod(object):
    '''Defines padding method.'''

    def __init__(self, output_size=32000*5, position=0, sr=32000, mask_pattern=np.zeros(1)):
        self.output_size = output_size
        self.position = position
        self.sr = sr
        self.mask_pattern = mask_pattern

    def pad(self, data):
        '''Returns array of data with padding applied.'''
        output = np.tile(
            self.mask_pattern,
            int(np.ceil(float(self.output_size)/len(self.mask_pattern))))[:self.output_size]

        if self.position + len(data) > len(output):
            raise ValueError(
                f'Invalid position: {self.position} (|input|={len(data)}, |output|={len(output)}')
        output[self.position:self.position+len(data)] = data
        return output


class RepeatPadding(PaddingMethod):
    '''Fill frame by repeating the input audio'''
    
    def __init__(self, output_size=32000*5, sr=32000):
        self.output_size = output_size
        self.sr = sr

    def pad(self, audio):
        output = np.tile(audio,
                         int(np.ceil(float(self.output_size)/len(audio))))[:self.output_size]
        return output


def padding_from_config(cfg: ClassifierConfig) -> PaddingMethod:
    '''Constructs a PaddingMethod object based on config.'''
    if cfg.padding_method == 'repeat':
        frame_size = cfg.embedding_config.frame_size
        sampling_rate = cfg.embedding_config.sampling_rate
        return RepeatPadding(output_size=int(frame_size * sampling_rate), sr=sampling_rate)

    raise ValueError(f'Unknown padding method: {cfg.padding_method}')
