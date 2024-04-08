"""This module defines various padding methods to be used with audio embedding models."""

import numpy as np

class PaddingMethod(object):
    '''Defines padding method.'''

    def __init__(self, output_size=32000*5, position=0, sr=32000, mask_pattern=np.zeros(1)):
        self.output_size = output_size
        self.position = position
        self.sr = sr
        self.mask_pattern = mask_pattern

    def pad(self, input):
        output = np.tile(self.mask_pattern,
                         int(np.ceil(float(self.output_size)/len(self.mask_pattern))))[:self.output_size]

        if self.position + len(input) > len(output):
            raise ValueError(f'Invalid position: {self.position} (|input|={len(input)}, |output|={len(output)}')
        output[self.position:self.position+len(input)] = input
        return output


class RepeatPadding(PaddingMethod):
    def __init__(self, output_size=32000*5, sr=32000):
        self.output_size = output_size
        self.sr = sr

    def pad(self, input):
        output = np.tile(input,
                         int(np.ceil(float(self.output_size)/len(input))))[:self.output_size]
        return output
