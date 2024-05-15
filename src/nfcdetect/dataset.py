'''Module that defines functionalities to extract features from input data.'''

import csv
import os
import random

import librosa
import numpy as np
import pydantic
import tensorflow as tf

from .config import ClassifierConfig
from .model import create_embedding_model
from .padding import padding_from_config


class Example:
    '''Class to hold information of an example.'''

    filename: str
    label: str
    label_start: float
    label_duration: float
    sample_start: float
    audio: np.ndarray
    embedding: np.ndarray

    def __init__(self, filename, label, label_start, label_duration, sample_start, audio,
                 embedding=None):
        self.filename = filename
        self.label = label
        self.label_start = label_start
        self.label_duration = label_duration
        self.sample_start = sample_start
        self.audio = audio
        self.embedding = embedding

    def __repr__(self):
        return str(
            [self.filename,
             self.label,
             self.label_start,
             self.label_duration,
             self.sample_start,
             f'audio: array[{len(self.audio)}]',
             f'embedding: array[{len(self.embedding)}]'
            ]
        )

    def bytes_feature(self, value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        if isinstance(value, str):
            value = value.encode()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def to_tf_example(self) -> tf.train.Example:
        '''Converts itself into a TensorfFlow Example object.'''
        feature = {
            'filename': self.bytes_feature(self.filename),
            'label': self.bytes_feature(self.label),
            'label_start': self.float_feature(self.label_start),
            'label_duration': self.float_feature(self.label_duration),
            'sample_start': self.float_feature(self.sample_start),
            'audio': self.bytes_feature(self.audio.tobytes()),
            'embedding': self.bytes_feature(self.embedding.tobytes())
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @staticmethod
    def from_tf_example(example: tf.train.Example):
        '''Constructs an instance of Example class from TensorFlow Example.'''
        feature = example.features.feature
        return Example(
            filename=feature['filename'].bytes_list.value[0].decode(),
            label=feature['label'].bytes_list.value[0].decode(),
            label_start=feature['label_start'].float_list.value[0],
            label_duration=feature['label_duration'].float_list.value[0],
            sample_start=feature['sample_start'].float_list.value[0],
            audio=np.frombuffer(feature['audio'].bytes_list.value[0], dtype=np.float32),
            embedding=np.frombuffer(feature['embedding'].bytes_list.value[0], dtype=np.float32)
        )


class AudioSet:
    '''Information about audio files in dataset.'''

    def __init__(self):
        self.files = []
        self.durations = []

    def add_file(self, filename, duration):
        self.files.append(filename)
        self.durations.append(duration)

    def sample_files(self, n):
        '''Returns sample of files weighted by their duration.'''
        # Sample files weighted by its duration
        sample_files = random.choices(self.files, weights=self.durations, k=n)
        files, sample_counts = np.unique(sample_files, return_counts=True)
        return files, sample_counts


class PreprocessOptions(pydantic.BaseModel):
    '''Options for prepare_dataset()'''

    cfg: ClassifierConfig
    jitter: bool = False
    augmentation_per_sample: int = 0
    negative_samples: int = 0
    negative_label: str = 'neg'
    negative_duration: float = 0.2


def prepare_dataset(annotation_file: str,
                    data_dir: str,
                    output: str,
                    options: PreprocessOptions,
                    batch_size=256):
    '''Generate dataset from audio data with annotations.'''

    audio_set = AudioSet()

    # Set up padding and embedding model
    padding = padding_from_config(options.cfg)
    embedding_model = create_embedding_model(options.cfg.embedding_config)

    # Process labeled data
    file2annotation = {}  # filename -> list of tuple of (label, start_time, duration)

    # Columns of annotation CSV file:
    #   filename
    #   label
    #   start
    #   duration
    with open(annotation_file, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            record = (row['label'], float(row['start']), float(row['duration']))
            if filename not in file2annotation:
                file2annotation[filename] = [record]
            else:
                file2annotation[filename].append(record)

    # Extract audio per annotation
    examples = []
    for filename, records in file2annotation.items():
        target_file = os.path.join(data_dir, filename)
        if not os.path.exists(target_file):
            print(f'Target file does not exist: {target_file}')
            print(f'Discarding {len(records)} examples')
            continue

        total_duration = librosa.get_duration(path=target_file)
        audio_set.add_file(filename, total_duration)

        for record in records:
            start = record[1]
            duration = record[2]
            if start < 0 or start + duration > total_duration:
                print(f'Segment out of bounds: {record} (total_duration={total_duration})')

            sample_per_record = max(options.augmentation_per_sample + 1, 1)
            # Repeat extracting samples per augementation_per_sample
            for _ in range(sample_per_record):
                # Apply jitter to crop position
                if options.jitter:
                    min_pos = start + duration - options.cfg.crop_size
                    max_pos = start
                    crop_start = np.random.random_sample() * (max_pos - min_pos) + min_pos
                elif duration < options.cfg.crop_size:
                    # If no jitter, place the annotated segement at the center of frame
                    crop_start = start + (options.cfg.crop_size - duration) * 0.5
                else:
                    # If duration is longer than frame size, fill the frame from the beginning of
                    # annotated segment and discard the rest.
                    crop_start = start

                audio, _ = librosa.load(target_file,
                                        offset=crop_start,
                                        sr=options.cfg.embedding_config.sampling_rate,
                                        duration=options.cfg.crop_size)
                examples.append(Example(filename=filename,
                                        label=record[0],
                                        label_start=start,
                                        label_duration=duration,
                                        sample_start=crop_start,
                                        audio=audio))

    # Sample negative examples
    if options.negative_samples > 0:
        files, counts = audio_set.sample_files(options.negative_samples)
        for filename, count in zip(files, counts):
            target_file = os.path.join(data_dir, filename)
            duration = librosa.get_duration(path=target_file)

            for neg_offset in np.random.random_sample(count) * (duration - options.cfg.crop_size):
                audio, _ = librosa.load(target_file,
                                        offset=neg_offset,
                                        sr=options.cfg.embedding_config.sampling_rate,
                                        duration=options.cfg.crop_size)
                examples.append(Example(filename=filename,
                                        label=options.negative_label,
                                        label_start=neg_offset,
                                        label_duration=options.negative_duration,
                                        sample_start=neg_offset,
                                        audio=audio))

    # Compute embedding per annotation
    num_examples = len(examples)
    all_embeddings = None
    print('num_examples=', num_examples)

    num_batches = np.ceil(num_examples / batch_size).astype(int)
    print('num_batches=', num_batches)
    for batch_id in range(num_batches):
        start_idx = batch_id * batch_size
        end_idx = min(start_idx+batch_size, num_examples)
        batch = examples[start_idx:end_idx]
        frames = [padding.pad(example.audio) for example in batch]

        # Extract features
        embeddings = embedding_model.embed(frames)
        all_embeddings = (np.concatenate((all_embeddings, embeddings))
                          if all_embeddings is not None else embeddings)

    # Write output file
    record_options = tf.io.TFRecordOptions(compression_type='GZIP')
    with tf.io.TFRecordWriter(output, record_options) as writer:
        for i in range(num_examples):
            example = examples[i]
            example.embedding = all_embeddings[i,:]
            writer.write(example.to_tf_example().SerializeToString())
    writer.close()


def read_dataset(dataset_file: str):
    '''Read dataset from file.'''

    examples = []
    records = tf.data.TFRecordDataset([dataset_file], compression_type='GZIP')
    for record in records:
        tf_example = tf.train.Example()
        tf_example.ParseFromString(record.numpy())
        examples.append(Example.from_tf_example(tf_example))
    return examples
