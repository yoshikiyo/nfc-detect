"""This module defines classifier class."""

import sklearn.linear_model
from .config import ClassifierConfig
from .dataset import Example
from .model import (create_embedding_model, AudioEmbeddingModel)
from .padding import (padding_from_config, PaddingMethod)

import os

import numpy as np
import joblib
import sklearn
import tensorflow as tf


def extract_feature_and_label(examples: list[Example]):
    '''Extracts features and labels from list of Example.'''
    x = [example.embedding for example in examples]
    y = [example.label for example in examples]
    return x, y


def compute_pr_auc(m, X_test, y_test):
    '''Computes PR-AUC for model `m' using X_test and y_test'''
    p, r, t = sklearn.metrics.precision_recall_curve(
        y_test, [elem[0] for elem in m.predict_proba(X_test)], pos_label='call')
    auc = sklearn.metrics.auc(r, p)
    return p, r, t, auc


class EmbeddingClassifier:
    '''Classifier based on embeddings.'''

    def __init__(self):
        self.cfg = None
        self.config_dir = '.'
        self.padding = None
        self.embedding_model = None
        self.cls_model = None

    def load_models(self, classifier_config: ClassifierConfig, config_dir: str = '.'):
        '''Load models from config.'''
        self.cfg = classifier_config
        self.config_dir = config_dir
        self.padding = padding_from_config(self.cfg)
        self.embedding_model = create_embedding_model(self.cfg.embedding_config)
        self.cls_model = joblib.load(
            os.path.join(self.config_dir, self.cfg.classifier_model_path))

    def save(self, model_name: str, outdir: str = '.'):
        '''Saves classification model and the configuration.'''
        if not self.cfg:
            raise AssertionError('ClassifierConfig is not set up. '
                                 'Call set_config() before saving the model.')
        if not self.cls_model:
            raise AssertionError('Classifier is empty. '
                                 'Train one with train() or load one with load_models().')

        config_file = f'{model_name}_config.json'
        joblib_file = f'{model_name}.joblib'

        joblib.dump(self.cls_model, os.path.join(outdir, joblib_file))
        self.cfg.classifier_model_path = joblib_file
        json = self.cfg.model_dump_json()
        with open(os.path.join(outdir, config_file), 'w', encoding='utf-8') as f:
            f.write(json)

    def set_config(self, config: ClassifierConfig):
        self.cfg = config

    def get_classes(self):
        if not self.cls_model:
            raise AssertionError('Classifier not initialized.')
        return self.cls_model.classes_

    def predict(self, audio: np.ndarray) -> np.ndarray:
        '''Classify audio input.
        
        Assumes `audio' is raw audio sample with a sampling rate specified in config.
        '''

        sr = self.cfg.embedding_config.sampling_rate
        window = int(self.cfg.crop_size * sr)

        stride = window
        num_frames = np.ceil(audio.size / window).astype(int)
        last_pos = (num_frames - 1) * stride

        # Pad zeros at the end
        if last_pos + window > audio.size:
            audio = np.append(audio, np.zeros(last_pos + window - audio.size))

        strides = (audio.itemsize * stride, audio.itemsize)
        frames = np.lib.stride_tricks.as_strided(audio, shape=(num_frames, window), strides=strides)

        # Apply padding
        frames = [self.padding.pad(frame) for frame in list(frames)]

        # Extract features
        embeddings = self.embedding_model.embed(frames)

        return self.cls_model.predict_proba(embeddings), self.get_classes()

    def train(self,
              train_set: list[Example],
              classifier_type: str,
              regularization: float = 1.0,
              lr_penalty: str = 'l1',
              lr_solver: str = 'saga',
              lr_max_iteration: int = 200,
              svc_gamma: str = 'scale',
              svc_kernel: str = 'rbf'):
        '''Trains a classifier from training data.'''

        if classifier_type == 'logistic_regression':
            self.cls_model = sklearn.linear_model.LogisticRegression(
                C=regularization, penalty=lr_penalty, solver=lr_solver, max_iter=lr_max_iteration)
        elif classifier_type == 'svc':
            self.cls_model = sklearn.svm.SVC(
                C=regularization, gamma=svc_gamma, kernel=svc_kernel, probability=True)
        else:
            raise ValueError(f'Unknown classifier_type: {classifier_type}')

        x_train, y_train = extract_feature_and_label(train_set)

        self.cls_model.fit(x_train, y_train)

    @staticmethod
    def create_from_config(config_file: str):
        '''Create EmbeddingClassifier object from config file.'''
        config_dir = os.path.dirname(config_file)

        with open(config_file, 'r', encoding='utf-8') as f:
            json_data = f.read()
        cls_config = ClassifierConfig.model_validate_json(json_data)
        cls = EmbeddingClassifier()
        cls.load_models(cls_config, config_dir)
        return cls
