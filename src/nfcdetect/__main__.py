'''A sample script that shows how to use nfc-detect package.
'''

import argparse
import csv
import os

# Suppress info/warning messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import librosa
import numpy as np
import tensorflow as tf

from nfcdetect import classifier, config, dataset


def analyze(classifier_config: config.ClassifierConfig,
            input_file: str,
            start: float,
            duration: float,
            output_file: str):
    '''Apply classifier to audio file and output prediction as CSV file.'''

    cls = classifier.EmbeddingClassifier.create_from_config(classifier_config)

    sr = cls.cfg.embedding_config.sampling_rate
    audio, sr = librosa.load(input_file, sr=sr, offset=start, duration=duration)

    # Truncate array as librosa can sometimes load more samples than requested.
    audio = audio[:int(np.ceil(duration * sr))]

    predictions, classes = cls.predict(audio)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['position']
        header.extend(classes)
        writer.writerow(header)
        for idx, prediction in enumerate(predictions):
            pos = start + idx
            row = [f'{pos:.3f}']
            row.extend([f'{p:.6f}' for p in prediction])
            writer.writerow(row)


def train(train_set: str,
          model_output: str,
          embedding_model: str = 'birdnet',
          classifier_type: str = 'svc',
          regularization: float = 1.0,
          test_set: str = None
          ):
    cls_model = classifier.EmbeddingClassifier()
    cfg = config.ClassifierConfig(
        embedding_config=config.EMBEDDING_CONFIG[embedding_model])
    cls_model.set_config(cfg)

    examples = dataset.read_dataset(train_set)
    cls_model.train(examples, classifier_type, regularization=regularization)

    cls_model.save(model_output)

    if test_set:
        examples_test = dataset.read_dataset(test_set)
        metrics = {}
        X, y = classifier.extract_feature_and_label(examples_test)
        metrics['score'] = cls_model.cls_model.score(X, y)
        _, _, _, pr_auc = classifier.compute_pr_auc(cls_model.cls_model, X, y)
        metrics['pr_auc'] = pr_auc
        print(metrics)


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    parser = argparse.ArgumentParser(prog='python3 -m nfcdetect')

    parser.add_argument('command', type=str, choices=['analyze', 'prepare_data', 'train'],
                        help='Command')

    # Common arguments

    # Arguments for 'analyze'
    parser.add_argument('--input', type=str,
                        help='Audio file to analyze.')
    parser.add_argument('--start', type=float, default=0.0,
                        help='Start position of audio segment to analyze in seconds.')
    parser.add_argument('--duration', type=float, default=300,
                        help='Duration of audio segment to analyze in seconds.')
    parser.add_argument('--classifier_config', type=str,
                        help='Classifier config file')
    parser.add_argument('--output', type=str, default='nfc_output.csv',
                        help='Path to output of NFC classification results in CSV format.')

    # Arguments for 'prepare_data'
    parser.add_argument('--annotation_file', type=str)
    parser.add_argument('--data_dir', type=str,
                        help='Path to directory that contains audio files referenced in '
                        '`annotation_file\'')
    parser.add_argument('--embedding_model', type=str,
                        help='Type of embedding model. Accepts "perch" or "birdnet"')
    parser.add_argument('--padding_method', )
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size to use for inference.')
    parser.add_argument('--jitter', action='store_true', default=False,
                        help='Apply jitter when cropping frame from annotated audio.')
    parser.add_argument('--augmentation_per_sample', type=int, default=0,
                        help='Additional examples sampled per example in annotation file. '
                             'Using --augmentation_per_sample=N means there will be N+1 examples '
                             'from the same annotation. Best used with --jitter option.')
    parser.add_argument('--negative_samples', type=int, default=0,
                        help='Number of negative examples to be sampled from audio to be included '
                             'in the dataset. '
                             'When a value greater than zero is given, examples are randomly '
                             'sampled from audio files referenced in the annotation file. '
                             'Note that no check is performed whether the sample overlaps '
                             'with positive examples, thus these can be noisy.')
    parser.add_argument('--negative_label', type=str, default='neg',
                        help='Label to use for negative examples sampled when using '
                             '--negative-samples option.')
    parser.add_argument('--negative_duration', type=float, default=0.2,
                        help='Duration of negative examples. No need to change this in general.')

    # Arguments for 'train'
    parser.add_argument('--classifier_type', type=str, default='svc',
                        choices=['logistic_regression', 'svc'])
    parser.add_argument('--regularization', type=float, default=1.0)
    parser.add_argument('--train_set', type=str,
                        help='Path to dataset to train the model. Expects TFRecord file.')
    parser.add_argument('--test_set', type=str, default=None,
                        help='Path to test dataset to evaluate the model. Expects TFRecord file.')

    args = parser.parse_args()

    if args.command == 'analyze':
        analyze(args.classifier_config, args.input, args.start, args.duration, args.output)

    elif args.command == 'prepare_data':
        options = dataset.PreprocessOptions(
            cfg=config.ClassifierConfig(
                embedding_config=config.EMBEDDING_CONFIG[args.embedding_model]),
            jitter=args.jitter,
            augmentation_per_sample=args.augmentation_per_sample,
            negative_samples=args.negative_samples,
            negative_label=args.negative_label,
            negative_duration=args.negative_duration
        )
        dataset.prepare_dataset(args.annotation_file,
                                args.data_dir,
                                args.output,
                                options=options,
                                batch_size=args.batch_size)

    elif args.command == 'train':
        train(args.train_set,
              args.output,
              args.embedding_model,
              args.classifier_type,
              args.regularization,
              test_set=args.test_set)

    else:
        print(f'Unknown command: {args.command}')
        parser.print_help()


if __name__ == '__main__':
    main()
