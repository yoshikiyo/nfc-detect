'''A sample script that shows how to use nfc-detect package.
'''

import argparse
import csv

import librosa
import numpy as np

from nfcdetect import classifier

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, required=True,
                        help='Audio file to analyze.')
    parser.add_argument('--start', type=float, default=0.0,
                        help='Start position of audio segment to analyze in seconds.')
    parser.add_argument('--duration', type=float, default=300,
                        help='Duration of audio segment to analyze in seconds.')
    parser.add_argument('--classifier_config', type=str, required=True,
                        help='Classifier config file')
    parser.add_argument('--output', type=str, default='nfc_output.csv',
                        help='Path to output of NFC classification results in CSV format.')
    args = parser.parse_args()

    cls = classifier.EmbeddingClassifier.create_from_config(args.classifier_config)

    sr = cls.cfg.embedding_config.sampling_rate
    audio, sr = librosa.load(args.input,
                             sr=sr,
                             offset=args.start,
                             duration=args.duration)
    
    # Truncate array as librosa can sometimes load more samples than requested.
    audio = audio[:int(np.ceil(args.duration * sr))]

    predictions, classes = cls.predict(audio)

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['position']
        header.extend(classes)
        writer.writerow(header)
        for idx, prediction in enumerate(predictions):
            pos = args.start + idx
            row = [f'{pos:.3f}']
            row.extend([f'{p:.6f}' for p in prediction])
            writer.writerow(row)

if __name__ == '__main__':
    main()
