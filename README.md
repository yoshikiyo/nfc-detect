# NFC Detect

NFC Detect is a tool to train audio classification models from labeled audio data.
It leverages publicly available audio neural network models trained using bird song recordings, such as
[Google Perch](https://github.com/google-research/perch/tree/main) or
[BirdNET](https://github.com/kahst/BirdNET-Analyzer).
Pre-trained models are used to compute embeddings from audio input, which is then used as input features of a classification model.

The primary goal of the project is to detect nocturnal flight calls of migratory birds.

## Installation

You can add nfc-detect to your dependency using `poetry add`.

```
poetry add git+https://github.com/yoshikiyo/nfc-detect.git
```

## Dataset preparation

Dataset consists of audio files and annotations associated with them.

* Filename
* Annotation
  * Label
  * Start time
  * Duration

CSV format is expected for annotation data. The following four fields should be present for each record. Other fields are ignored.

* `filename` (str): Path to an audio file (relative to a directory)
* `label` (str): Label of the segment
* `start` (float): Start time of the segment into the audio file specified by `filename` field in seconds
* `duration` (float): Duration of the segment in seconds

Below is an example.
```
filename,label,start,duration
01.wav,call,10.1,0.5
01.wav,noise,23.7,0.7
...
02.wav,xxx,xx,xx
02.wav,xxx,xx,xx
```

You then need to convert the annotations and audio file into dataset in TFRecord format.
The following command will extract annotated segements from the audio, applies the embedding model,
and stores the data into TFRecord file.

```
python3 -m nfcdetect prepare_data \
  --annotation_file <path to annotation CSV file> \
  --data_dir <path to directory that holds audio files> \
  --embedding_model <birdnet or perch> \
  --output <output filename>
```

For example:
```
python3 -m nfcdetect prepare_data \
  --annotation_file my_annotation.csv \
  --data_dir /path/to/audio/files \
  --embedding_model birdnet \
  --output my_data_train.tfrecords
```

## Training

You can train a model with `train` command as follows:
```
python3 -m nfcdetect train \
  --train_set <path to tfrecords file to be used as training data> \
  --output <filename to be used to save the trained model>
  --embedding_model <embedding model to use> \
  --classifier_type <svc or logistic_regression> \
  --regularization <regularization term (common to svc and logistic_regression)>
```

For example, the following command trains a SVC model using `my_data_train.tfrecords`. You should see two files generated: `/path/to/my_classifier.joblib` and `/path/to/my_classifier_config.json` after successful run.
```
python3 -m nfcdetect train \
  --train_set my_data_train.tfrecords \
  --output /path/to/my_classifier
  --embedding_model birdnet \
  --classifier_type svc \
  --regularization 1.0
```

## Inference

Trained models can be loaded and used for inference using `analyze` command:

```
python3 -m nfcdetect analyze \
  --classifier_config /path/to/my_classifier_config.json \
  --input /path/to/input/audio_file.flac \
  --start 0.0 \
  --duration 60.0
```

This applies the classifier to the first 60 seconds of `audio_file.flac` with 1 second intervals. The classification results is written to `nfc_output.csv` in the current directory. You can change to which file the results are written to by specifying `--output` option.
