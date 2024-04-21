# NFC Detect

NFC Detect is a tool to train audio classification models from labeled audio data.
It leverages publicly available audio neural network models trained using bird song recordings, such as
[Google Perch](https://github.com/google-research/perch/tree/main) or
[BirdNET](https://github.com/kahst/BirdNET-Analyzer).
Pre-trained models are used to compute embeddings from audio input, which is then used as input features of classification model.

The primary goal of the project is to detect nocturnal flight calls of migratory birds.

# Getting started

## Installation

You can add nfc-detect to your dependency using `poetry add`.

```
poetry add git+https://github.com/yoshikiyo/nfc-detect.git
```


## Inference

Trained models can be loaded and used for inference, as shown below:

```
import librosa

from nfcdetect import (
  config,
  classifier
)

CLASSFIER_CONFIG = 'sample_model.json'
INPUT = 'sample_audio.WAV'
START = 0.0      # Start position of audio segment to analyze
DURATION = 30.0  # Duration of audio segment to analyze

cls = classifier.EmbeddingClassifier.create_from_json(CLASSIFIER_CONFIG)

sr = cls.get_config().embedding_config.sampling_rate
audio, sr = librosa.load(INPUT, sr=sr, offset=START, duration=DURATION)

predictions = cls.predict(audio)
```

See `analyze.py` for a full example.
