# Training data

Training data consists of audio files and annotations associated with them.

* Filename
* Annotation
  * Label
  * Start time
  * Duration

## CSV format

CSV format is expected for annotation data. The following four fields should be present for each record. Other fields are ignored.

* `filename`: Path to an audio file (string)
* `label`: Label of the segment (string)
* `start`: Start time of the segment into the audio file specified by `filename` field in seconds (float)
* `duration`: Duration of the segment in seconds (float)

### Example
```
filename,label,start,duration
01.wav,zeep,10.1,0.5
01.wav,noise,23.7,0.7
...
02.wav,xxx,xx,xx
02.wav,xxx,xx,xx
```
