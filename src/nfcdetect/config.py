from pydantic import BaseModel

class EmbeddingConfig(BaseModel):
    name: str
    sampling_rate: int
    frame_size: float  # in seconds

EMBEDDING_CONFIG = {
    'birdnet': EmbeddingConfig(name='birdnet', sampling_rate=48000, frame_size=3.0),
    'perch': EmbeddingConfig(name='perch', sampling_rate=32000, frame_size=5.0)
}

class ClassifierConfig(BaseModel):
    embedding_config: EmbeddingConfig
    crop_size: float = 1.0  # In seconds
    padding_method: str = 'repeat'
    padding_offset: float = 0
    classifier_model_path: str = None
