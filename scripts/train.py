from transformers import AutoFeatureExtractor, Trainer, TrainingArguments
import numpy as np
from datasets import Audio


def preprocess_data(dataset):
    model_id = "facebook/wav2vec2-base"
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_id, do_normalize=True, return_attention_mask=True
    )

    sampling_rate = feature_extractor.sampling_rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    return dataset




if __name__ == "__main__":
    preprocess_data()