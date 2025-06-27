import os
import torch
import torchaudio
from datasets import Dataset, load_dataset, Audio
from transformers import AutoFeatureExtractor
import numpy as np

"""
Create a dataset from audio files in a specified directory.
"""
def create_dataset(raw_data_dir) -> Dataset:
    data = {"file": [], "label": [], "audio": []}

    # add data
    for label in ["clear-raw", "unclear-raw"]:
        dir_path = os.path.join(raw_data_dir, label)
        for filename in os.listdir(dir_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(dir_path, filename)
                data["file"].append(file_path)
                data["audio"].append(file_path)
                data["label"].append(0 if label == "clear-raw" else 1) # 0 for clear, 1 for unclear

    # decode .wav files into arrays
    dataset = preprocess_data(dataset=Dataset.from_dict(data))

    dataset = dataset.shuffle(seed=42)
    dataset = split_dataset(dataset)

    return dataset 


"""
Resamples the audio files & normalizes them using a feature extractor.
"""
def preprocess_data(dataset):
    model_id = "facebook/wav2vec2-base"
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_id, do_normalize=True, return_attention_mask=True
    )

    sampling_rate = feature_extractor.sampling_rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))


    def preprocess_fn(batch):
        audio_arrays = [x["array"] for x in batch["audio"]]

        # apply feature extractor (returns input_values, attention_mask)
        inputs = feature_extractor(audio_arrays,
                                   sampling_rate=sampling_rate,
                                   max_length=int(feature_extractor.sampling_rate * 5),
                                   truncation=True,
                                   return_attention_mask=True
                                   )
        return inputs
    
    dataset_encoded = dataset.map(
        preprocess_fn,
        remove_columns=["audio", "file"],   # remove audio and file paths - only need encoded columns
        batched=True,
        batch_size=100,
        num_proc=1,  # Adjust based on your CPU cores
        )

    return dataset_encoded



def load_audio_files():
    dataset = create_dataset("data/raw/")
    dataset.save_to_disk("data/processed")


def split_dataset(dataset):
    dataset = dataset.train_test_split(seed=42, shuffle=True, test_size=0.1) 
    return dataset


def check_if_normalized(dataset):
    sample = dataset["train"][0]["input_values"]
    print(f"Mean: {np.mean(sample):.3}, Variance: {np.var(sample):.3}")
     


if __name__ == "__main__":
    dataset = create_dataset("data/raw/")
    check_if_normalized(dataset)