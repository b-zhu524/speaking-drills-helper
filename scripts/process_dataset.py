import os
import torch
import torchaudio
from datasets import Dataset, load_dataset, Audio

"""
Create a dataset from audio files in a specified directory.
"""
def create_dataset(raw_data_dir) -> Dataset:
    data = {"audio_path": [], "label": []}

    # add data
    for label in ["clear-raw", "unclear-raw"]:
        dir_path = os.path.join(raw_data_dir, label)
        for filename in os.listdir(dir_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(dir_path, filename)
                data["audio_path"].append(file_path)
                data["label"].append(0 if label == "clear-raw" else 1) # 0 for clear, 1 for unclear

    return Dataset.from_dict(data)


def load_audio_files():
    dataset = create_dataset("data")
    dataset = dataset.shuffle(seed=42)
    dataset.save_to_disk("data/processed")


def split_dataset(dataset):
    dataset = dataset.train_test_split(seed=42, shuffle=True, test_size=0.1) 
    print(dataset)
    return dataset


if __name__ == "__main__":
    load_audio_files()
    print("Dataset created and saved to 'data/processed'")
   
