import librosa
import numpy as np
import os
import pandas as pd


def extract_audio_features(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=1)



def create_features_and_labels():
    features = []
    labels = []


    for label_dir in os.listdir("data/raw"):
        for file in os.listdr(f"data/raw/{label_dir}"):
            if file.endswith(".wav"):
                file_path = os.path.join("data/raw", label_dir, file)
                feature_vector = extract_audio_features(file_path)
                features.append(feature_vector)
                labels.append(label_dir)
    return np.array(features), np.array(labels)


def make_csv(features, labels):
    df = pd.DataFrame(features)
    df["label"] = labels
    df.to_csv("data/processed/audio_features.csv", index=False)
    print("Feature extraction complete. CSV saved to data/processed/audio_features.csv")
    return df


if __name__ == "__main__":
    make_csv(*create_features_and_labels())
