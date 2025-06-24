from transformers import Wav2VecProcessor, Wav2VecModel
import torchaudio
import torch
import os
import numpy as np


def get_embeddings(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained Wav2Vec model
    processor = Wav2VecProcessor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2VecModel.from_pretrained("facebook/wav2vec2-base").to(device)
    model.eval()


    # Load an audio clip
    waveform, sample_rate = torchaudio.load(file_path)

    # Resample if necessary
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    # Get model input
    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(device)

    # Extract features
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.squeeze().cpu().numpy()


def extract_embeddings_from_directory(directory, label):
    features = []
    labels = []

    for fname in os.listdir(directory):
        if not fname.endswith('.wav'):
            continue
        file_path = os.path.join(directory, fname)
        embeddings = get_embeddings(file_path)
        features.append(embeddings)
        labels.append(label)
    
    return features, labels


if __name__ == "__main__":
    pass
