from transformers import Wav2VecProcessor, Wav2VecModel
import torchaudio
import torch


def get_embeddings(file_path):
    # Load the pre-trained Wav2Vec model
    processor = Wav2VecProcessor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2VecModel.from_pretrained("facebook/wav2vec2-base")

    # Load an audio clips
    waveform, sample_rate = torchaudio.load(file_path)

    # Resample if necessary
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    
    # Get model input
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")

    # Extract features
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings.squeeze().numpy()


def classify_audio(embeddings):
    # Placeholder for classification logic
    # This could be a simple linear layer, a more complex model, etc.
    # For now, we will just return the embeddings
    return embeddings