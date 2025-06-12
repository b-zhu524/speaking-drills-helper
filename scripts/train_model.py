from transformers import Wav2VecProcessor, Wav2VecModel
import torchaudio
import torch


# Load the pre-trained Wav2Vec model
processor = Wav2VecProcessor.from_pretrained("facebook/wav2vec2-base")
model = Wav2VecModel.from_pretrained("facebook/wav2vec2-base")