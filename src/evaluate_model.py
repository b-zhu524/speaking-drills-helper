import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torchaudio
import os


def load_model(model_path):
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode

    return model, processor


def load_audio(audio_path, processor):
    waveform, sr = torchaudio.load(audio_path)
    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    return inputs


def forward_pass(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax(dim=-1).item()
        predicted_label = model.config.id2label[predicted_class_id]
        print(predicted_label)


def save_model(model, model_path):
    model.save_pretrained(model_path)
    processor.save_pretrained(model_path)


if __name__ == "__main__":
    model_path = "./models/wav2vec2-base-960h"
    checkpoint_path = "./models/wav2vec2-base-960h/checkpoint-700"

    # model, processor = load_model(model_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(model_path)

    save_model(model, model_path)

    audio_path = "./data/test/example.wav"
    inputs = load_audio(audio_path, processor)

    forward_pass(model, inputs)
    print("Manual model evaluation complete.")
