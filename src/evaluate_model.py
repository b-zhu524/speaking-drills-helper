import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from transformers import AutoModel, AutoTokenizer
import torchaudio


def load_model_and_processor(model_path, processor_path=None):
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)

    # Load processor from saved path or from base model if not present
    if processor_path:
        processor = Wav2Vec2Processor.from_pretrained(processor_path)
    else:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")  # Adjust to match base model

    model.eval()
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
        print(f"Predicted label: {predicted_label}")


def save_model(model, processor, model_path):
    model.save_pretrained(model_path, safe_serialization=False)
    processor.save_pretrained(model_path)


if __name__ == "__main__":
    checkpoint_path = "./models/wav2vec2-base-960h/checkpoint-700"
    final_model_path = "./models/wav2vec2-base-960h-final"
    audio_path = "./data/test/example.wav"

    # instantiate model and processor
    model = AutoModel.from_pretrained("wav2vec2-base-960h")
    processor = AutoTokenizer.from_pretrained("wav2vec2-base-960h")

    # save model and processor
    model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)

    loaded_model = AutoModel.from_pretrained(final_model_path)
    loaded_processor = AutoTokenizer.from_pretrained(final_model_path)

    inputs = load_audio(audio_path, loaded_processor)
    forward_pass(loaded_model, inputs)



