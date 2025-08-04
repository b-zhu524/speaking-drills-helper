import torch
import os
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForAudioClassification
import torchaudio
from transformers import pipeline, AutoFeatureExtractor
import data_collection.segment_large_audio_files as segmenter
import process_dataset, train


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



def evaluate_model(classifier, audio_path_base):
    """
    goes through all the files in audio path base and evaluates them — prints {name}: result
    """
    segmenter.load_data_for_evaluation()

    for filename in os.listdir(audio_path_base):
        if filename.endswith(".wav"):
            audio_path = os.path.join(audio_path_base, filename)
            result = classifier(audio_path)
            print(f"{filename}: {result}")


def save_model(model, processor, checkpoint_path):
    """
    Saves the model and processor to the specified path.
    """
    os.makedirs(checkpoint_path, exist_ok=True)
    model.save_pretrained(checkpoint_path)
    processor.save_pretrained(checkpoint_path)
    print(f"Model and processor saved to {checkpoint_path}")


if __name__ == "__main__":
    checkpoint_path = "./models/wav2vec2-base-960h-new-f1/checkpoint-686"
    final_model_path = "./models/wav2vec2-base-960h-f1-final"
    save_path = "./models/wav2vec2-base-960h-f1-savedfinal"

    # save the model
    model = AutoModelForAudioClassification.from_pretrained(checkpoint_path)
    # processor = Wav2Vec2Processor.from_pretrained(checkpoint_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(checkpoint_path)

    # model.save_pretrained(save_path, from_pt=True)
    # processor.save_pretrained(save_path)

    # print("done saving")


    # model = AutoModelForAudioClassification.from_pretrained(save_path)
    # processor = Wav2Vec2Processor.from_pretrained(save_path)

    # feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

    classifier = pipeline("audio-classification", model=model, feature_extractor=feature_extractor)

    evaluate_model(classifier, "data/raw/testing")



