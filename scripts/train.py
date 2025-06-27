from transformers import AutofeatureExtractor, Trainer, TrainingArguments


def preprocess_data(dataset):
    model_id = "facebook/wav2vec2-base"
    feature_extractor = AutofeatureExtractor.from_pretrained(
        model_id, do_normalize=True, return_attention_mask=True
    )

    