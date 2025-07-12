if __name__ == "__main__":
    import process_dataset
    import train

    # overfit a model on a small dataset
    model_id = "facebook/wav2vec2-base-960h"
    dataset = process_dataset.create_dataset("data/raw/", model_id=model_id)
    small_dataset = dataset["train"].shuffle(seed=42).select(range(10))  # select 10 samples for a sanity check

    training_args = train.get_training_args(model_id)
    feature_extractor = process_dataset.get_feature_extractor()
    