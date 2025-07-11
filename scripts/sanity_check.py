if __name__ == "__main__":
    import process_dataset
    import train

    # Load the dataset
    dataset = process_dataset.create_dataset("data/raw/")

    # Check if the dataset is normalized
    process_dataset.check_if_normalized(dataset)

    # Get the feature extractor
    feature_extractor = process_dataset.get_feature_extractor("facebook/wav2vec2-base-960h")

    # Get label mappings
    id2label = train.get_id2label()
    label2id = train.get_label2id()

    # Print dataset info
    print(f"Dataset keys: {dataset.column_names}")
    print(f"Feature extractor: {feature_extractor}")
    print(f"Label mappings: {id2label}, {label2id}")

    # get dataset sample
    sample = dataset["train"][0]
    print(f"Sample input values: {sample['input_values'][:10]}...")
