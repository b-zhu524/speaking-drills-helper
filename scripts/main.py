import train, process_dataset

if __name__ == "__main__":
    dataset = process_dataset.load_audio_files()
    dataset = process_dataset.split_dataset(dataset)

    new_ds = train.preprocess_data(dataset)
    print(new_ds["train"][0])
