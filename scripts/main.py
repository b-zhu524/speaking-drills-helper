import train, process_dataset

if __name__ == "__main__":
    # get dataset
    dataset = process_dataset.create_dataset("data/raw/")

    # train model