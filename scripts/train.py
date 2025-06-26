from datasets import load_dataset


def split_dataset(dataset):
    dataset = dataset.train_test_split(seed=42, shuffle=True, test_size=0.1) 
    print(dataset)
    return dataset



if __name__ == "__main__":
    pass