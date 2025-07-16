import train, process_dataset
from transformers import Trainer
from datasets import load_from_disk
import numpy as np

model_id = "facebook/wav2vec2-base-960h"
raw_data_dir = "data/raw/"
dataset_dir = "data/dataset1"

def create_and_save_dataset():
    dataset = process_dataset.create_dataset(raw_data_dir, model_id=model_id)
    dataset.save_to_disk(dataset_dir)


    
def train_model():
    # get dataset
    #dataset = process_dataset.create_dataset("data/raw/", model_id=model_id)
    dataset = load_from_disk(dataset_dir)

    # train model
    training_args = train.get_training_args(model_id)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    feature_extractor = process_dataset.get_feature_extractor(model_id)
    id2label = train.get_id2label()
    label2id = train.get_label2id()

    num_labels = len(set(dataset["train"]["label"]))
    model = train.get_model(model_id, num_labels, label2id, id2label)

    # train
    trainer = train.train_model(model, training_args, train_dataset, eval_dataset, feature_extractor)
    print("Training complete. Model saved at:", training_args.output_dir)

    # evaluate
    train.evaluate_model(trainer)
    print("Evaluation complete.")


if __name__ == "__main__":
    #create_and_save_dataset()
    train_model()
