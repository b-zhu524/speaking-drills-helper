import train, process_dataset

if __name__ == "__main__":
    model_id = "facebook/wav2vec2-base-960h"

    # get dataset
    dataset = process_dataset.create_dataset("data/raw/", model_id=model_id)

    # train model
    training_args = train.get_training_args(model_id)
    train_dataset, eval_dataset = process_dataset.split_dataset(dataset)
    feature_extractor = process_dataset.get_feature_extractor(model_id)
    id2label = train.get_id2label()
    label2id = train.get_label2id()

    model = train.get_model(model_id, len(dataset.features["label"].names), id2label, label2id)
    train.train_model(model, training_args, train_dataset, eval_dataset, feature_extractor)
    print("Training complete. Model saved at:", training_args.output_dir)

    # evaluate
    train.evaluate_model(model, eval_dataset)
    print("Evaluation complete.")



