import train, process_dataset
from transformers import Trainer


if __name__ == "__main__":
    model_id = "facebook/wav2vec2-base-960h"

    # get dataset
    dataset = process_dataset.create_dataset("data/raw/", model_id=model_id)

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

    # sanity check with a small dataset
    small_train_dataset = train_dataset.shuffle(seed=42).select(range(10))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,  # small dataset for sanity check
        eval_dataset=small_train_dataset,
        compute_metrics=train.compute_metrics,
        data_collator=train.DataCollatorWithPadding(feature_extractor),
    )
    print("starting sanity check")
    trainer.train()
    print("sanity check done")


    # # real training
    # trainer = train.train_model(model, training_args, train_dataset, eval_dataset, feature_extractor)
    # print("Training complete. Model saved at:", training_args.output_dir)

    # # evaluate
    # train.evaluate_model(trainer)
    # print("Evaluation complete.")



