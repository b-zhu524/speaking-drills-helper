from transformers import Trainer, TrainingArguments, DataCollatorWithPadding 
import numpy as np
import evaluate
from datasets import Audio
from transformers import AutoModelForAudioClassification, TrainerCallback
import torch

def get_label2id():
    return {
        "clear": 0,
        "unclear": 1
    }


def get_id2label():
    return {
        0: "clear",
        1: "unclear"
    }


def get_model(model_id, num_labels, label2id, id2label):
    model = AutoModelForAudioClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    return model



def get_training_args(model_id):
    model_name = model_id.split("/")[-1]
    batch_size = 4  # adjust based on your GPU memory 
    gradient_accumulation_steps = 4
    num_train_epochs = 50

    training_args = TrainingArguments(
        output_dir=f"./models/{model_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.1,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        #fp16=True,
        fp16=False, # disabled for testing
        # disable push to hub for now
        push_to_hub=False,
    )

    return training_args


'''
computes accuray on a batch of predictions 
'''
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

class DebuggerCallback(TrainerCallback):
    """Debug Util"""
    def on_step_begin(self, args, state, control, **kwargs):
        optimizer = kwargs.get("optimizer")
        if optimizer:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Step {state.global_step} - Learning rate: {lr}")


## class DebugTrainer(Trainer):
##     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
##         # Inspect the inputs and logits
##         outputs = model(**inputs)
##         logits = outputs.logits
## 
##         if torch.isnan(logits).any() or torch.isinf(logits).any():
##             print("NaN or Inf in logit!")
##             print(logits)
## 
##         loss = super().compute_loss(model, inputs, return_outputs)
##         return (loss, outputs) if return_outputs else loss



def train_model(model, training_args, train_dataset, eval_dataset, feature_extractor):
    torch.autograd.set_detect_anomaly(True)
    data_collator = DataCollatorWithPadding(feature_extractor)

    sample = train_dataset[0]
    print("input_values type:", type(sample["input_values"]))
    print("input_values shape:", np.shape(sample["input_values"]))
    print("min/max:", np.min(sample["input_values"]), np.max(sample["input_values"]))


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,    # training dataset
        eval_dataset=eval_dataset,  # test dataset 
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[DebuggerCallback()],
    )

    # Start training
    print("Starting training...")
    trainer.train()

    return trainer


def evaluate_model(trainer):
    metrics = trainer.evaluate()
    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return metrics


if __name__ == "__main__":
    pass
