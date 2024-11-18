import torch
import numpy as np
import argparse
import evaluate
from transformers import TrainingArguments, Trainer

from dataset import build_datasets_models

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)   

def collate_fn(examples):
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def main(args):

    model_ckpt = args.model_name
    train_dataset, val_dataset, model, img_processor = build_datasets_models(args.model_name)

    print(f"Training clips: {train_dataset.num_videos} \
            Validation clips: {val_dataset.num_videos}")
    
    model_name = model_ckpt.split("/")[-1]
    new_model_name = f"{model_name}-readminds-assignment"

    num_epochs = args.n_epochs

    args = TrainingArguments(
        output_dir=new_model_name,
        remove_unused_columns=False,
        eval_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        prediction_loss_only=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_steps=(train_dataset.num_videos // args.batch_size) * num_epochs,
        push_to_hub=True,
        eval_accumulation_steps=10
    )

    trainer = Trainer( 
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=img_processor,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )   

    train_results = trainer.train()
    trainer.push_to_hub()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="MCG-NJU/videomae-base", help='pretrained model')
    parser.add_argument("--batch_size", "--b", default=2)
    parser.add_argument("--n_epochs", default=10)
    parser.add_argument("--learning_rate", "--lr", default=5e-5)
    args = parser.parse_args()

    main(args)


