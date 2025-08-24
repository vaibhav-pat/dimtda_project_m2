import os
import json
import torch
import jieba
import re
import argparse 
import evaluate
from evaluate import load
import numpy as np

# Load the standard metrics
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # The model outputs logits, we need to get the predicted class index
    predictions = np.argmax(logits, axis=-1)

    # The trainer pads labels with -100, we must ignore them for metric calculation
    # We flatten the arrays to compare token by token
    labels_flat = labels.flatten()
    predictions_flat = predictions.flatten()
    
    mask = labels_flat != -100
    labels_flat = labels_flat[mask]
    predictions_flat = predictions_flat[mask]

    # Calculate the metrics
    acc = accuracy.compute(predictions=predictions_flat, references=labels_flat)["accuracy"]
    prec = precision.compute(predictions=predictions_flat, references=labels_flat, average="weighted")["precision"]
    rec = recall.compute(predictions=predictions_flat, references=labels_flat, average="weighted")["recall"]
    f1_score = f1.compute(predictions=predictions_flat, references=labels_flat, average="weighted")["f1"]

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1_score,
    }
    
def train(args):
    MAX_LENGTH = args.max_length
    
    from transformers import AutoTokenizer, DonutProcessor, BeitImageProcessor
    dit_processor = BeitImageProcessor.from_pretrained(args.dit_model_dir)
    nougat_processor = DonutProcessor.from_pretrained(args.image_processor_dir)
    zh_tokenizer = AutoTokenizer.from_pretrained(args.zh_tokenizer_dir)

    with open(args.split_json_file_path, 'r') as f:
        json_dict = json.load(f)
    train_name_list = json_dict['train_name_list']
    valid_name_list = json_dict['valid_name_list']

    from my_dataset import DoTADataset
    valid_dataset = DoTADataset(dit_processor, nougat_processor, zh_tokenizer, args.image_dir, args.zh_mmd_dir, valid_name_list, MAX_LENGTH)
    train_dataset = DoTADataset(dit_processor, nougat_processor, zh_tokenizer, args.image_dir, args.zh_mmd_dir, train_name_list, MAX_LENGTH)

    from transformers import EncoderDecoderModel, VisionEncoderDecoderModel, BeitModel, EncoderDecoderConfig
    # trans_model = EncoderDecoderModel.from_pretrained(args.trans_model_dir)
    # this entire start to end of modification is a replacement for this single line trans_model
    # --- START OF MODIFICATION ---
    from transformers import EncoderDecoderConfig, EncoderDecoderModel

    # Load the original config, but then modify it
    config = EncoderDecoderConfig.from_pretrained(args.trans_model_dir)
    # Reduce the number of decoder layers from 6 to 3
    config.decoder.num_hidden_layers = 3

    # Initialize the model from the modified config, but load the weights from the checkpoint
    # The `ignore_mismatched_sizes=True` is crucial because the number of layers has changed.
    trans_model = EncoderDecoderModel.from_pretrained(args.trans_model_dir, config=config, ignore_mismatched_sizes=True)
    # --- END OF MODIFICATION ---
    dit_model = BeitModel.from_pretrained(args.dit_model_dir)
    nougat_model = VisionEncoderDecoderModel.from_pretrained(args.nougat_model_dir)
    
    from my_model import DIMTDAModel
    my_config = EncoderDecoderConfig.from_pretrained(args.trans_model_dir)
    model = DIMTDAModel(my_config, trans_model, dit_model, nougat_model, args.num_queries, args.qformer_config_dir)

    num_gpu = 1 if torch.backends.mps.is_available() or not torch.cuda.is_available() else torch.cuda.device_count()
    gradient_accumulation_steps = args.batch_size // (num_gpu * args.batch_size_per_gpu)
    
    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size_per_gpu,
        per_device_eval_batch_size=args.batch_size_per_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        logging_strategy='steps',
        logging_steps=1,
        evaluation_strategy='epoch',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        fp16=False,
        use_mps_device=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(training_args)

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trans_model_dir", type=str)
    parser.add_argument("--dit_model_dir", type=str)
    parser.add_argument("--nougat_model_dir", type=str)
    parser.add_argument("--image_processor_dir", type=str)
    parser.add_argument("--zh_tokenizer_dir", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--zh_mmd_dir", type=str)
    parser.add_argument("--split_json_file_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--qformer_config_dir", type=str)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--num_queries", type=int, default=256)
    
    args = parser.parse_args()
    
    train(args)