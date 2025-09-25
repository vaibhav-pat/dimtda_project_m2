import os
import json
import torch
import jieba
import re
import argparse 
import evaluate
from evaluate import load
import numpy as np
from transformers import TrainerCallback
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

    # Handle edge cases where no valid tokens exist
    if len(labels_flat) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    # Calculate the metrics
    acc = accuracy.compute(predictions=predictions_flat, references=labels_flat)["accuracy"]
    prec = precision.compute(predictions=predictions_flat, references=labels_flat, average="weighted")["precision"]
    rec = recall.compute(predictions=predictions_flat, references=labels_flat, average="weighted")["recall"]
    f1_score = f1.compute(predictions=predictions_flat, references=labels_flat, average="weighted")["f1"]

    # Also calculate macro averages for better insight
    prec_macro = precision.compute(predictions=predictions_flat, references=labels_flat, average="macro")["precision"]
    rec_macro = recall.compute(predictions=predictions_flat, references=labels_flat, average="macro")["recall"]
    f1_macro = f1.compute(predictions=predictions_flat, references=labels_flat, average="macro")["f1"]

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1_score,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "valid_tokens": len(labels_flat),
    }

class MetricsCallback(TrainerCallback):
    """Custom callback to log metrics during training"""
    
    def __init__(self):
        self.best_metrics = {}
        self.final_metrics = {}
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Print metrics in a nice format
            if 'eval_accuracy' in logs:
                print(f"\n=== PRETRAINING EVALUATION METRICS ===")
                print(f"Accuracy:  {logs.get('eval_accuracy', 0):.4f}")
                print(f"Precision: {logs.get('eval_precision', 0):.4f}")
                print(f"Recall:    {logs.get('eval_recall', 0):.4f}")
                print(f"F1 Score:  {logs.get('eval_f1', 0):.4f}")
                print(f"Valid Tokens: {logs.get('eval_valid_tokens', 0)}")
                print("=" * 35)
                
                # Store the latest metrics as final metrics
                self.final_metrics = {
                    'accuracy': logs.get('eval_accuracy', 0),
                    'precision': logs.get('eval_precision', 0),
                    'recall': logs.get('eval_recall', 0),
                    'f1': logs.get('eval_f1', 0),
                    'precision_macro': logs.get('eval_precision_macro', 0),
                    'recall_macro': logs.get('eval_recall_macro', 0),
                    'f1_macro': logs.get('eval_f1_macro', 0),
                    'valid_tokens': logs.get('eval_valid_tokens', 0),
                }
                
                # Track best metrics based on F1 score
                current_f1 = logs.get('eval_f1', 0)
                if not self.best_metrics or current_f1 > self.best_metrics.get('f1', 0):
                    self.best_metrics = self.final_metrics.copy()
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called when training ends - print final summary"""
        print("\n" + "="*60)
        print("🎯 PRETRAINING COMPLETED - FINAL METRICS SUMMARY")
        print("="*60)
        
        if self.final_metrics:
            print("\n📊 FINAL EVALUATION METRICS:")
            print(f"   Accuracy:     {self.final_metrics['accuracy']:.4f}")
            print(f"   Precision:    {self.final_metrics['precision']:.4f}")
            print(f"   Recall:       {self.final_metrics['recall']:.4f}")
            print(f"   F1 Score:     {self.final_metrics['f1']:.4f}")
            print(f"   Valid Tokens: {self.final_metrics['valid_tokens']:,}")
            
            print("\n📈 MACRO AVERAGES:")
            print(f"   Precision (Macro): {self.final_metrics['precision_macro']:.4f}")
            print(f"   Recall (Macro):    {self.final_metrics['recall_macro']:.4f}")
            print(f"   F1 (Macro):        {self.final_metrics['f1_macro']:.4f}")
        
        if self.best_metrics:
            print("\n🏆 BEST METRICS (During Training):")
            print(f"   Best Accuracy:     {self.best_metrics['accuracy']:.4f}")
            print(f"   Best Precision:    {self.best_metrics['precision']:.4f}")
            print(f"   Best Recall:       {self.best_metrics['recall']:.4f}")
            print(f"   Best F1 Score:     {self.best_metrics['f1']:.4f}")
        
        print("\n✅ Training completed successfully!")
        print("="*60)
def train(args):

    MAX_LENGTH = args.max_length
    
    from transformers import AutoTokenizer
    en_tokenizer = AutoTokenizer.from_pretrained(args.en_tokenizer_dir)
    zh_tokenizer = AutoTokenizer.from_pretrained(args.zh_tokenizer_dir)

    with open(args.split_json_file_path, 'r') as f:
        json_dict = json.load(f)
    train_name_list = json_dict['train_name_list']
    valid_name_list = json_dict['valid_name_list']

    from my_dataset import DoTADatasetTrans
    valid_dataset = DoTADatasetTrans(en_tokenizer, zh_tokenizer, args.en_mmd_dir, args.zh_mmd_dir, valid_name_list, MAX_LENGTH)
    train_dataset = DoTADatasetTrans(en_tokenizer, zh_tokenizer, args.en_mmd_dir, args.zh_mmd_dir, train_name_list, MAX_LENGTH)

    from transformers import EncoderDecoderModel, EncoderDecoderConfig, BertConfig

    encoder_config = BertConfig()
    decoder_config = BertConfig()
    encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    
    # Modify config according to transformer-base
    encoder_decoder_config.encoder.bos_token_id = en_tokenizer.bos_token_id
    encoder_decoder_config.encoder.eos_token_id = en_tokenizer.eos_token_id
    encoder_decoder_config.encoder.hidden_size = 512
    encoder_decoder_config.encoder.intermediate_size = 2048
    encoder_decoder_config.encoder.max_length = MAX_LENGTH
    encoder_decoder_config.encoder.max_position_embeddings = MAX_LENGTH
    encoder_decoder_config.encoder.num_attention_heads = 8
    encoder_decoder_config.encoder.num_hidden_layers = 6
    encoder_decoder_config.encoder.pad_token_id = en_tokenizer.pad_token_id
    encoder_decoder_config.encoder.type_vocab_size = 1
    encoder_decoder_config.encoder.vocab_size = len(en_tokenizer)

    encoder_decoder_config.decoder.bos_token_id = zh_tokenizer.bos_token_id
    encoder_decoder_config.decoder.decoder_start_token_id = zh_tokenizer.bos_token_id
    encoder_decoder_config.decoder.eos_token_id = zh_tokenizer.eos_token_id
    encoder_decoder_config.decoder.hidden_size = 512
    encoder_decoder_config.decoder.intermediate_size = 2048
    encoder_decoder_config.decoder.max_length = MAX_LENGTH
    encoder_decoder_config.decoder.max_position_embeddings = MAX_LENGTH
    encoder_decoder_config.decoder.num_attention_heads = 8
    encoder_decoder_config.decoder.num_hidden_layers = 6
    encoder_decoder_config.decoder.pad_token_id = zh_tokenizer.pad_token_id
    encoder_decoder_config.decoder.type_vocab_size = 1
    encoder_decoder_config.decoder.vocab_size = len(zh_tokenizer)

    encoder_decoder_config.decoder_start_token_id = zh_tokenizer.bos_token_id
    encoder_decoder_config.pad_token_id = zh_tokenizer.pad_token_id
    encoder_decoder_config.eos_token_id = zh_tokenizer.eos_token_id
    encoder_decoder_config.max_length = MAX_LENGTH
    encoder_decoder_config.early_stopping = True
    encoder_decoder_config.no_repeat_ngram_size = 3
    encoder_decoder_config.length_penalty = 1.0
    encoder_decoder_config.num_beams = 4
    encoder_decoder_config.vocab_size = len(zh_tokenizer)

    model = EncoderDecoderModel(config=encoder_decoder_config)

    num_gpu = 1 if torch.backends.mps.is_available() or not torch.cuda.is_available() else torch.cuda.device_count()
    # gradient_accumulation_steps = args.batch_size // (num_gpu * args.batch_size_per_gpu)
    
    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size_per_gpu,
        per_device_eval_batch_size=args.batch_size_per_gpu,
        gradient_accumulation_steps=8, 
        dataloader_pin_memory=False,  
        logging_strategy='steps',
        logging_steps=10,  # Reduced from 1 to avoid too frequent logging
        do_train=True,
        evaluation_strategy='steps',
        eval_steps=50,  # More frequent evaluation for small dataset
        save_strategy='steps',
        save_steps=args.save_steps,
        fp16=args.fp16, 
        use_mps_device=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues
        load_best_model_at_end=True,   # Load best model at end
        metric_for_best_model="eval_f1",  # Use F1 score for best model selection
        greater_is_better=True,        # Higher F1 is better
    )

    # Create metrics callback
    metrics_callback = MetricsCallback()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback],
    )

    # Print dataset info for debugging
    print(f"\n=== DATASET INFO ===")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")
    print(f"Batch size: {args.batch_size_per_gpu}")
    print(f"Gradient accumulation steps: 8")
    print(f"Effective batch size: {args.batch_size_per_gpu * 8}")
    print(f"Steps per epoch: {len(train_dataset) // (args.batch_size_per_gpu * 8)}")
    print(f"Total steps: {len(train_dataset) // (args.batch_size_per_gpu * 8) * args.num_train_epochs}")
    print("=" * 20)

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(training_args)

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_tokenizer_dir", type=str)
    parser.add_argument("--zh_tokenizer_dir", type=str)
    parser.add_argument("--en_mmd_dir", type=str)
    parser.add_argument("--zh_mmd_dir", type=str)
    parser.add_argument("--split_json_file_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--learning_rate", type=float, default=7e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    
    args = parser.parse_args()
    
    train(args)