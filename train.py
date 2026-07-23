#!/usr/bin/env python3
"""
Training Script for Punjabi-English Neural Machine Translation (NMT)
Fine-tunes Helsinki-NLP MarianMT models using Hugging Face Transformers and Trainer.
"""

import os
import sys
import subprocess

# Auto-install dependencies if they are missing (highly useful for Kaggle/Colab runtimes)
REQUIRED_PACKAGES = {
    "yaml": "pyyaml",
    "evaluate": "evaluate",
    "sacrebleu": "sacrebleu",
    "datasets": "datasets",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "sentencepiece": "sentencepiece"
}

for module_name, pip_name in REQUIRED_PACKAGES.items():
    try:
        __import__(module_name)
    except ImportError:
        print(f"Installing missing dependency: {pip_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
            print(f"Successfully installed {pip_name}!")
        except Exception as e:
            print(f"Warning: Failed to install {pip_name} automatically: {e}")

import yaml
import argparse
import zipfile
import torch
import pandas as pd
import numpy as np
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MarianMT for Punjabi-English NMT")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to the config.yaml file"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Run a quick execution test with a tiny dataset subset and 1 epoch"
    )
    args, _ = parser.parse_known_args()
    return args

DEFAULT_CONFIG = {
    "data": {
        "output_csv": "punjabi_english_100k.csv"
    },
    "model": {
        "base_model": "Helsinki-NLP/opus-mt-pa-en",
        "max_length": 128
    },
    "training": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "fp16": True,
        "save_total_limit": 2,
        "logging_steps": 100
    }
}

def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file '{config_path}' not found! Using default configuration.")
        return DEFAULT_CONFIG
    with open(config_path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(f"Error parsing configuration YAML: {exc}")
            sys.exit(1)

def find_dataset(configured_path):
    possible_paths = [
        configured_path,
        "hindi_english_100k.csv",
        "punjabi_english_100k.csv",
        "data/processed/hindi_english_100k.csv",
        "data/processed/punjabi_english_100k.csv",
        "../punjabi_english_100k.csv",
        "../data/processed/hindi_english_100k.csv",
        "/content/hindi_english_100k.csv",
        "/content/punjabi_english_100k.csv",
        "/content/OmniBind/data/processed/hindi_english_100k.csv",
        "/content/OmniBind/data/processed/punjabi_english_100k.csv",
        "/kaggle/working/hindi_english_100k.csv",
        "/kaggle/working/punjabi_english_100k.csv",
        "/kaggle/working/data/processed/hindi_english_100k.csv",
        "/kaggle/working/data/processed/punjabi_english_100k.csv",
        "/kaggle/input/punjabi-english-100k/punjabi_english_100k.csv",
        "/kaggle/input/datasets/wizardb2k/punjabi-english-100k/punjabi_english_100k.csv"
    ]
    for path in possible_paths:
        if path and os.path.exists(path):
            return path
    return None

def main():
    args = parse_args()
    config = load_config(args.config)
    
    print("=" * 60)
    print("  OmniBind: Punjabi-English NMT Training Pipeline")
    print("=" * 60)
    
    # 1. Locate Dataset
    csv_path = find_dataset(config['data']['output_csv'])
    if not csv_path:
        print("\n" + "!" * 70)
        print("Error: Could not locate 'punjabi_english_100k.csv'!")
        print("Please upload the dataset file:")
        print("  - If running in Colab: click the Folder icon on the left sidebar,")
        print("    and click the 'Upload to session storage' button to upload the CSV.")
        print("  - If running in Kaggle: click '+ Add Data' on the right sidebar,")
        print("    upload 'punjabi_english_100k.csv' as a dataset, or upload to /kaggle/working/.")
        print("!" * 70 + "\n")
        sys.exit(1)
        
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Detect the target language column (which is not 'english' or 'split')
    columns = list(df.columns)
    if 'split' in columns:
        columns.remove('split')
    if 'english' in columns:
        columns.remove('english')
    target_col = columns[0] if columns else 'punjabi'
    print(f"Detected target language column: '{target_col}'")
    
    # Extract splits
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    if args.dry_run:
        print("--> DRY RUN MODE ENABLED: Subsetting dataset to 50 samples each...")
        train_df = train_df.head(50)
        val_df = val_df.head(10)
        test_df = test_df.head(10)
        
    raw_datasets = DatasetDict({
        'train': Dataset.from_pandas(train_df[['english', target_col]]),
        'validation': Dataset.from_pandas(val_df[['english', target_col]]),
        'test': Dataset.from_pandas(test_df[['english', target_col]])
    })
    
    print(f"Train samples: {len(raw_datasets['train']):,}")
    print(f"Val samples:   {len(raw_datasets['validation']):,}")
    print(f"Test samples:  {len(raw_datasets['test']):,}")
    
    # 2. Load Model & Tokenizer
    base_model_name = config['model']['base_model']
    print(f"Loading tokenizer and model: {base_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    print("Model loaded successfully!")
    
    # Determine maximum sequence length
    max_length = config['model']['max_length']
    
    # 3. Preprocess / Tokenize Dataset
    # We dynamically check if it is target-to-english or english-to-target
    # E.g. 'pa-en', 'hi-en', 'mul-en' is target-to-english
    is_target_to_en = "-en" in base_model_name.lower() or "mul-en" in base_model_name.lower()
    
    def preprocess_function(examples):
        if is_target_to_en:
            inputs = [ex for ex in examples[target_col]]
            targets = [ex for ex in examples['english']]
        else:
            inputs = [ex for ex in examples['english']]
            targets = [ex for ex in examples[target_col]]
            
        model_inputs = tokenizer(
            inputs, 
            text_target=targets, 
            max_length=max_length, 
            truncation=True
        )
        return model_inputs

    print("Tokenizing datasets...")
    tokenized_datasets = raw_datasets.map(
        preprocess_function, 
        batched=True, 
        remove_columns=raw_datasets["train"].column_names
    )
    print("Tokenization complete!")
    
    # 4. Metrics Setup
    metric_bleu = evaluate.load("sacrebleu")
    metric_chrf = evaluate.load("chrf")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 in labels as we cannot decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        
        bleu_result = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
        chrf_result = metric_chrf.compute(predictions=decoded_preds, references=[l[0] for l in decoded_labels])
        
        return {
            "bleu": round(bleu_result["score"], 2),
            "chrf": round(chrf_result["score"], 2)
        }
        
    # 5. Training Arguments
    t_config = config['training']
    output_dir = "./punjabi_english_marian_results"
    
    epochs = 1 if args.dry_run else t_config.get('num_train_epochs', 3)
    batch_size = 8 if args.dry_run else t_config.get('per_device_train_batch_size', 16)
    
    # Ensure memory safety
    use_fp16 = torch.cuda.is_available() and t_config.get('fp16', True)
    
    print(f"Configuring training arguments (FP16={use_fp16}, Batch Size={batch_size}, Epochs={epochs})...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(t_config.get('learning_rate', 3e-5)),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=t_config.get('weight_decay', 0.01),
        save_total_limit=t_config.get('save_total_limit', 2),
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=use_fp16,
        logging_steps=t_config.get('logging_steps', 100),
        report_to="none"
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # 6. Execute Fine-Tuning
    print("Starting model fine-tuning...")
    trainer.train()
    print("Fine-tuning complete!")
    
    # 7. Final Evaluation on Test Set
    print("Evaluating model on test dataset...")
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"], metric_key_prefix="test")
    print("\n" + "=" * 30 + "\n--- Test Results ---\n" + "=" * 30)
    for key, val in test_results.items():
        print(f"{key}: {val}")
        
    # 8. Save and Zip Model
    save_dir = "./punjabi_english_marian_final"
    print(f"Saving final model and tokenizer to '{save_dir}'...")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    zip_path = "punjabi_english_marian_final.zip"
    print(f"Archiving saved weights to '{zip_path}'...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, save_dir)
                zipf.write(filepath, arcname)
                
    print("=" * 60)
    print(f"SUCCESS! Fine-tuned weights archived to '{zip_path}'")
    print("=" * 60)

if __name__ == "__main__":
    main()
