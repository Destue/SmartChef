# fine_tuning/scripts/train.py
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
import logging
from typing import Dict, List

class RecipeDataset(Dataset):
    """Custom dataset for recipe fine-tuning"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        # Load the data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        full_text = f"{item['input']}\n{item['output']}"
        
        # Tokenize the text
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze()
        }

def train_model():
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set up padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set padding token to EOS token")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True
    )
    
    # Resize model embeddings to account for padding token
    model.resize_token_embeddings(len(tokenizer))
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = RecipeDataset("fine_tuning/data/train.json", tokenizer)
    val_dataset = RecipeDataset("fine_tuning/data/val.json", tokenizer)
    
    logger.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    
    # Training arguments with fixed strategies
    training_args = TrainingArguments(
        output_dir="fine_tuning/models/recipe-bot",
        num_train_epochs=3,                
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=10,                   
        weight_decay=0.01,
        logging_dir="fine_tuning/logs",
        logging_steps=50,                  
        evaluation_strategy="steps",       
        eval_steps=200,                    
        save_strategy="steps",             
        save_steps=200,                    
        save_total_limit=1,               
        load_best_model_at_end=True,
        use_cpu=True,                      
        fp16=False,                       
        dataloader_num_workers=0,         
        gradient_accumulation_steps=4,    
        remove_unused_columns=False,      
        report_to="none",                 
        learning_rate=2e-5
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    # Start training
    logger.info("Starting training...")
    try:
        trainer.train()
        
        # Save final model
        logger.info("Saving model...")
        model.save_pretrained("fine_tuning/models/recipe-bot-final")
        tokenizer.save_pretrained("fine_tuning/models/recipe-bot-final")
        
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()