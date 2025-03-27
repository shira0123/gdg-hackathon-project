# Import libraries
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure dataset exists
dataset_path = "./datasets/labeled_data.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

# Load dataset
df = pd.read_csv(dataset_path)

# Convert class labels to numerical values
label_map = {0: 0, 1: 1, 2: 2}  # 0 = neutral, 1 = offensive, 2 = hate
df['label'] = df['class'].map(label_map)  
texts = df['tweet'].tolist()
labels = df['label'].tolist()  # Numerical labels

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Load DistilBERT tokenizer & model
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.to(device)

# Tokenize dataset
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Create PyTorch dataset class
class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset instances
train_dataset = HateSpeechDataset(train_encodings, train_labels)
val_dataset = HateSpeechDataset(val_encodings, val_labels)

# Define training arguments for GPU optimization
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,  # Reduce if memory issues
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=2,  # Reduce VRAM usage
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",  # Evaluate at end of each epoch
    save_strategy="epoch",  # Save checkpoint at end of each epoch
    save_total_limit=1,  # Keep only the latest checkpoint
    metric_for_best_model="eval_loss",  # Track best model using validation loss
    fp16=torch.cuda.is_available(),  # Enable mixed precision if CUDA is available
)

# Train the model using Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

print("Training started...")
trainer.train()
print("Training complete!")

# Save fine-tuned model locally
output_dir = "./fine_tuned_hate_speech_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f" Model saved to {output_dir}")
