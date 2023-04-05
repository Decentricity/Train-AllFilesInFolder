import os
import glob
from transformers import (
    BertTokenizer, BertForSequenceClassification, 
    Trainer, TrainingArguments
)
from datasets import Dataset, load_dataset


# List all non-Python files in the same folder as the script
folder_path = os.path.dirname(os.path.realpath(__file__))
data_files = [f for f in glob.glob(folder_path + "/*") if not f.endswith(".py")]

# Load the files as plain text datasets
datasets = load_dataset("text", data_files={"train": data_files})

# Merge all loaded files into a single train dataset
train_dataset = datasets["train"]

# Preprocess the dataset
def tokenize_and_preprocess(examples):
    tokenizer = BertTokenizer.from_pretrained("cahya/bert-base-indonesian-1.5G")
    
    # Tokenize the text
    input_encodings = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    return input_encodings

train_dataset = train_dataset.map(tokenize_and_preprocess, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Train the model using Hugging Face Trainer
model = BertForSequenceClassification.from_pretrained("cahya/bert-base-indonesian-1.5G")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
