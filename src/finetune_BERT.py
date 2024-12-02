from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

# Load the dataset
train_data = Dataset.from_csv('train_data.csv')
val_data = Dataset.from_csv('val_data.csv')

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

# Tokenize data
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

train_data = train_data.map(tokenize, batched=True)
val_data = val_data.map(tokenize, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir='./bert_results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Metrics
def compute_metrics(pred):
    predictions, labels = pred
    preds = torch.argmax(torch.tensor(predictions), dim=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train and save the model
trainer.train()
model.save_pretrained('./bert_swahili_model')
tokenizer.save_pretrained('./bert_swahili_model')
