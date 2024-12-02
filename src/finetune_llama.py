from transformers import LlamaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
# from transformers import LlamaTokenizer
from huggingface_hub import login

from transformers import AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


# Log in programmatically (replace 'your_huggingface_token' with your token)
login(token="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")

# tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=True)

# Load the dataset
train_data = Dataset.from_csv('train_data.csv')
val_data = Dataset.from_csv('val_data.csv')

# Load Llama-2 tokenizer and model
# tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
model = LlamaForSequenceClassification.from_pretrained('meta-llama/Llama-2-7b-hf', num_labels=6)

# Tokenize data
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

train_data = train_data.map(tokenize, batched=True)
val_data = val_data.map(tokenize, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir='./llama2_results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Larger models may require smaller batches
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,  # Mixed precision for large models
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
)

# Train and save the model
trainer.train()
model.save_pretrained('./llama2_swahili_model')
tokenizer.save_pretrained('./llama2_swahili_model')
