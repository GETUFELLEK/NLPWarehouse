from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the Swahili News Dataset
dataset = load_dataset("swahili_news")

# Convert to DataFrame
df = pd.DataFrame(dataset['train'])

# Debugging: Print column names
print("Available columns:", df.columns)

# Use the correct column names based on dataset structure
df = df[['text', 'label']]  # Adjusted column names to match the dataset

# Split into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
)

# Save processed data
train_data = pd.DataFrame({'text': train_texts, 'label': train_labels})
val_data = pd.DataFrame({'text': val_texts, 'label': val_labels})

train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)

print("Preprocessing complete. Train and validation data saved.")