# train.py

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import evaluate
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # 👈 Force PyTorch backend

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# 1. Load data
df = pd.read_csv('data.csv')  # CSV must have columns 'text' and 'label'
# If needed, rename or handle missing headers:
if 'label' not in df.columns or 'text' not in df.columns:
    df.columns = ['text', 'label']
texts = df['text'].astype(str).tolist()
labels = df['label'].astype(str).tolist()

# 2. Encode labels to integers
le = LabelEncoder()
labels_enc = le.fit_transform(labels)
num_labels = len(le.classes_)
id2label = {i: label for i, label in enumerate(le.classes_)}
label2id = {label: i for i, label in enumerate(le.classes_)}

# 3. Split into train/validation sets (stratified)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels_enc, test_size=0.2, random_state=42, stratify=labels_enc)

# 4. Tokenize texts
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize_fn(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)
train_dataset = list(zip(train_texts, train_labels))
val_dataset = list(zip(val_texts, val_labels))

# Using HuggingFace Datasets API for convenience
from datasets import Dataset
train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_ds   = Dataset.from_dict({"text": val_texts,   "label": val_labels})
train_ds = train_ds.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length'), batched=True)
val_ds   = val_ds.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length'), batched=True)
train_ds = train_ds.rename_column("label", "labels")
val_ds   = val_ds.rename_column("label", "labels")
train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 5. Compute class weights for imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# 6. Prepare model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  # 3 classes
])


# 7. Define Trainer with weighted loss to handle class imbalance
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments(
    output_dir="./model_output",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 8. Train the model
train_result = trainer.train()
metrics = trainer.evaluate()
print(f"Validation results: {metrics}")

# 9. Save the trained model and tokenizer
trainer.save_model("model_output")  # saves config, model (.bin), etc.
tokenizer.save_pretrained("model_output")
print("Model and tokenizer saved to ./model_output/")
