from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
dataset["train"][100]


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
