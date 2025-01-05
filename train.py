from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import argparse

def train_model(train_file, val_file, model_name, output_dir, num_labels):
    dataset = load_dataset('csv', data_files={'train': train_file, 'val': val_file}, delimiter='\t')
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val']
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained model for text classification.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data file")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation data file")
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased", help="Pre-trained model name")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the fine-tuned model")
    args = parser.parse_args()
    train_model(args.train_file, args.val_file, args.model_name, args.output_dir, num_labels=2)
