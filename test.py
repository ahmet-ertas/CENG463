from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch
from sklearn.metrics import classification_report
import argparse

def evaluate_model(test_file, model_dir, output_file):
    data = pd.read_csv(test_file, sep='\t')
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    texts = list(data['text'])

    tokens = tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(**tokens)
        predictions = torch.argmax(outputs.logits, axis=1).numpy()

    data['predicted_label'] = predictions
    data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    print("Performance Metrics:")
    print(classification_report(data['label'], predictions))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on a test dataset.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test data file")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory with the fine-tuned model")
    parser.add_argument("--output_file", type=str, default="results/test_predictions.csv", help="Output file for predictions")
    args = parser.parse_args()
    evaluate_model(args.test_file, args.model_dir, args.output_file)
