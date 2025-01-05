import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def preprocess_data(input_file, output_train_file, output_val_file, test_size=0.1):
    print(f"Processing file: {input_file}")
    data = pd.read_csv(input_file, sep='\t')
    print(f"Dataset size: {data.shape}")
    train_data, val_data = train_test_split(data, test_size=test_size, stratify=data['label'], random_state=42)
    train_data.to_csv(output_train_file, sep='\t', index=False)
    val_data.to_csv(output_val_file, sep='\t', index=False)
    print(f"Preprocessing complete. Training set: {len(train_data)}, Validation set: {len(val_data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for training and validation.")
    parser.add_argument("--data", type=str, required=True, help="Input data file path")
    parser.add_argument("--train_output", type=str, default="data/train_split.tsv", help="Training data output file path")
    parser.add_argument("--val_output", type=str, default="data/val_split.tsv", help="Validation data output file path")
    args = parser.parse_args()
    preprocess_data(args.data, args.train_output, args.val_output)
