# Turkish Parliamentary Speech Classification

This repository contains scripts for classifying parliamentary speeches into:
1. **Political Ideology**: Identifying whether the speaker's party leans left (0) or right (1).
2. **Power Orientation**: Identifying whether the speaker’s party is currently in government (0) or in opposition (1).

## Repository Contents
- `preprocess.py`: Splits and prepares the dataset for training by creating training and validation sets.
- `train.py`: Fine-tunes a pre-trained model for text classification tasks.
- `test.py`: Evaluates the fine-tuned model on the test dataset and saves predictions.

## Requirements
- python:
- transformers
- torch
- pandas
- scikit-learn
