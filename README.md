# Transformers_AI_Models
 
 # Log Analysis and GPT Model Framework

## Overview

This repository contains two Python scripts designed for analyzing log data and generating text responses using GPT models. The primary objectives are to classify log entries and to interactively query a generative pre-trained transformer (GPT) model.

### Scripts

1. **GPT Model Interaction (gpt_model.py)**

   This script provides a framework for loading a GPT model from Hugging Face, allowing users to interactively query the model and receive text responses. It supports downloading the model, loading it from local storage, and generating text based on user input.

   **Key Features:**
   - Load models from Hugging Face or local files.
   - Tokenize input queries and generate text using a pre-trained GPT model.
   - Save and load models to/from a specified directory.
   - Interactive command line interface for real-time queries.

   **Intended Use Cases:**
   - Generate creative text responses based on user input.
   - Explore the capabilities of generative language models in various applications.

2. **Log Analysis and Sentiment Classification (log_analysis.py)**

   This script is designed for analyzing log files and classifying log entries using pre-trained models. It includes functions for reading logs, preprocessing the data, training a classification model, and making predictions based on the log content.

   **Key Features:**
   - Load and parse log files into structured data.
   - Use pre-trained models like BERT and RoBERTa for sentiment analysis and log classification.
   - Convert log severity levels into numerical labels for model training.
   - Train a new model or load an existing one to classify log entries.
   - Generate predictions for new log entries based on trained models.

   **Intended Use Cases:**
   - Classify log entries based on severity levels (e.g., ERROR, WARNING, INFO).
   - Automate log analysis to identify potential issues in system operations.
   - Train custom models on user-specific log data for tailored predictions.

## Dependencies

To run these scripts, ensure you have the following packages installed:
- `transformers`
- `tensorflow`
- `pandas`
- `datasets`

You can install the required packages using pip:

```bash
pip install transformers tensorflow pandas datasets

