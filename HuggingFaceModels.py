from transformers import pipeline
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, TFTrainingArguments
#from tensorflow.keras.optimizers import Adam
# import transformers
# print (transformers.__version__)

def f1():
    # Load a pre-trained sentiment-analysis model from Hugging Face
    classifier = pipeline("text-classification", model="roberta-base-openai-detector")

    # Input text data (you can replace this with your log file text or other data)
    text_data = [
        "The system is running smoothly without any issues.",
        "There is a failure in the data processing pipeline.",
        "All modules executed successfully, and logs look clean.",
        "Error occurred while trying to fetch data from the database.",
        "Error in storage"
    ]

    # Run the classifier on each text input
    for text in text_data:
        result = classifier(text)
        print(f"Input: {text}\nClassification: {result}\n")

def f2():
    from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
    import tensorflow as tf

    # Load a pre-trained BERT model and tokenizer for TensorFlow
    model_name = "bert-base-cased"
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Example log data for classification
    log_data = [
        "Error: Failed to connect to the database. Timeout occurred.",
        "System running smoothly with no issues.",
        "Warning: Memory usage is high.",
        "Fatal error in module X: unexpected shutdown."
    ]

    # Tokenize the log data
    inputs = tokenizer(log_data, return_tensors="tf", padding=True, truncation=True)

    # Use the model to classify the logs
    outputs = model(inputs['input_ids'])
    predictions = tf.nn.softmax(outputs.logits, axis=-1)

    # Print classification results
    print("Log Predictions:")
    for i, log in enumerate(log_data):
        label = tf.argmax(predictions[i]).numpy()
        print(f"Log: {log} | Prediction Label: {label}")

def read_log_file():
    # Load your log file into a pandas DataFrame
    log_file_path = './Zookeeper_2k.log'

    import os
    print(os.getcwd())

    # Change the working directory to the directory of the executing .py file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(os.getcwd())

    # Read the log file
    with open(log_file_path, 'r') as file:
        logs = file.readlines()

    # Convert to a DataFrame
    logs = pd.DataFrame(logs, columns=["log_entry"])

    # Function to parse log entries
    def parse_log_entry(log_entry):
        try:
            parts = log_entry.split(' - ')
            date_time = parts[0]
            severity = parts[1].split()[0]  # Get severity level (e.g., INFO)
            msg = parts[1].split()[1] + ' - ' + parts[2]  # Directly take the message
            return date_time, severity, msg
        except Exception:
            return None, None, None  # Handle parsing errors


    # Apply the parsing function
    parsed_logs = logs['log_entry'].apply(parse_log_entry)

    # Create a new DataFrame with structured columns
    structured_logs = pd.DataFrame(parsed_logs.tolist(), columns=["date_time", "severity", "msg"])

    # Create a Hugging Face Dataset
    dataset = Dataset.from_pandas(structured_logs)

    # Inspect the dataset
    print(dataset)
    
    return structured_logs

# Define the original models dictionary
models = { #model_name : model_Id
    "distilbert" : "distilbert/distilbert-base-uncased",
    "bert" : "bert-base-uncased"
}

def build_model(structured_logs, model_name):

    # Sample 100 entries to reduce training time
    # Use the first 100 entries to reduce training time
    structured_logs = structured_logs.iloc[:100]

    # Create a Hugging Face Dataset from your structured logs
    dataset = Dataset.from_pandas(structured_logs)

    # Convert severity levels to numerical labels
    severity_mapping = {severity: idx for idx, severity in enumerate(structured_logs['severity'].unique())}
    structured_logs['label'] = structured_logs['severity'].map(severity_mapping)

    # Update the dataset with labels
    dataset = Dataset.from_pandas(structured_logs)

    # Load a tokenizer
    model_Id = models[model_name]  # or another model of your choice
    tokenizer = AutoTokenizer.from_pretrained(model_Id)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['msg'], padding="max_length", truncation=True)

    # Tokenize the messages
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Split into train and test sets
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    return severity_mapping, train_dataset, test_dataset, tokenizer

def train_the_model(model_name, severity_mapping, train_dataset, test_dataset, tokenizer):
    # Load the model
    model = TFAutoModelForSequenceClassification.from_pretrained(models[model_name], num_labels=len(severity_mapping))

    tf_dataset = model.prepare_tf_dataset(train_dataset, batch_size=16, shuffle=True, tokenizer=tokenizer)

    model.compile(optimizer='adam')  # No loss argument!
    model.fit(tf_dataset)  # doctest: +SKIP

    return model

    # Evaluate the model
    # eval_results = trainer.evaluate()
    # print(eval_results)

def search(model_name, model, tokenizer)       :
    # Example log messages for prediction
    new_logs = ["[/10.10.34.11:3888:QuorumCnxManager$Listener@493] - Received connection request /10.10.34.11:46812"]
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(models[model_name])
    new_inputs = tokenizer(new_logs, padding=True, truncation=True, return_tensors="tf")

    # Get predictions
    predictions = model(new_inputs)
    #predicted_labels = predictions.logits.argmax(axis=-1)
    
    # Get predicted labels
    predicted_labels = tf.argmax(predictions.logits, axis=-1).numpy()

    # Map predictions back to severity
    predicted_severity = [list(severity_mapping.keys())[idx] for idx in predicted_labels]
    print(predicted_severity)

    # Map predictions back to severity
    # predicted_severity = [list(severity_mapping.keys())[idx] for idx in predicted_labels.numpy()]
    # print(predicted_severity)

def save_model(model, model_name):
    #model.save(f"./tmodels/{model_name}.keras")
    model.save_pretrained(f"./tmodels/{model_name}.h5")
    print(f"Model saved to tmodels/{model_name}.h5")

def load_model(modal_name, severity_mapping):
    #model = tf.keras.saving.load_model(f"./tmodels/{modal_name}.keras", , custom_objects=None, compile=True, safe_mode=True)
    model = TFAutoModelForSequenceClassification.from_pretrained(f"./tmodels/{modal_name}.h5")
    model.compile(optimizer='adam')  # No loss argument!
    print(f"Model loaded from tmodels/{modal_name}.h5")
    return model

structured_logs = read_log_file()

model_name = "bert"
# Check if a model already exists
try:
    severity_mapping, train_dataset, test_dataset, tokenizer = build_model(structured_logs, model_name)
    model = load_model(model_name, severity_mapping)  # Try to load the existing model
    print("Existing model found, name: {models[model_name]}")
except Exception as e:
    print("No existing model found, training a new one.")
    
    model = train_the_model(model_name, severity_mapping, train_dataset, test_dataset, tokenizer)
    save_model(model, model_name)  # Save the model after training

# Use the model for predictions
search(model_name, model, tokenizer)

