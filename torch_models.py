import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from transformers import pipeline
import os

def choose_model(model_names):
    print("Available models:")
    for idx, name in enumerate(model_names):
        print(f"{idx + 1}. {name}")
    
    choice = int(input("Choose a model by number: ")) - 1
    return model_names[choice]

def chat_with_model_full(model, tokenizer, model_type):
    print("Chat mode activated. Type 'exit' to stop.")
    
    # Ensure pad_token is set for GPT-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Exiting chat...")
            break
        
        if model_type == "generation":
            inputs = tokenizer(user_input, return_tensors="pt", padding=False, truncation=True, max_length=1000).to(model.device)
            #inputs['pad_token_id'] = tokenizer.eos_token_id  # Set pad token id to eos token id
            
            with torch.no_grad():
                outputs = model.generate(inputs['input_ids'], 
                                         max_length=512, 
                                         num_return_sequences=1, 
                                         no_repeat_ngram_size=2, 
                                         attention_mask=inputs['attention_mask'])
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Model: {response}")
        else:
            print("The chosen model is not designed for text generation. Please choose a model like T5 or GPT-2.")
            break

def extract_text_from_pdf(pdf_path):
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

from datasets import Dataset

def prepare_training_data(text, tokenizer, max_length=512):
    # Ensure pad_token is set, use eos_token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    
    inputs = []
    targets = []
    
    # Tokenize the input text with truncation and padding
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)
    
    input_ids = tokens['input_ids'][0]
    
    # Create inputs and targets for training
    inputs.append(input_ids[:-1])  # All tokens except the last one
    targets.append(input_ids[1:])  # All tokens except the first one
    
    return Dataset.from_dict({'input_ids': inputs, 'labels': targets})

def train_model(model, tokenizer, train_data):
    print("Training model...")
    
    train_dataset = prepare_training_data(train_data, tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./results",  # Output directory for model checkpoints
        num_train_epochs=3,      # Number of training epochs
        per_device_train_batch_size=4,  # Batch size for training
        save_steps=10_000,       # Save checkpoint every 10,000 steps
        save_total_limit=2,      # Keep only 2 checkpoints
        logging_dir="./logs",    # Directory for logging
        logging_steps=500,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    return model

def load_model_and_train(model_name, pdf_path):
    # Load model and tokenizer based on chosen model
    if "t5" in model_name.lower():
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    elif "gpt2" in model_name.lower():
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Fine-tune the model
    model = train_model(model, tokenizer, pdf_text)
    
    return model, tokenizer

def save_model(model, tokenizer, save_path="./saved_models_torch"):
    print(f"Saving model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

# Function to generate a response using the trained model
from transformers import pipeline

def chat_with_model(model, tokenizer):
    # Create a pipeline for text generation (you can customize it based on your model)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    
    print("Chat with the model (type 'exit' to stop):")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting chat...")
            break
        
        # Generate response from the model based on the user input
        response = generator(user_input, max_length=50, num_return_sequences=1, truncation=True)
        print("Model:", response[0]['generated_text'])

# Load a previously saved model
def load_saved_model(saved_model_path="./saved_models_torch"):
    if os.path.exists(saved_model_path):
        print(f"Loading saved model from {saved_model_path}...")
        model = GPT2LMHeadModel.from_pretrained(saved_model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(saved_model_path)
        return model, tokenizer
    else:
        print(f"No saved model found at {saved_model_path}. Starting fresh.")
        return None, None

# Function to start model training
def train_model_and_save(model_name, pdf_path):
    # Load and fine-tune the model on the PDF content
    model, tokenizer = load_model_and_train(model_name, pdf_path)
    
    # Save the trained model
    save_model(model, tokenizer)
    
    return model, tokenizer

if __name__ == "__main__":
    model_names = ["t5-small", "t5-base", "gpt2", "bert-base-uncased", "distilbert-base-uncased", "EleutherAI/gpt-j-6B"]
    model_name = choose_model(model_names)
    pdf_path = "./test_data/test_ant.pdf"  # Replace with the path to your PDF file

    # Ask the user if they want to load a saved model or train a new one
    user_choice = input("Do you want to (1) Train the model or (2) Load the saved model? Enter 1 or 2: ")

    if user_choice == '1':
        # Train and save the model
        model, tokenizer = train_model_and_save(model_name, pdf_path)
    elif user_choice == '2':
        # Load the saved model if it exists
        model, tokenizer = load_saved_model()

        if model is None or tokenizer is None:
            print("No model found to load. You need to train the model first.")
            exit()
    else:
        print("Invalid choice. Exiting.")
        exit()

    # Start chatting with the model after loading or training
    chat_with_model(model, tokenizer)