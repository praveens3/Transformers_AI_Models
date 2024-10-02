import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PreTrainedTokenizerFast, PreTrainedModel
from transformers import TFAutoModelForCausalLM
#from transformers import transformers as tf_transformers

class GPTModel:
    def __init__(self, model_name="EleutherAI/gpt-j-6B", download=False):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        if download:
            self.download_model()  # Download model if flag is set
        else:
            #self.load_model()  # Load model from local cache
            self.load_model_from_directory()

    def download_model(self):
        """Download the model and tokenizer from Hugging Face."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = TFAutoModelForCausalLM.from_pretrained(self.model_name)
        # Optional: Save the model if needed
        self.save_model()

    def load_model(self):
        """Load the model and tokenizer from local files only."""
        # Set environment for offline mode
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.model = TFAutoModelForCausalLM.from_pretrained(self.model_name, local_files_only=True)
            print(f"Successfully loaded model and tokenizer for {self.model_name}.")
        except (OSError, ValueError) as e:
            print(f"Error loading model '{self.model_name}': {e}")
            # Handle additional logic here if necessary (e.g., fallback mechanism)

    def save_model(self):
        """Save the model and tokenizer to the specified directory."""
        save_directory = f"./tmodels/" + self.model_name
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        print(f"Model and tokenizer saved to {save_directory}.")

    def load_model_from_directory(self):
        """Load the model and tokenizer from the specified directory."""
        try:
            load_directory = f"./tmodels/" + self.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(load_directory)
            self.model = TFAutoModelForCausalLM.from_pretrained(load_directory)
            print(f"Successfully loaded model and tokenizer from {load_directory}.")
        except (OSError, ValueError) as e:
            print(f"Error loading model from '{load_directory}': {e}")

    def query(self, inputs):
        """Generate text based on input."""
        # Tokenize the input
        inputs = self.tokenizer(inputs, return_tensors="tf")  # Use "pt" for PyTorch or "tf" for TensorFlow
        # Generate text
        outputs = self.model.generate(**inputs, max_length=200)
        # Decode and return the output
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    # To download the model
    #gpt_model = GPT2Model("EleutherAI/gpt-j-6B", download=True)  # Set download=True to download the model
    #gpt_model = GPT2Model("w11wo/indo-gpt2-small", download=False)
    #gpt_model = GPT2Model("Salesforce/codegen-6B-mono", download=True)
    gpt_model = GPTModel("openai-community/gpt2")  # Set download=True to download the model

    # To load the model from local files
    # gpt_model = GPT2Model()  # By default, it will load from local files

    while True:
        inputs = input("Enter your query (or type 'exit' to quit): ")
        
        if inputs.lower() == "exit":
            print("Exiting the loop.")
            break  # Exit the loop
        
        output = gpt_model.query(inputs)  # Generate output based on input
        print("\033[92m" + output + "\033[0m")  # Green text
        # If you want to highlight with a background color (e.g., yellow)
        # print("\033[93m" + output + "\033[0m")  # Yellow text