import ollama
import json
from termcolor import colored
import fitz
import tkinter as tk
from tkinter import filedialog

# Set up the connection to Ollama API
# ollama.set_api_key("YOUR_OLLAMA_API_KEY")  # Replace with your Ollama API key if needed

def pdf_to_text(pdf_path):
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Extract text from all pages
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        return text
    except Exception as e:
        return f"Error: {str(e)}"

def choose_file():
    # Create a Tk root window (it won't show up)
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open a file dialog and return the chosen file path
    file_path = filedialog.askopenfilename(title="Choose a file")
    
    # Return the selected file path (if any)
    if file_path:
        return file_path
    else:
        return "No file selected."

def chat_with_deepseek(prompt):
    try:
        response = ollama.chat(model="deepseek-r1:1.5b", messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("Chat with DeepSeek R1 (type 'exit' to quit)")
    
    while True:
        prompt = input("You: ")
        if prompt.lower() == 'fileupload':
            confirmation = input("Are you sure you want to upload a file? (yes/no): ")
            if confirmation.lower() == 'yes':
                filepath = choose_file()            
                prompt = pdf_to_text(filepath)
            prompt += "\n" + input("file uploaded, add you query: ")
            response = chat_with_deepseek(prompt)
            print(colored(f"DeepSeek R1: {response}", 'yellow'))
        elif prompt.lower() == 'exit' or prompt.lower() == 'quit':
            print("Exiting.")
            break
        else:
            response = chat_with_deepseek(prompt)
            print(colored(f"DeepSeek R1: {response}", 'yellow'))

if __name__ == "__main__":
    main()
