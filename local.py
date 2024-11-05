import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(model_name="gpt2", prompt="Once upon a time", max_length=50):
    # Load the model and tokenizer
    try:
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Check if GPU is available and use it if possible
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate text and measure time
        print("Generating text...")
        start_time = time.time()
        
        with torch.no_grad():  # Disable gradient calculation for faster inference
            outputs = model.generate(inputs.input_ids, max_length=max_length, do_sample=True, temperature=0.7)

        end_time = time.time()
        
        # Decode and display the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated Text:", generated_text)
        
        # Print time taken
        print(f"\nTime taken to generate text: {end_time - start_time:.2f} seconds")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage: Try with different model names, like "gpt2", "EleutherAI/gpt-neo-1.3B", "distilgpt2", etc.
generate_text(model_name="microsoft/Phi-3.5-mini-instruct", prompt="The future of AI is", max_length=100)
