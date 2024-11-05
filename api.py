from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

def generate_text_with_huggingface_inference(model_name: str, prompt: str, max_tokens: int=100) -> None:
    '''
    Generate text using the Hugging Face Inference API and prints output and tokens used.

    Args:
    model_name: str
    The model name to use for text generation.

    prompt: str
    The input text prompt to generate text from.

    max_tokens: int
    The maximum number of tokens to generate.

    Returns: None
    '''    
    # Initialize the Inference API client with the model name and your API token
    client = InferenceClient(model_name, token=os.getenv("HUGGINGFACE_TOKEN"), headers={"x-wait-for-model": "true"})

    # Prepare the input messages
    messages = [{"role": "user", "content": prompt}]
    
    # Make the API call
    try:
        response = client.chat_completion(messages=messages, max_tokens=max_tokens)

        # Extract the generated text and tokens used from the response
        generated_text = response["choices"][0]["message"]["content"]
        generated_tokens = response["usage"]

        # Print output
        print("Generated Text:", generated_text)
        print("Used amount of tokens:", generated_tokens)
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Load .env file for Hugging Face API token
load_dotenv()

# Example usage: using the mistralai/Mixtral-8x7B-Instruct-v0.1 model to generate text
generate_text_with_huggingface_inference(
    model_name="Qwen/Qwen2.5-72B-Instruct", 
    prompt="Write an introduction to AI and its applications in healthcare.",
    max_tokens=500
    )