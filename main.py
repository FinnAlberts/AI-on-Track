from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import random
import time

def generate_text(model_name: str, prompt: str, max_tokens: int=100) -> tuple[str, int]:
    '''
    Generate text using the Hugging Face Inference API and prints output and tokens used.

    Args:
    model_name: str
    The model name to use for text generation.

    prompt: str
    The input text prompt to generate text from.

    max_tokens: int
    The maximum number of tokens to generate.

    Returns: str, int
    The generated text and the estimated number of tokens used.
    '''    
    # Initialize the Inference API client with the model name and your API token
    # The headers allow the model to wait for the model to be loaded and not use the cache
    client = InferenceClient(model_name, token=os.getenv("HUGGINGFACE_TOKEN"), headers={"x-wait-for-model": "true", "x-use-cache": "false"})

    # Prepare the input messages
    messages = [{"role": "user", "content": prompt}]
    
    # Make the API call. We set stream=True to get a stream of responses. This prevents the API from timing out for large responses.
    print(f"[INFO] Generating text with model: {model_name}")
    response = client.chat_completion(messages=messages, max_tokens=max_tokens, stream=True)

    # Wait loop until we get data from the generator or timeout
    timeout = 180  # seconds
    start_time = time.time()
    while True:
        try:
            # Attempt to get the first item from the generator. If not, this raises a StopIteration exception
            first_message = next(response)

            # We got data, so process first token and break the loop
            print(f"[INFO] Generating response", end="\r")
            token_usage_estimate = 1 # The token usage is an estimate, because of system tokens or special tokens that might be used
            output = first_message.choices[0].delta.content
            break
        except StopIteration:
            # Check if we've reached the timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout: {timeout} seconds exceeded. No data received from the generator.")
            time.sleep(0.1)  # Small delay to avoid busy-waiting                

    # Continue processing the rest of the responses
    for message in response:
        print(f"[INFO] Generating response{'.' * (token_usage_estimate % 3)}{' ' * (3 - token_usage_estimate % 3)}", end="\r")
        token_usage_estimate += 1
        output += message.choices[0].delta.content
    
    print(f"[INFO] Generation complete")

    return output, token_usage_estimate
    
def prompt_builder(data: str, engineering_method: str, examples_path: str) -> str:
    '''
    Build a prompt for the model to generate the report for the athletics match.

    Args:
    data: str
    The data to include in the report

    engineering_method: str
    The type of prompt engineering to use. Options are "zero-shot", "one-shot", "few-shot", "chain of thought" and "auto chain of thought".
    When using few-shot, three examples will be used. For both one-shot and few-shot, the examples will be picked randomly.

    examples_path: str
    The path to the examples folder to use for one-shot and few-shot prompts.

    Returns: str
    The prompt to use for the model.
    '''

    # Check if the type is valid
    valid_types = ["zero-shot", "one-shot", "few-shot", "chain of thought", "auto chain of thought"]
    if engineering_method not in valid_types:
        raise ValueError(f"Invalid type: {engineering_method}. Valid types are: {valid_types}")
    
    # Set up chain_of_thought and examples variables. If not used, they will be empty strings
    chain_of_thought = ""
    examples = []

    # For auto chain of thought, set up the prompt to generate the chain of thought automatically
    if engineering_method == "auto chain of thought":
        chain_of_thought_prompt = "You are tasked with writing a report about the athletics match. You are given the results of the match and do not have access to additional data. Come up with a chain of thought to write the report. Include which steps should be taken to write the report and what information should be included. Output should be a numbered list of steps."

        max_tokens = 500
        chain_of_thought, chain_of_thought_usage = generate_text(model_name="Qwen/Qwen2.5-72B-Instruct", prompt=chain_of_thought_prompt, max_tokens=max_tokens)

        if max_tokens == chain_of_thought_usage:
            print("[WARNING] Maximum tokens used when generating chain of thought. The chain of thought may not be complete.")

        print(f"[INFO] Generated chain of thought: {chain_of_thought}")

    # For (manual) chain of thought, set up the steps to write the report
    if engineering_method == "chain of thought":
        chain_of_thought = "1. Carefully read the data and extract the results of all athletes from Scopias Atletiek.\n2. Write an introduction to the report in which you include what the name of the match was and where it took place.\n3. Write a summary of the match, in which you include the results of Scopias Atletiek athletes.\n4. Write a conclusion in which you summarize the results of the match and give your opinion on the performance of the athletes."

    # For one-shot and few-shot, read the examples from the files
    if engineering_method in ["one-shot", "few-shot"]:
        example_filenames = [f for f in os.listdir(examples_path) if os.path.isfile(os.path.join(examples_path, f))]

        # Shuffle the examples to get a random selection
        random.shuffle(example_filenames)

        # Read the examples from the files and append them to the examples list
        if engineering_method == "one-shot":
            print(f"[INFO] Using example: {example_filenames[0]}")
            with open(os.path.join(examples_path, example_filenames[0]), "r") as file:
                examples.append(file.read())
        else:
            print(f"[INFO] Using examples: {example_filenames[:3]}")
            for filename in example_filenames[:3]:
                with open(os.path.join(examples_path, filename), "r") as file:
                    examples.append(file.read())

    # Build the full prompt
    prompt = "You are a professional copywriter tasked with writing a report about an athletics match, in which Scopias Atletiek participated. The report will be published on the Scopias Atletiek website. Your goal is to write a report that is informative and engaging for the readers. It should thus not simply be a sum up of the results of the match, but also be a pleasure to read. All results of Scopias Atletiek athletes should be included in the report.\n\n Your target audience is the members of Scopias Atletiek, as well as other athletics enthusiasts. The report should be written in a professional and engaging tone, and should be easy to read and understand. Output the report in a markdown format.\n\n"

    if engineering_method in ["chain of thought", "auto chain of thought"]:
        prompt += f"When writing the report follow this chain of thought:\n{chain_of_thought}\n\n"

    if engineering_method in ["one-shot", "few-shot"]:
        prompt += "To help you write the report, here are some examples of previous reports:\n"
        for i, example in enumerate(examples):
            prompt += f"Example {i+1}:\n{example}\n\n"

    prompt += f"Data:\n{data}\n"

    return prompt

if __name__ == "__main__":
    # Load the environment variables
    load_dotenv()

    # Set the model name
    model_name = "Qwen/Qwen2.5-72B-Instruct"

    # Ask user for path to data
    data_path = input("Enter the path to the data file: ")

    # Read the data from the file
    with open(data_path, "r") as file:
        data = file.read()

    # Ask user for the type of prompt engineering to use
    engineering_method = input("Enter the type of prompt engineering to use (zero-shot, one-shot, few-shot, chain of thought, auto chain of thought): ")

    # Ask the user where to save the report
    save_path = input("Enter the path to save the report: ")
    save_path = os.path.normpath(save_path)

    # Build the prompt
    prompt = prompt_builder(data, engineering_method, "example_reports")

    # Generate the text
    report, used_tokens = generate_text(model_name, prompt, max_tokens=4000)

    # Save the report to the specified path
    directory_path = os.path.dirname(save_path)
    if directory_path:
        os.makedirs(directory_path, exist_ok=True)
    with open(save_path, "w") as file:
        file.write(report)

    print(f"Report saved to: {save_path}")
    print(f"Tokens used: {used_tokens}")
