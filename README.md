# LLM Inference Project

This repository contains code for using Large Language Models (LLMs) both locally and via the Hugging Face API. Follow the instructions below to set up the environment and configure access to Hugging Face's Inference API.

## Requirements

- Python 3.9.13
- Hugging Face account with API access (Free tier allows models up to 10GB, with a limit of 1000 requests per day)

## Setup

1. **Clone the repository and navigate to the project directory**:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment (recommended)**:

   ```bash
    python3 -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch:**

   Since PyTorch installations vary based on your operating system and hardware (e.g., CUDA support), please refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/). Follow the instructions provided to install the appropriate version of PyTorch.

5. **Create `.env` file**:

   Use the `.env-example` file in the repository as a template. Copy it and replace `<your-huggingface-api-key>` with your actual API key. Ensure the key has permissions for "Make calls to the serverless Inference API."

   ```
   HUGGINGFACE_TOKEN=<your-huggingface-api-key>
   ```

## Hugging Face Free Tier
Using the Hugging Face API free tier, you can:

- Run models up to 10GB in size
- Make up to 1000 requests per day

This is sufficient for testing and experimenting with many Hugging Face models.

## Usage
Run `api.py` to interact with the Hugging Face API. The script will load the model and tokenizer specified in the `.env` file and generate text based on the provided prompt. The comments in the code provide additional information on how to customize the script.

Run `local.py` to use the model locally. The script will load the model from Hugging Face and generate text based on the provided prompt. The comments in the code provide additional information on how to customize the script. Please take into account that running large models locally may require significant computational resources.
