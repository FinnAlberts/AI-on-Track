# AI on Track - Generating Athletics Articles with Large Language Models

This repository contains the code for generating athletics articles using large language models. The project uses the Hugging Face API to interact with pre-trained models and generate text based on a diverse set of prompts, leveraging different prompt engineering techniques to control the output.
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

4. **Create `.env` file**:

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
Run the `main.py` script to generate articles using the Hugging Face API. When run, the script will prompt you to enter the filename of the file containing the data, ask you which prompt engineering technique to use, and where it should save the generated text.

Please note that generating text using large language models can be computationally expensive and may take some time. Also, the API request will time out after 180 seconds. If the request times out, you can re-run the script to try again.

## Article and citations
This repository is part of a larger project described in the article:

"AI on Track: Generating Accurate and Engaging Athletics Articles with Language Models" by Finn Alberts and Ewoud Vosse (2024).

The article details the research questions, methods, and findings of this project, exploring how large language models (LLMs) can be used to generate high-quality articles for athletics events. If you're interested in the background, methodology, or results, please refer to the article for a comprehensive overview, and is available at [my personal website](https://finnalberts.nl/projecten/open-universiteit/ai-on-track).

If you use this code in your work, please consider citing the article as follows:
```
@article{AiOnTrack,
  title={AI on Track: Generating Accurate and Engaging Athletics Articles with Language Models},
  author={Alberts, Finn and Vosse, Ewoud},
  year={2024},
  institution={Open Universiteit}
}
```
