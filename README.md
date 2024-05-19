# MediBot ☤🩺

![](https://github.com/your_username/your_repository/raw/main/example_image.png)
##BioMistral-7B.Q4_K_M.gguf with Langchain

This repository provides a quick setup for using the BioMistral-7B.Q4_K_M.gguf model with the Langchain library.

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates how to use the BioMistral-7B.Q4_K_M.gguf model with Langchain for natural language processing tasks. BioMistral-7B is a specialized language model designed for biomedical text, and Langchain is a flexible library that simplifies the integration and usage of large language models.

## Setup

### Requirements

- Python 3.7+
- `requirements.txt` file for dependencies
- BioMistral-7B.Q4_K_M.gguf model file

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/biomistral-langchain-setup.git
   cd biomistral-langchain-setup
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```
3. Install the required libraries:
   ```bash
   pip install langchain
   ```
4. Download the BioMistral-7B.Q4_K_M.gguf model file and place it in your desired directory. Update the path in the usage example accordingly.

### Usage
Here's an example script to get started with the BioMistral model using Langchain:
```python
from langchain.llms import CTransformers

# Define the path to the BioMistral-7B.Q4_K_M.gguf model file
biomistral_model_path = "C:\\path\\to\\your\\BioMistral-7B.Q4_K_M.gguf.bin"

# Initialize the BioMistral model with the appropriate parameters
biomistral_model = CTransformers(
    model=biomistral_model_path,
    model_type="mistral",
    config={
        'max_new_tokens': 1000,
        'temperature': 0.75,
        'context_length': 2000
    }
)

# Example usage of the model
prompt = "What are the symptoms of diabetes?"
response = biomistral_model(prompt)
print(response)
```
Replace "C:\\path\\to\\your\\BioMistral-7B.Q4_K_M.gguf.bin" with the actual path to your BioMistral model file.


### Configuration
You can customize the model's behavior by modifying the config dictionary:

1. max_new_tokens: Maximum number of tokens to generate.
2. temperature: Sampling temperature, controlling the randomness of predictions.
3. context_length: Maximum context length for the input.

### Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

### License
This project is licensed under the MIT License. 
