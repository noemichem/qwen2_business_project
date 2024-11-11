from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])

model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

conversation_layer_1 = [
    {
        "role": "user",
        "content": """
        You are a data engineer skilled in artificial intelligence. Your task is to analyze the dataset descriptor and identify the necessary preprocessing steps for data transformation.

        - **Handling Missing Values**:
            - If a column has few missing values (e.g., less than 50% null data), impute the missing value with the mean.
            - If a column has many missing values (more than 50%), consider removing the column.

        - **Encoding Categorical Variables**:
            - One-Hot Encoding: Transform categorical columns without intrinsic order into binary variables.
            - Label Encoding: If the categories have a logical order (e.g., education levels), use Label Encoding or another ordinal encoding.

        - **Normalization or Standardization of Numerical Columns**:
            - Standardization (zero mean, unit variance): Fit data to a standard normal distribution, useful for scale-sensitive algorithms like linear regression.
            - Normalization (scaling to 0-1 range): Fit data to a fixed interval, suitable for distance-based algorithms like neural networks.

        - **Handling Outliers**:
            - Detect and handle outliers using the Interquartile Range (IQR):
            - Calculate the first quartile (Q1) and the third quartile (Q3) of the column.
            - Define limits using: Lower bound = Q1 - 1.5 * IQR, Upper bound = Q3 + 1.5 * IQR.
            - Filter out data points outside these bounds to improve data quality.

        Based on these guidelines, wait for my dataset descriptor and return a structured list of the preprocessing steps required without providing any code or additional explanations.
        """
    }]

conversation_layer_2 = [{"role": "assistant", "content": """
    You are a data engineer skilled in artificial intelligence. Your task is to preprocess a dataset based on the required preprocessing steps. Given the dataset filepath, write Python code to perform the preprocessing operations listed below, applying each operation directly on the dataset. Avoid any extra explanations or comments in your response return only the code. Ensure the code contains the function `preprocess(df)` with as input the dataframe of the dataset and that returns the preprocessed df. Wait for the next message.
"""
}]

def initialize_and_load_model():
    """
    Initializes and loads the model and tokenizer.

    :returns: None
    """
    global model, tokenizer
    logging.info("Initializing model and tokenizer...")  # Log info message
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    try:
        # Load pre-trained model and tokenizer from Hugging Face model hub
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto"
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info("Model and tokenizer successfully loaded.")  # Log info message
    except Exception as e:
        # Handle any errors that occur during loading
        logging.error(f"Error loading model or tokenizer: {e}")  # Log error message

def add_user_message(message, conversation):
    """
    Adds a user message to the conversation.

    :param message: str, The user's message.
    :returns: None
    """
    logging.debug(f"Adding user message: {message}")  # Log debug message
    conversation.append({"role": "user", "content": message})

def add_assistant_message(message, conversation):
    """
    Adds an assistant message to the conversation.

    :param message: str, The assistant's message.
    :returns: None
    """
    logging.debug(f"Adding assistant message: {message}")  # Log debug message
    conversation.append({"role": "assistant", "content": message})

def get_response(messages):
    """
    Generates a response from the model based on the input messages.

    :param messages: list, A list of dictionaries containing user and assistant messages.
    :returns: str, The generated response.
    """
    global model, tokenizer
    logging.debug(f"Generating response for messages: {messages}")  # Log debug message
    try:
        # Convert conversation to a format suitable for input to the model
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Prepare input data for model
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        # Generate response from model
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Convert generated IDs to text
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        add_assistant_message(response, messages)
        logging.debug(f"Generated response: {response}")  # Log debug message
        return response
    except Exception as e:
        # Handle any errors that occur during generation
        logging.error(f"Error generating response: {e}")  # Log error message
        return ""

def ask_and_get_response(descriptor: dict):
    """
    Requests a response from the Qwen2 service with the input descriptor.

    :param descriptor: dict, A dictionary containing information about the data table.
    :returns: str, The generated response.
    """
    global conversation_layer_1, conversation_layer_2
    if not isinstance(descriptor, dict):
        raise ValueError("The 'descriptor' parameter must be a dictionary.")
    
    # # Verify that the dictionary has the correct keys
    # required_keys = ["missing_data", "duplicates", "categorical_columns", "correlation_matrix", "descriptive_statistics"]
    # if not all(key in descriptor for key in required_keys):
    #     raise ValueError("The input dictionary must contain all required keys.")
    
    prompt = f"""
        df descriptor:
        missing_data: {descriptor['missing_data']}
        duplicates: {descriptor['duplicates']}
        categorical_columns: {', '.join(descriptor['categorical_columns'])}
        descriptive_statistics: {descriptor['descriptive_statistics']}
    """
    # First layer of LLM preprocessing
    logging.debug(f"Requesting with prompt:\n{prompt}")  # Log debug message
    add_user_message(prompt, conversation_layer_1)
    response = get_response(conversation_layer_1)
    
    logging.info(f"Received response from first layer: {response}")  # Log info message

    # Second layer of LLM preprocessing
    add_user_message('Required preprocessing operations:\n'+ response, conversation_layer_2)
    response = get_response(conversation_layer_2)
    logging.info(f"Received response from second layer: {response}")  # Log info message
    fixed_response = extract_code_blocks(response)
    logging.info(f"Extracted code blocks: {fixed_response}")  # Log info message

    return fixed_response

# def format_conversation(output):
#     conversation = []
#     for item in output:
#         role = item['role']
#         content = item['content']
#         if role == 'user':
#             conversation.append(f"User:\n- {content}")
#         elif role == 'assistant':
#             conversation.append(f"Assistant:\n- {content}")
#     return conversation


def extract_code_blocks(text):
    """
    Extracts code blocks from a given text.

    :param text: str, The input text.
    :returns: list, A list of extracted code blocks.
    """
    # Use regular expression to find all Python code blocks
    pattern = r'```python(.*?)```'
    code_blocks = re.findall(pattern, text, re.DOTALL)  # re.DOTALL allows newline in match

    # If no code blocks are found, consider the entire text as a single block of code
    if not code_blocks:
        logging.warning("No code blocks found, returning entire text as code.")  # Log warning message
        code_blocks = [text]

    # Check if generated_code is a list and convert it to a string
    if isinstance(code_blocks, list):
        # Join the elements of the list into a string, separating them with newline
        code_blocks = '\n'.join(code_blocks)

    return code_blocks