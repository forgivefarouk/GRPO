import torch
import numpy as np
import pandas as pd
import random
import copy
import os
import re
import wandb
import torch.nn as nn
import torch.nn.utils.rnn as pad_sequence
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY')
os.environ["WANDB_PROJECT"] = "GRPO"


SYSTEM_PROMPT="""
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# Set random seed for reproducibility
def set_random_seed(seed : int= 42):
    """
    Set random seed for reproducibility
    
    Args:
    seed : int : random seed
    
    Returns:
    None
    
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Extract answer from model output
def extract_answer_from_model_output(output):
    """
    Extract answer from model output
    
    Args:
    output : str : model output
    
    Returns:
    answer : str : answer
    
    """
    first = output.split('<answer>')[1]
    if len(first) < 2:
        return None
    
    if first.find('</answer>') == -1:
        return None
    answer = first.split('</answer>')[0].strip()
    
    return None if answer=="..." else answer


# Extract the final answer from GSM8K dataset
def extract_answer_from_dataset_output(text):
    """
   Extracts the answer from the GSM8K dataset examples.

   Args:
       text (str): The dataset example text containing a question and answer.

   Returns:
       str or None: The extracted answer part after the '####' delimiter, or None if not found.

   Explanation:
       1. Checks if the text contains the '####' delimiter that separates question from answer.
       2. If found, splits the text at this delimiter and returns the second part (the answer).
       3. The answer is stripped of leading/trailing whitespace.
       4. Returns None if no delimiter is present.
   """
   
    if '####' not in text:
       return None
   
    return text.split('####')[-1].strip()
    

    
def prepare_dataset(split="train"):

    """
    Prepare the GSM8K dataset for training and evaluation.

    Args:
        split (str): The dataset split to use, one of "train", "validation", or "test".

    Returns:
        List[Dict[str, str]]: The prepared dataset as a list of dictionaries.

    Explanation:
        1. Loads the GSM8K dataset using the Hugging Face datasets library.
        2. Selects the specified split (train, validation, or test).
        3. Converts the dataset to a string using build_prompt().
        4. Extracts the answer part from the dataset examples.
        5. Returns the prepared dataset.
    """
    
    dataset = load_dataset('openai/gsm8k', 'main')[split]
    formated_data = []
    for example in dataset:
        prompt_str = build_prompt([
            {
                "role":"system","content":SYSTEM_PROMPT,
                "role":"user","content":example["question"]
            }])
        formated_example = {
            "prompt":prompt_str,
            "answer":extract_answer_from_dataset_output(example["answer"])
            
        }
        formated_data.append(formated_example)

    return formated_data


def build_prompt(messages):
   """
   Build a single prompt string from a list of messages.

   Args:
       messages (list): A list of message dictionaries, each with 'role' and 'content' keys.

   Returns:
       str: A concatenated string of all message contents.

   Explanation:
       1. Takes a list of message dictionaries in the typical chat format.
       2. Extracts the 'content' field from each message and strips whitespace.
       3. Joins all content strings with newlines to create a single prompt.
       4. This preserves the training format while converting from structured messages to a string.
   """
   return "\n".join([msg["content"].strip() for msg in messages])





    