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
    os.environ['PYTHONHASHSEED'] = str(seed)

set_random_seed(42)
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
    

    
    

    