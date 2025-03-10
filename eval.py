from utils import extract_answer_from_model_output
import re
import torch

def extract_last_number(text):
   """
   Extracts the last number appearing in the text.

   Args:
       text (str): The text to extract a number from.

   Returns:
       float or None: The last number in the text, or None if no number is found.

   Explanation:
       1. Removes dollar signs and percent symbols from the text.
       2. Uses regex to find a number that appears at the end of the text (possibly after whitespace).
       3. The pattern matches numbers that appear at the end of the string, with or without decimal points.
       4. Returns the found number as a float, or None if no match is found.
   """
   text = text.replace('$', '').replace('%', '')
   pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'
   match = re.search(pattern, text)
   return float(match.group(1)) if match else None

def extract_single_number(text):
   """
   Extracts a single number from text if exactly one number is present.

   Args:
       text (str): The text to extract a number from.

   Returns:
       float or None: The single number in the text, or None if zero or multiple numbers are found.

   Explanation:
       1. Uses regex to find all numbers in the text (including negative numbers and decimals).
       2. If exactly one number is found, returns it as a float.
       3. If zero or multiple numbers are found, returns None.
   """
   numbers = re.findall(r'-?\d*\.?\d+', text)
   return float(numbers[0]) if len(numbers) == 1 else None


def evaluate_model(model, tokenizer,examples, device):
    """
    Evaluate the model on the GSM8K dataset.

    Args:
        model : torch.nn.Module : model to evaluate
        tokenizer : transformers.PreTrainedTokenizer : tokenizer for the model
        examples : list : list of examples from the GSM8K dataset
        device : str : device to run the model on

    Returns:
        dict : evaluation results

    """
    model.eval()
    model.to(device)
    correct = 0
    total = len(examples)
    for example in examples:
        question = example['prompt']
        expected = example['answer']
        inputs = tokenizer(question,return_tensors='pt').to(device)
        
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_length=512,
                number_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                forced_eos_token_id=tokenizer.eos_token_id,
                temperature=0.7,
                early_stopping=False
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        try:
            # Extract answer and check correctness
            predicted = extract_answer_from_model_output(response)

            # Try different matching methods
            if predicted == expected:  # Exact match
                is_correct = True
            else:
                # Try single number matching
                pred_num = extract_single_number(str(predicted))
                exp_num = extract_single_number(str(expected))
                if pred_num is not None and exp_num is not None and pred_num == exp_num:
                    is_correct = True
                else:
                    # Try last number matching
                    pred_num = extract_last_number(str(predicted))
                    exp_num = extract_last_number(str(expected))
                    is_correct = (pred_num is not None and exp_num is not None and
                                pred_num == exp_num)

            # Update counter for correct answers
            if is_correct:
                correct += 1

            # Print evaluation details
            print("\nPrompt:")
            print(full_prompt)
            print("\nExpected Answer:")
            print(expected)
            print("\nExtracted Answer:")
            print(predicted)
            print("\nFull Generated Response:")
            print(response)
            print("\nCorrect:", "✓" if is_correct else "✗")
            print("-"*50)

        except Exception as e:
            print("\nFailed to parse model output for prompt:")
            print(full_prompt)
            print("Error:", e)
            print("-"*50)

    # Calculate and print final accuracy
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    print("="*50)

    # Return model to training mode
    model.train()
    return accuracy
        
        
