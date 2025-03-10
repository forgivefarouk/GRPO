from utils import extract_answer_from_model_output, extract_single_number

def correctness_reward(completions , answers):
    """
    Calculate correctness reward for each completion.

    Args:
        completions : list : list of completions

    Returns:
        list : list of rewards
    """
    rewards = []
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_answer_from_model_output(response) for response in responses]
    for response, answer in zip(extracted, answers):
        if response == answer:
            rewards.append(2.0)
        else:
            response_num = extract_single_number(str(response))
            answer_num = extract_single_number(str(answer))
            if response_num is not None and answer_num is not None and response_num == answer_num:
                rewards.append(1.5)
            else:
                rewards.append(0.0)
    completions_len =[len(completion) for completion in completions]
    return rewards


def format_reward(completions):
    """
    Assigns a reward for adhering to the desired XML format.

    Args:
        completions (list): List of model completions, each containing content.
        **kwargs: Additional keyword arguments.

    Returns:
        list: List of format compliance scores for each completion.

    Explanation:
        1. Extracts the content from each completion.
        2. Evaluates format compliance by checking for required XML tags:
            - 0.2 points for each tag present (<reasoning>, </reasoning>, <answer>, </answer>)
            - Maximum score of 0.8 for perfect format compliance
        3. Stores and returns the format compliance scores.
    """
    formatted_rewards = []
    responces = [completion[0]['content'] for completion in completions]
    for responce in responces:
        score = 0.0
        if responce.find('<reasoning>') != -1:
            score += 2.0
        if responce.find('</reasoning>') != -1:
            score += 2.0
        if responce.find('<answer>') != -1:
            score += 2.0
        if responce.find('</answer>') != -1:
            score += 2.0
        formatted_rewards.append(score)
    return formatted_rewards


def combined_reward(completions, answers):
    """
    Calculate combined reward for each completion.

    Args:
        completions : list : list of completions

    Returns:
        list : list of rewards
    """
    correctness_rewards = correctness_reward(completions, answers)
    format_rewards = format_reward(completions)
    combined_rewards = [correctness_rewards[i] + format_rewards[i] for i in range(len(completions))]
    return combined_rewards