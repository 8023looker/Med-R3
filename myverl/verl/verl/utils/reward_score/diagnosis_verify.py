import requests
from functools import wraps
import time
from typing import Dict

def retry(default_result, max: int=1, sleep: int=0):
    """ support error retry. """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == max - 1:
                        print("[Remote RM Error] function: {} Retry failed after {} times: {} !!!!!!".format(func.__name__, max, e))
                        return default_result
                    elif sleep:
                        time.sleep(sleep)
        return wrapper
    return decorator

@retry(default_result={"reward": -1.0, "acc": False}, max=3, sleep=0.0)
def verify(url, data_source, reward_actor, prompt_str, response_str, ground_truth, extra_info=None, params=None):
    json_data = {
        "data_source": data_source,
        "reward_actor": reward_actor,
        "prompt_str": prompt_str,
        "response_str": response_str,
        "ground_truth": ground_truth,
        "extra_info": extra_info,
    }
    if params:
        json_data["params"] = params
    response = requests.post(  
        f'{url}/verify',  
        json=json_data,  
        headers={'Content-Type': 'application/json'},
        timeout=600,
    )  
    return response.json() 

def compute_score(url: str,
                  data_source: str,
                  reward_actor: str,
                  prompt_str: str,
                  response_str: str,
                  ground_truth: str,
                  extra_info: Dict,
                  params: Dict) -> Dict:
    """Compute the reward score for a solution.
    
    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        
    Returns:
        Reward score (1.0 for correct, 0.0 for incorrect)
    """
    # Limit solution length for efficiency
    return verify(url, data_source, reward_actor, prompt_str, response_str, ground_truth, extra_info, params)

