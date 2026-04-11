# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import json
import requests
from functools import wraps
import time
from typing import Dict
import os

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


@retry(default_result={"score": -1.0, "acc": False, "pred": "[INVALID]"}, max=3, sleep=0.0)  
def math_verify(solution: str, ground_truth: str, template_name: str=None, url=os.environ.get("MATH_VERIFY_SERVER_URL", None)):  
    if url is None:
        raise ValueError("MATH_VERIFY_SERVER_URL is not set")
    json_data = {
        "solution": solution,
        "ground_truth": ground_truth,
    }
    if template_name:
        json_data["template_name"] = template_name
    response = requests.post(  
        f'{url}/verify',  
        json=json_data,  
        headers={'Content-Type': 'application/json'}  
    )  
    return response.json() 

def compute_score(solution_str: str,
                  ground_truth: str) -> Dict:
    """Compute the reward score for a solution.
    
    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        
    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Limit solution length for efficiency
    return math_verify(solution=solution_str, ground_truth=ground_truth)

