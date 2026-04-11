import os
import sys
import asyncio
import aiohttp
import requests
import json
import time
from aiohttp import ClientSession
from concurrent.futures import ThreadPoolExecutor

from typing import List, Dict, Any
from .agent_prompts import _SYSTEM_ENV_FEEDBACK_PROMPT, _ENV_FEEDBACK_PROMPT, _SYSTEM_RE_PLANNING_PROMPT, _RE_PLANNING_PROMPT, _GLOBAL_PLAN_SELECTION_PROMPT, _DYNAMIC_GLOBAL_PLAN_SELECTION_PROMPT, _INITIAL_DYNAMIC_GLOBAL_PLAN_PROMPT, _SYSTEM_INITIAL_DYNAMIC_GLOBAL_PLAN_PROMPT, _SYSTEM_DYNAMIC_GOLDEN_GLOBAL_PLAN_EVALUATION_PROMPT, _USER_DYNAMIC_GOLDEN_GLOBAL_PLAN_EVALUATION_PROMPT
from .agent_api import oneapi_post_by_langchain, read_json

def batch_search(url: str, queries: List[str], max_workers: int = 8) -> Any:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for query in queries:
            future = executor.submit(search, url, query)
            futures.append(future)
        results = []
        for i, future in enumerate(futures):
            result = future.result()
            results.append(result)
    return results

# for batch
def tool_simulation_batch(
    # params, 
    agent_action_input_batch: List, 
    # tool_url="http://qwen-14b-instruct-32k.bcloud.proxy.ml.baichuan-inc.com"
    params: Dict = None):
    # include environment feedback and global plan regeneration
    if not agent_action_input_batch:
        return None
    
    # stage 1: environment feedback
    # construct prompt for environment feedback
    index_list = []
    env_system_prompt = _SYSTEM_ENV_FEEDBACK_PROMPT
    env_prompt_list = []
    for i, action_input_json in enumerate(agent_action_input_batch):
        """
        action_input_json = {"instruction", "previous_observation", "agent_action", "data_source"}
        """
        env_prompt_list.append(
            _ENV_FEEDBACK_PROMPT[action_input_json["data_source"]].format(
                instruction=action_input_json.get("instruction", ""),
                previous_observation=action_input_json.get("previous_observation", ""),
                agent_action=action_input_json.get("agent_action", ""),
            )
        )
        index_list.append(i)
        
    # call oneapi (qwen2.5-14b)
    batch_size = 256
    env_response_list = []
    for i in range(0, len(env_prompt_list), batch_size):
        stt = time.time()
        env_response_list += oneapi_post_by_langchain(
            prompt=env_prompt_list[i:i+batch_size],
            system_prompt=env_system_prompt,
            # base_model=Score,
            **params
        )
        edt = time.time()
        print(f"-------- base agent compute score batch size: {len(env_prompt_list[i:i+batch_size])}, oneapi time: {edt - stt} s")
    
    env_result = [res for res in env_response_list]
    
    # stage 2: global plan regeneration
    # construct prompt for global plan regeneration
    planning_system_prompt = _SYSTEM_RE_PLANNING_PROMPT
    planning_prompt_list = []
    for i, action_input_json in enumerate(agent_action_input_batch):
        """
        action_input_json = {"instruction", "previous_observation", "agent_action", "execution_step_index", "data_source"}
        """
        if env_result[i] != "Task Completed!": 
            planning_prompt_list.append(
                _RE_PLANNING_PROMPT[action_input_json["data_source"]].format(
                    instruction=action_input_json.get("instruction", ""),
                    previous_observation=action_input_json.get("previous_observation", ""),
                    execution_step_index=action_input_json.get("execution_step_index", ""),
                )
            )
        else:
            planning_prompt_list.append("Task Completed! No need to re-plan.")
            
    # call oneapi (qwen2.5-14b)
    planning_response_list = []
    for i in range(0, len(planning_prompt_list), batch_size):
        stt = time.time()
        planning_response_list += oneapi_post_by_langchain(
            prompt=planning_prompt_list[i:i+batch_size],
            system_prompt=planning_system_prompt,
            # base_model=Score,
            **params
            # url=tool_url,
        )
        edt = time.time()
        print(f"-------- base agent compute score batch size: {len(planning_prompt_list[i:i+batch_size])}, oneapi time: {edt - stt} s")
    
    planning_result = [res for res in planning_response_list]
    
    return {
        "env_feedback_result": env_result,
        "planning_result": planning_result,
    }
    

# Stage 1: for single item
# env_feedback && global_plan_regeneration
def tool_simulation(
    # params, 
    action_input_json: Dict, 
    # tool_url="http://qwen-14b-instruct-32k.bcloud.proxy.ml.baichuan-inc.com",
    params, 
    data_source="alfworld"):
    """
    action_input_json = {"instruction", "previous_observation", "agent_action", "data_source"}
    """
    
    # include environment feedback and global plan regeneration
    if not action_input_json:
        return None
    
    # phase 1: environment feedback
    # construct prompt for environment feedback
    env_system_prompt = _SYSTEM_ENV_FEEDBACK_PROMPT
                # .format(
                #     instruction=action_input_json.get("instruction", ""),
                #     previous_observation=action_input_json.get("previous_observation", ""),
                #     agent_action=action_input_json.get("agent_action", ""),
                # )
                
    env_prompt = _ENV_FEEDBACK_PROMPT[data_source].format(
                    instruction=action_input_json.get("instruction", ""),
                    previous_observation=action_input_json.get("previous_observation", ""),
                    agent_action=action_input_json.get("agent_action", ""),
                )
    # env_prompt = ""
    
    # call oneapi (qwen2.5-14b)
    stt = time.time()
    env_result = oneapi_post_by_langchain(
                    prompt=env_prompt,
                    system_prompt=env_system_prompt,
                    # base_model=Score,
                    **params
                    # url=tool_url,
                )
    edt = time.time()
    print(f"-------- base agent env feedback oneapi time: {edt - stt} s")
    
    # phase 2: global plan regeneration
    # construct prompt for global plan regeneration
    planning_system_prompt = _SYSTEM_RE_PLANNING_PROMPT
    planning_prompt = "Task Completed! No need to re-plan." # set as default
    if env_result[i] != "Task Completed!":
        planning_prompt = _RE_PLANNING_PROMPT[data_source].format(
                            instruction=action_input_json.get("instruction", ""),
                            previous_observation=action_input_json.get("previous_observation", ""),
                            # execution_step_index=action_input_json.get("execution_step_index", ""),
                        )
    stt = time.time()
    planning_result = oneapi_post_by_langchain(
                        prompt=planning_prompt,
                        system_prompt=planning_system_prompt,
                        # base_model=Score,
                        **params
                        # url=tool_url,
                    )
    edt = time.time()
    print(f"-------- base agent plan regeneration oneapi time: {edt - stt} s")
    
    # return {
    #     "env_feedback_result": env_result,
    #     "planning_result": planning_result,
    # }
    return env_result, planning_result
    

# Stage 2: for single item
# only env_feedback
def tool_simulation_stage2(
    action_input_json: Dict, 
    # tool_url="http://qwen-14b-instruct-32k.bcloud.proxy.ml.baichuan-inc.com",
    params, 
    data_source="alfworld"):
    """
    action_input_json = {"instruction", "previous_observation", "agent_action", "data_source"}
    """
    # print(f"运行到 tool_simulation_stage2 啦!!!")
    # include environment feedback and global plan regeneration
    if not action_input_json:
        return None
    
    # environment feedback
    # construct prompt for environment feedback
    env_system_prompt = _SYSTEM_ENV_FEEDBACK_PROMPT
                
    env_prompt = _ENV_FEEDBACK_PROMPT[data_source].format(
                    instruction=action_input_json.get("instruction", ""),
                    previous_observation=action_input_json.get("previous_observation", ""),
                    agent_action=action_input_json.get("agent_action", ""),
                )
    
    # call oneapi (qwen2.5-14b)
    stt = time.time()
    env_result = oneapi_post_by_langchain(
                    prompt=env_prompt,
                    system_prompt=env_system_prompt,
                    # base_model=Score,
                    **params
                    # url=tool_url,
                )
    edt = time.time()
    print(f"-------- base agent env feedback oneapi time: {edt - stt} s")
    # print(f"env_result: {env_result}")
    
    env_result
    
    
def global_plan_evaluation(
    action_input_json: Dict, 
    # tool_url="http://qwen-14b-instruct-32k.bcloud.proxy.ml.baichuan-inc.com",
    params
):
    if not action_input_json:
        return None

    # construct prompt for plan evaluation
    eval_system_prompt = _SYSTEM_DYNAMIC_GOLDEN_GLOBAL_PLAN_EVALUATION_PROMPT
                
    eval_prompt = _USER_DYNAMIC_GOLDEN_GLOBAL_PLAN_EVALUATION_PROMPT.format(
                    instruction=action_input_json.get("instruction", ""),
                    global_plan=action_input_json.get("global_plan", ""),
                    agent_action=action_input_json.get("agent_action", ""),
                    env_feedback=action_input_json.get("env_feedback", ""),
                )

    # call oneapi (qwen2.5-14b)
    if_succeed = False
    eval_result = {
        "correctness_score": 0.0,
        "correctness_reason": "...",
        "followability_score": 0.0,
        "followability_reason": "...",
        "standardization_score": 0.0,
        "standardization_reason": "..."
    }
    # while not if_succeed:
    #     stt = time.time()
    #     eval_result = oneapi_post_by_langchain(
    #                     prompt=eval_prompt,
    #                     system_prompt=eval_system_prompt,
    #                     **params
    #                     # url=tool_url,
    #                 )
    #     edt = time.time()
    #     print(f"-------- base agent env feedback oneapi time: {edt - stt} s")
    #     try:
    #         eval_result = read_json(eval_result)
    #         # check key_value
    #         # if eval_result and isinstance(eval_result, dict) and \
    #         #    "correctness_score" in eval_result and "correctness_reason" in eval_result and \
    #         #    "followability_score" in eval_result and "followability_reason" in eval_result and \
    #         #    "standardization_score" in eval_result and "standardization_reason" in eval_result:
    #         if eval_result and isinstance(eval_result, dict) and \
    #            "correctness_score" in eval_result and \
    #            "followability_score" in eval_result and \
    #            "standardization_score" in eval_result:
    #                 if_succeed = True
    #         else:
    #             print(f"eval_result does not contain expected keys: {eval_result}")
    #     except Exception as e:
    #         print(f"Error parsing eval_result: {e}, retrying...")
    #         time.sleep(2)
    
    stt = time.time()
    eval_result = oneapi_post_by_langchain(
                    prompt=eval_prompt,
                    system_prompt=eval_system_prompt,
                    **params
                    # url=tool_url,
                )
    edt = time.time()
    print(f"-------- base agent env feedback oneapi time: {edt - stt} s")
    try:
        eval_result = read_json(eval_result)
        # check key_value
        # if eval_result and isinstance(eval_result, dict) and \
        #    "correctness_score" in eval_result and "correctness_reason" in eval_result and \
        #    "followability_score" in eval_result and "followability_reason" in eval_result and \
        #    "standardization_score" in eval_result and "standardization_reason" in eval_result:
        if eval_result and isinstance(eval_result, dict) and \
            "correctness_score" in eval_result and \
            "followability_score" in eval_result and \
            "standardization_score" in eval_result:
                print(f"eval_result: {eval_result}")
        else:
            print(f"eval_result does not contain expected keys: {eval_result}")
            eval_result = {
                "correctness_score": 0.0,
                "correctness_reason": "Evaluation failed.",
                "followability_score": 0.0,
                "followability_reason": "Evaluation failed.",
                "standardization_score": 0.0,
                "standardization_reason": "Evaluation failed."
            }
    except Exception as e:
        print(f"Error parsing eval_result: {e}")
        eval_result = {
            "correctness_score": 0.0,
            "correctness_reason": "Evaluation failed.",
            "followability_score": 0.0,
            "followability_reason": "Evaluation failed.",
            "standardization_score": 0.0,
            "standardization_reason": "Evaluation failed."
        }
    return eval_result


def initialize_global_plans(
    tokenizer,
    instruction_batch: List, 
    # tool_url="http://qwen-14b-instruct-32k.bcloud.proxy.ml.baichuan-inc.com",
    params: Dict = None
):
    if not instruction_batch:
        return None

    instruction_text_batch = tokenizer.batch_decode(instruction_batch, skip_special_tokens=True)
    user_global_plans_prompt = [
        instruction_text + _INITIAL_DYNAMIC_GLOBAL_PLAN_PROMPT 
        for instruction_text in instruction_text_batch
    ]
    print("user_global_plans_prompt length:", len(user_global_plans_prompt))
    system_global_plans_prompt = _SYSTEM_INITIAL_DYNAMIC_GLOBAL_PLAN_PROMPT
    
    batch_size = 256
    global_plan_result_list = []
    for i in range(0, len(user_global_plans_prompt), batch_size):
        stt = time.time()
        global_plan_result_list += oneapi_post_by_langchain(
            prompt=user_global_plans_prompt[i:i+batch_size],
            system_prompt=system_global_plans_prompt,
            # base_model=Score,
            **params
        )
        edt = time.time()
        print("edt - stt:", edt - stt)
    # global_plan_result_list = oneapi_post_by_langchain(
    #                 prompt=user_global_plans_prompt,
    #                 system_prompt=system_global_plans_prompt,
    #                 # url=tool_url,
    #                 **params
    #             )
    print(f"global_plan_result_list: {global_plan_result_list}")
    return global_plan_result_list
    
        
def search(url: str, query: str, top_n=5):
    if not query:
        return None
        
    url = f'{url}/search'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer c4ca4238a0b923820dcc509a6f75849b',
    }
    query_data = {
        "request_id": "keerlutest",
        "query": query, 
        "top_n": top_n,
        "control": {
            "hit_size": 20,
            "timeout": 2000,
        # "debug_level": 1
        }
    }
    retry_count = 0
    while retry_count < 5:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(query_data), timeout=20)
        except Exception as e:
            print(f"Request failed with exception: {e=}, {retry_count=}")
            retry_count += 1
            continue

        if response.status_code == 200:
            response = response.json()
            try:
                debug = []
                response = response["result"]["hits"]
                retrieval_text = ''
                for line in response[:top_n]:
                    chunk = line['fields']['chunk']
                    chunk = chunk.replace("<search>", "").replace("</search>", "").replace("<document>", "").replace("</document>", "")
                    retrieval_text += f"<document>{chunk}</document>"
                    debug.append({
                        "query": query,
                        "chunk": line['fields'].get('chunk'),
                        "author": line['fields'].get('author'),
                        "journal": line['fields'].get('journal'),
                        "language": line['fields'].get('language'),
                        "id": line['fields'].get('id'),
                        "source_id": line['fields'].get('source_id'),
                        "title": line['fields'].get('title'),
                        "url": line['fields'].get('url'),
                        "collection": line.get('collection'),
                        "score": line.get('score'),
                    })
                retrieval_text = retrieval_text.strip()
                return retrieval_text, debug
            except Exception as e:
                print(f"Response parsing failed with exception: {e=}, {response=}, {retry_count=}")
                retry_count += 1
                continue
        else:
            print(f"Request failed with status code: {response.status_code}")
            return None
    
    return None
