""" Extract the reasoning traj from the entire reasoning process """
""" add retry mechanism """

import os
import json
import ujson
import openai
# from openai.error import OpenAIError
from tqdm import tqdm
import sys
import re
import concurrent.futures
import copy
import time
import requests
import subprocess

import prompts
import utils_func

# API setting constants
API_MAX_RETRY = 10
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"
PARALLELISM = 100 # 100

MAX_ATTEMPT = 10

MAX_TOKEN = {
    "deepseek-r1": 7400,
    "deepseek-v3": 8192,
}


class DeepSeekReasoningTraj: # extract the reasoning traj
    def __init__(self, extract_model_name, api_source, hit_size=20, timeout=2000):
        self.api_source = api_source
        if self.api_source == "xxx":
            self.client = openai.OpenAI(
                base_url="xxx",
                api_key="sk-xxx")
        elif self.api_source == "xxx":
            self.client = openai.OpenAI(
                base_url="xxx",
                api_key="sk-xxx")
        self.extract_model = extract_model_name
        
        # query
        self.url = "xxx"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': 'xxx',
            # 'traceparent': traceparent  # 替换为实际的 traceparent
        }
        
        
    def extract_reaoning_traj(self, query_dict, total_template_num=10, temperature=0.9, require_json=True): # for each instance
        reasoning_templates = query_dict["reasoning_templates"] # [[{"existing_answer", "generated_text"}], [], ..., []]
        reasoning_trajs = []
        for reasoning_template in reasoning_templates:
            tamplate_content = reasoning_template[-1]["existing_answer"] + reasoning_template[-1]["generated_text"]
            extract_prompt = prompts._EXTRACT_REASONING_TRAJ_EN.format(
                reasoning_template=tamplate_content,
            ) if query_dict["lang"] == "en" else prompts._EXTRACT_REASONING_TRAJ_ZH.format(
                reasoning_template=tamplate_content,
            )
            # print("extract_prompt: ", extract_prompt)
            # for _ in range(API_MAX_RETRY):
            while True:
                try:
                    extract_response = self.client.chat.completions.create(
                        model=self.extract_model, # deepseek-v3
                        messages=[
                            {"role": "user", "content": extract_prompt}
                        ], 
                        temperature=temperature,
                        max_tokens=MAX_TOKEN[self.extract_model], 
                        stream=False, # True, 
                        **({"response_format": {"type": "json_object"}} if require_json else {})
                    )
                    extract_resp = extract_response.choices[0].message
                    if require_json:
                        reasoning_trajs.append(extract_resp.content) # {"score": int, "reason": str}
                        # return json.loads(extract_resp.content.strip("```json`")) # {"score": int, "reason": str}
                    else:
                        print("extract_resp: ", extract_resp)
                        reasoning_trajs.append(extract_resp)
                    break
                # except OpenAIError as e:
                #     print(f"Error {e.http_status}: {e}. Retrying in {retry_delay} seconds...")
                #     time.sleep(API_RETRY_SLEEP)
                # except requests.exceptions.RequestException as e:
                #     print(f"Request failed: {e}. Retrying in {retry_delay} seconds...")
                #     time.sleep(API_RETRY_SLEEP)
                except Exception as e:
                    print(f"Unexpected error: {e}. Retrying in {API_RETRY_SLEEP} seconds...")
                    time.sleep(API_RETRY_SLEEP)
                # return extract_resp
        # print("reasoning_trajs: ", reasoning_trajs)
        return {
            "reasoning_trajs": reasoning_trajs # List
        }
            
            
    def handle_file(self, file_path, dataset_name, output_dir): # for each partxxxx.jsonl
        os.makedirs(output_dir, exist_ok=True)
        jdict = utils_func.jload_reasoning_json(file_path) # return: List
        
        output_file_line_num = utils_func.count_lines_number(os.path.join(output_dir, os.path.basename(file_path)))
        print(f"{output_dir}: {output_file_line_num}")
        
        if output_file_line_num != None:
            with open(os.path.join(output_dir, os.path.basename(file_path)), "a", encoding="utf-8") as fout:
                for index, item in enumerate(tqdm(jdict, desc=f"Processing {dataset_name}", total=len(jdict))):
                    if index < output_file_line_num: # 断点重启
                        continue
                    response = self.extract_reaoning_traj(query_dict=item, require_json=True)
                    output_instance_dict = {**item, **response}
                    fout.write(ujson.dumps(output_instance_dict, ensure_ascii=False) + "\n")
            
            
if __name__ == "__main__":
    extract_model = "deepseek-v3"
    api_source = "bcloud"
    deepseek_instance = DeepSeekReasoningTraj(extract_model, api_source)
    
    raw_data_folder = "/global_data/data/medicalRL/code/data_construction/output_parallel/"
    output_root_folder = "/global_data/data/medicalRL/code/data_construction/output_reasoning_traj/"
    
    dataset_list = ["MedQA",  "MedMCQA", "RareArena"]
    # dataset_name = sys.argv[1] if len(sys.argv) > 1 else "MedQA"
    for dataset_name in dataset_list:
        output_folder = os.path.join(output_root_folder, dataset_name)
        os.makedirs(output_folder, exist_ok=True)
        
        args_list = []
        for dirpath, dirnames, filenames in os.walk(os.path.join(raw_data_folder, dataset_name)):
            for i, filename in enumerate(tqdm(filenames, total=len(filenames))):
                file_path = os.path.join(dirpath, filename)
                args_list.append((file_path, dataset_name, output_folder))
                
        with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLELISM) as executor:
            futures = [executor.submit(deepseek_instance.handle_file, item[0], item[1], item[2]) for item in args_list]
        results = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures))]