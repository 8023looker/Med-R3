""" generate reasoning chains based on deepseek-r1 model """
""" retrieval-augmented reasoning """
""" parallel processing """

import os
import json
import ujson
import openai
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
API_RETRY_SLEEP = 1
API_ERROR_OUTPUT = "$ERROR$"
PARALLELISM = 95 # 100

MAX_ATTEMPT = 10

MAX_TOKEN = {
    "deepseek-r1": 7400,
    "deepseek-v3": 8192,
}


class DeepSeekThoughtTemplate: # retrieval-augmented reasoning
    def __init__(self, infer_model_name, verify_model_name, api_source, hit_size=20, timeout=2000):
        self.api_source = api_source
        if self.api_source == "xxx":
            self.client = openai.OpenAI( # svip
                base_url="query_url",
                api_key="sk-xxx")
        elif self.api_source == "xxx":
            self.client = openai.OpenAI( # bcloud
                base_url="query_url",
                api_key="sk-xxx")
        self.infer_model = infer_model_name
        self.verify_model = verify_model_name
        
        # query
        self.url = "http://xxx"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': 'xxx',
        }
        
        
    def verify_correctness(self, reasoning_content, query_dict, temperature=0.9, require_json=True): # verify the correctness of the generated answer (output: score[int])
        # verification based on deepseek-v3
        print("------ Start to verify ------")
        print("reasoning_content", reasoning_content)
        verify_prompt = prompts._VERIFY_PROMPT_EN.format(
            question=query_dict["question"],
            answer=reasoning_content,
            correct_answer=query_dict["options"][query_dict["label"]],
            misleading_options="\n".join([f"[{j+1}]: {v}" for j, (k, v) in enumerate(query_dict["options"].items()) if k != query_dict["label"]])
        ) if query_dict["language"] == "en" else prompts._VERIFY_PROMPT_ZH.format(
            question=query_dict["question"],
            answer=reasoning_content,
            correct_answer=query_dict["options"][query_dict["label"]],
            misleading_options="\n".join([f"[{j+1}]: {v}" for j, (k, v) in enumerate(query_dict["options"].items()) if k != query_dict["label"]])
        )
        
        verify_response = self.client.chat.completions.create(
            model=self.verify_model, # deepseek-v3
            messages=[
                {"role": "user", "content": verify_prompt}
            ], 
            temperature=temperature,
            max_tokens=MAX_TOKEN[self.verify_model], 
            stream=False, # True, 
            **({"response_format": {"type": "json_object"}} if require_json else {})
        )
        verify_resp = verify_response.choices[0].message
        if require_json:
            try:
                return json.loads(verify_resp.content.strip("```json`")) # {"score": int, "reason": str}
            except json.JSONDecodeError:
                return {"score": 0, "reason": "JSONDecodeError"}
        else:
            return verify_resp
        
        
    def get_reasoning_chains(self, query_dict, total_template_num=10, temperature=0.9, require_json=False): # for each instance
        # output_instance_dict = {**query_dict, "reasoning_chain": []}
        # generate reasoning chains
        reasoning_chain_list, model_answer_verify_list = [], []
        gen_template_num = 0
        while gen_template_num <= total_template_num: # generate several reasoning chains (10)
            cur_reasoning_chain = self.gen_infer_response(query_dict=query_dict, require_json=False) # for each chain of the template [List]
            final_reasoning_chain_text = cur_reasoning_chain[-1]["existing_answer"] + cur_reasoning_chain[-1]["generated_text"] # the last response
            verify_content = self.verify_correctness(final_reasoning_chain_text, query_dict)
            print("VERIFY_CONTENT", verify_content)
            
            if utils_func.judge_true_false_by_score(verify_content): # valid response
            # if float(verify_content["score"]) >= 3: # 3, 4, 5
                print(f"{gen_template_num}------PASS!")
                reasoning_chain_list.append(cur_reasoning_chain) # [[], [], ..., []]
                model_answer_verify_list.append(verify_content) # {"score": int, "reason": str}
                gen_template_num += 1
                
        return {
            "reasoning_templates": reasoning_chain_list, 
            "template_verify": model_answer_verify_list
        }
      
    
    def gen_infer_response(self, query_dict, temperature=0.9, require_json=False): # think → search → think ...
        # initialization
        cur_max_tokens = MAX_TOKEN[self.infer_model]
        cur_prompt = prompts._ANSWER_WITH_THINK_PROMPT_EN.format(**query_dict) if query_dict["language"] == "en" else (prompts._ANSWER_WITH_THINK_PROMPT_ZH.format(**query_dict) if query_dict["language"] == "zh" else prompts._ANSWER_WITH_THINK_PROMPT_ZH_HK.format(**query_dict))
        # cur_reasoning_chain: [{"existing_answer": "", "existing_answer_no_docs": "", "query_detail": {"query": "", "docs": ""}}]
        cur_reasoning_chain = [] # note: storage format is different from the processing one
        existing_answer = ""
        
        # for _ in range(API_MAX_RETRY):
        break_outer_loop = False
        while cur_max_tokens > 0:
            try:
                response = self.client.chat.completions.create(
                    model=self.infer_model, # deepseek-r1
                    messages=[
                        {"role": "user", "content": cur_prompt}
                    ], 
                    temperature=temperature, 
                    stop=['<|end_of_query|>'], # ['</think>'] 
                    max_tokens=cur_max_tokens, 
                    stream=False, # True, 
                    **({"response_format": {"type": "json_object"}} if require_json else {})
                )
                
                # print("response", response) # debugging
                
                for choice in response.choices:
                    finish_reason = choice.finish_reason
                    completion_tokens = response.usage.completion_tokens
                    # Check if the generated text ends with the stop token
                    print("generated_text", choice.message.content)
                    generated_text = choice.message.content
                    if finish_reason == "stop":
                        if generated_text.endswith('</answer>'): # natural completion
                            print("Model has naturally completed the generation.")
                            cur_reasoning_chain.append({
                                "existing_answer": existing_answer,
                                "generated_text": generated_text,
                            })
                            break_outer_loop = True
                            break
                        elif len(generated_text.split("<search>")[-1].split("</search>")[0]) == len(generated_text.split("<search>")[-1]):
                            print("Generation stopped because it encountered the stop sequence </search>.")
                            print("completion_tokens", completion_tokens)

                            # handle generated response && retrieve docs
                            query_docs = self.doc_retrieval(generated_text) if "<search>" in generated_text else ""
                            existing_answer = generated_text + "</search>" + query_docs # update existing_answer
                            # update cur_prompt
                            updated_query_dict = {
                                "question": query_dict["question"],
                                "existing_answer": existing_answer,
                                "language": query_dict["language"]
                            }
                            cur_prompt = prompts._ANSWER_WITH_THINK_PROMPT_EN_RESTART.format(**updated_query_dict) if updated_query_dict["language"] == "en" else (prompts._ANSWER_WITH_THINK_PROMPT_ZH_RESTART.format(**updated_query_dict) if updated_query_dict["language"] == "zh" else prompts._ANSWER_WITH_THINK_PROMPT_ZH_HK_RESTART.format(**updated_query_dict))
                            # update cur_max_tokens
                            cur_max_tokens -= completion_tokens # 截断机制 1
                            
                            # update cur_reasoning_chain
                            cur_reasoning_chain.append({
                                "existing_answer": existing_answer,
                                "generated_text": generated_text,
                            })
                                    
                        else:
                            print("Model has naturally completed the generation, but with unknown reason.")
                            cur_reasoning_chain.append({
                                "existing_answer": existing_answer,
                                "generated_text": generated_text,
                            })
                            break_outer_loop = True
                            break
                    elif finish_reason == "length": # max_token_limit
                        print("Generation stopped because it reached the maximum token limit.")
                        cur_reasoning_chain.append({
                            "existing_answer": existing_answer,
                            "generated_text": generated_text,
                        })
                        break_outer_loop = True
                        break
                    else:
                        print("Model has naturally completed the generation due to other reasons.")
                        cur_reasoning_chain.append({
                            "existing_answer": existing_answer,
                            "generated_text": generated_text,
                        })
                        break_outer_loop = True
                        break
                
            except Exception as e:
                time.sleep(API_RETRY_SLEEP)
                
            if break_outer_loop:
                break
        
        return cur_reasoning_chain
        
                 
    def doc_retrieval(self, existing_answer):
        # extract the query from the existing answer
        query_text = existing_answer.split("<search>")[-1].split("</search>")[0].strip() # the last query
        query_data = {
            "request_id": "query",
                "query": query_text,
                "control": {
                    "hit_size": 20,
                    "timeout": 2000,
                # "debug_level": 1
                }
        }
        response = self.query(query_data)
        
        if "result" in response:
            result_list = response["result"]["hits"] if "hits" in response["result"] else [] # List
            # extract the top 3 documents
            top_3_docs = [result["content"] for result in result_list[:3]] if len(result_list) >= 3 else result_list
            top_3_docs_string = "<document>" + "\n\n".join(top_3_docs) + "</document>"
            return top_3_docs_string
    
    
    def query(self, query_data):
        print("query_data", query_data)
        response = requests.post(self.url, headers=self.headers, data=json.dumps(query_data))
        # 检查响应状态码
        if response.status_code == 200:
            print(f"Request successful!")
            # print("Response:", response.json())  # 响应是 JSON 格式
            return response.json()
        else:
            print(f"Request failed with status code: {response.status_code}")
            # print("Response:", response.text)
            return response.text
        
                
    def handle_file(self, file_path, dataset_name, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        jdict = utils_func.jload_json(file_path, dataset_name) # return: List
        
        output_file_line_num = utils_func.count_lines_number(os.path.join(output_dir, os.path.basename(file_path)))
        print(f"{os.path.join(output_dir, os.path.basename(file_path))}: {output_file_line_num}")
        
        if output_file_line_num != None:
            with open(os.path.join(output_dir, os.path.basename(file_path)), "a", encoding="utf-8") as fout:
                # for item in tqdm(jdict, desc=f"Processing {dataset_name}", total=len(jdict)):
                for index, item in enumerate(tqdm(jdict, desc=f"Processing {dataset_name}", total=len(jdict))):
                    if index < output_file_line_num: # 断点重启
                        continue
                    query_dict = {
                        "question": item["reformat_question"], 
                        "language": item["lang"],
                        "options": item["options"],
                        "label": item["label"],
                    }
                    # prompt_gen_resp = prompts._ANSWER_WITH_THINK_PROMPT_EN.format(**query_dict) if item["lang"] == "en" else (prompts._ANSWER_WITH_THINK_PROMPT_ZH.format(**query_dict) if item["lang"] == "zh" else prompts._ANSWER_WITH_THINK_PROMPT_ZH_HK.format(**query_dict))
                    # print(f"Prompt: {prompt_gen_resp}")
                    response = self.get_reasoning_chains(query_dict=query_dict, require_json=False)
                    
                    output_instance_dict = {**item, **response}
                    fout.write(ujson.dumps(output_instance_dict, ensure_ascii=False) + "\n")         


if __name__ == "__main__":
    infer_model = 'deepseek-r1'
    verify_model = "deepseek-v3"
    api_source = "xxx"
    deepseek_instance = DeepSeekThoughtTemplate(infer_model, verify_model, api_source)
    
    raw_data_folder = "/global_data/data/medical_data/"
    output_root_folder = "/global_data/data/medicalRL/code/data_construction/output_parallel/"
    dataset_list = ["MedQA", "MedMCQA", "RareArena"]
    args_list = []
    for dataset_name in dataset_list:
        output_folder = os.path.join(output_root_folder, dataset_name)
        os.makedirs(output_folder, exist_ok=True)
        for dirpath, dirnames, filenames in os.walk(raw_data_folder):
            for i, filename in enumerate(tqdm(filenames, total=len(filenames))):
                file_path = os.path.join(dirpath, filename)
                args_list.append((file_path, dataset_name, output_folder))
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLELISM) as executor:
        futures = [executor.submit(deepseek_instance.handle_file, item[0], item[1], item[2]) for item in args_list]
        results = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures))]