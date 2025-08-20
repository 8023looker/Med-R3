""" Added by Keer Lu """
import re
import os
import sys
import json
from tqdm import tqdm
import pandas
import pyarrow.parquet as pq
import pyarrow as pa

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt import *

if __name__ == '__main__':

    data_source = "medqa"

    choose_path = ""
    # choose_path = "/global_data/data/liangzheng/reasoner/myverl/data/train/medqa/medqa_hard_case_lt0d5.txt"
    # path = "/global_data/data/liangzheng/cpt/sysdata/medqa/train_data_20250312_v2.jsonl"
    # output_path = f"/global_data/data/liangzheng/reasoner/myverl/data/train/medqa/{data_source}_20250304_v1_hard_case_lt0d5.parquet"
    path = "/global_data/data//medicalRL/code/data_construction/output/MedQA-2025-04-30.jsonl"
    output_path = f"/global_data/data//medicalRL/code/myverl/data/train_/MedQA/{data_source}-2025-04-30-multi.parquet"

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    choose_questions = []
    if os.path.exists(choose_path):
        for line in open(choose_path, "r", encoding="utf-8").readlines():
            line = line.strip()
            item = json.loads(line)
            choose_questions.append(item["question"])
    choose_questions = set(choose_questions)
    print(f"choose_questions: {len(choose_questions)}")
    
    new_data = []
    for line in tqdm(open(path, "r", encoding="utf-8").readlines()):
        line = line.strip()
        item = json.loads(line)
        if choose_questions and (item["question"] not in choose_questions and item["reformat_question"] not in choose_questions):
            continue

        new_item = {
            "data_source": data_source,
            "reward_actor": "RewardActorMedReasoningLogical",
            # "prompt": [
            #     {
            #         "role": "system",
            #         "content": R1_INSTRUCT_SYSTEM_PROMPT
            #     },
            #     {
            #         "role": "user",
            #         "content": R1_INSTRUCT_USER_PROMPT.format(prompt=item["reformat_question"])
            #     }
            # ],
            "prompt": R1_ORIGIN_PROMPT_ADD_LANGUAGE_AND_SEARCH.format(prompt=item["reformat_question"]),
            "reward_model": {
                "style": "rule",
                "ground_truth": item["options"][item["label"]],
            },
            "extra_info": {
                "original_question": item["question"],
                "question": item["reformat_question"], # 必须存在，用来 verify
                "options": item["options"],
                # "model_answer_correct": item["model_answer_correct"],
                # "model_try_count": item["model_try_count"],
                "index": len(new_data),
                "reasoning_templates": item["reasoning_templates"], # list format
                "reasoning_KG": item["reasoning_KG"], # list format
                "reasoning_vector": item["reasoning_vector"], # list format
            }
        }
        new_data.append(new_item)
    
    # to parquet
    df = pandas.DataFrame(new_data)
    print(df)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)
