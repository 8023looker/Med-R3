import ujson
import json
from typing import Optional, Sequence, Union, Dict
import io
import subprocess
import os


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload_json(f, dataset_name, mode="r"):
    f = _make_r_io_base(f, mode)
    
    jdict = []
    if dataset_name == "MedQA": 
        for line in f:
            cur_data_dict = json.loads(line)
            
            try:
                del cur_data_dict["model_try_count"]
                del cur_data_dict["model_answer"]
                del cur_data_dict["model_answer_verify"]
                del cur_data_dict["model_answer_correct"]
            except KeyError:
                print("KeyError: model_try_count, model_answer, model_answer_verify")
            jdict.append(cur_data_dict)
        
    elif dataset_name == "MedMCQA":
        for line in f:
            cur_data_dict = json.loads(line)
            try:
                del cur_data_dict["model_try_count"]
                del cur_data_dict["model_answer"]
                del cur_data_dict["model_answer_verify"]
                del cur_data_dict["model_answer_correct"]
            except KeyError:
                print("KeyError: model_try_count, model_answer, model_answer_verify")
            jdict.append(cur_data_dict)
    
    elif dataset_name == "RareArena":
        PROMPT_GEN_QUESTION = """
        As an expert in rare disease field, please provide the most likely diagnosis for the following patient. Only consider rare diseases.

        Here is the patient's condition: 
        {case_report}
        """
        data_dict_buffer = []
        for idx, line in enumerate(f):
            cur_data_dict = json.loads(line)
            cur_data_dict["question"] = PROMPT_GEN_QUESTION.format(case_report=cur_data_dict["case_report"])
            data_dict_buffer.append(cur_data_dict)
            if idx % 100 == 0:
                jdict.append(data_dict_buffer)
                data_dict_buffer = []
        if len(data_dict_buffer) > 0:
            jdict.append(data_dict_buffer)
        
    return jdict


def jload_reasoning_json(f, mode="r"):
    f = _make_r_io_base(f, mode)
    
    jdict = []
    for line in f:
        cur_data_dict = json.loads(line)
        jdict.append(cur_data_dict)
        
    return jdict


def extract_answer(json_line_data, dataset_name: str): # for each jsonl
    if dataset_name == "MedQA":
        try:
            instance_answer = json_line_data["options"][json_line_data["label"]]
        except KeyError:
            instance_answer = json_line_data["options"][0]
    elif dataset_name == "MedMCQA":
        pass
    elif dataset_name == "RareArena":
        pass
    return instance_answer


def judge_true_false_by_score(result):
    if isinstance(result, dict):
        if "score" in result:
            if isinstance(result["score"], int):
                score = result["score"]
                if score >= 5:
                    return True
    return False


def count_lines_number(file_path):
    if os.path.exists(file_path):
        try:
            result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, text=True, check=True)
            output = result.stdout.strip()
            line_count = int(output.split()[0])
            return line_count
        except subprocess.CalledProcessError as e:
            print(f"执行命令时发生错误: {e}")
            return None
        except FileNotFoundError:
            print(f"未找到 wc 命令或文件 {file_path} 未找到。")
            return None
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
            return None
    else:
        return 0