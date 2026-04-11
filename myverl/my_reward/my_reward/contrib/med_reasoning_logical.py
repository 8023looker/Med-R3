# combined version of MedMCQA, MedQA, RareArena_RDC

VERIFY_SYSTEM_PROMPT_COT_ZH = """## 角色
请你作为老师判断学生在解答问题时的思考过程是否符合思考模板

## 任务
请你基于 (1) 问题 (2) 学生在解答问题时的思考过程 (3) 解答问题时可供参考的思考模板
从结构相似性、语义相似性与逻辑合理性等方面判断学生的思考过程与思考模板是否符合，评为 0/1/2/3 分
3分标准:
思考过程虽然与思考模板有差异，但是其思考过程就逻辑合理性等方面而言更胜一筹
2分标准:
思考过程与思考模板的思路一致，允许在细节上有所差异
1分标准:
思考过程与思考模板大体一致，允许有一定的差异，但不能够包含错误信息
0分标准:
思考过程与思考模板不一致，或者包含逻辑矛盾以及错误信息，或者包含乱码、格式错误、乱序、多余无关信息

在模型的思考过程中，对于不确定的部分，它会使用 <|begin_of_query|> 搜索关键词 <|end_of_query|> 标签进行包裹，以便后续查询相关信息。当查询到相关内容后，模型将基于检索到的信息继续其推理过程。
需要注意的是，<|begin_of_documents|> 搜索结果 <|end_of_documents|> 中的内容是外部检索的结果，并非模型自己生成的。因此，这些内容不应被用于衡量模型本身的逻辑推理能力，而仅作为评估其处理和整合外部信息的能力的依据。
请在给出评分原因后，输出具体分数。

## 输入
{{
    "question": "问题",
    "reasoning_process": "学生思考过程",
    "reasoning_template": "思考模板"
}}

## 输出格式
请按照下面的json格式返回
```json
{{
    "reason": "评分原因",
    "score": 0/1/2/3
}}
```
""".strip()

_EXTRACT_REASONING_KG_ZH = """你是一名医学专家。
给定一段关于解决医学问题的思考过程，这段话包括 <think>...</think><answer>...</answer> 标签。在 <think> 部分中，作者展示了其思考过程，并可能使用 <|begin_of_query|> 搜索关键词 <|end_of_query|> 来标记需要查询的不确定知识。搜索系统会以 <|begin_of_documents|> 搜索结果 <|end_of_documents|> 提供相关检索信息。
你的任务是从该文本中结构化提取关键实体、关系及属性，构建知识图谱。需结合已有知识图谱的实体和关系描述（优先使用已有标准描述，仅在必要时补充新描述），确保描述统一性。

### 返回格式:
```json
[[
    "entity1", "relationship", "entity2", "if_retrieval"]
], ...]
```

定义：
1. entity1、relationship、entity2：需明确提取自自然语言中的实体或关系（如“患者”、“有症状”、“发热”）。
2. if_retrieval：布尔值（True/False），表示当前三元组的实体或关系是否通过检索（即来自 <|begin_of_documents|> 搜索结果 <|end_of_documents|> 的内容）

处理规则：
1. 总体：仅提取与医学问题解决相关的实体和关系，忽略与医学问题无关的背景信息，提取的知识图谱应简洁明了，避免冗余，同时应基于事实，避免主观推测
2. 实体提取：必须是具体医学概念（疾病、症状、药物、检查项目等），排除模糊描述（如"某些情况""相关因素"等），需合并同义表述（如"心梗"和"心肌梗死"统一为后者）。优先使用已有知识图谱的术语，若已有知识图谱未包含该实体，则保留原文术语并标记为新实体
3. 关系定义：使用动词短语（引发、抑制、导致、伴随等），必须明确方向性（A->B 或 B->A需符合医学逻辑），否定关系需显式标注（如"不引起""排除"）
4. 检索标记判定：若实体或关系直接来自 <|begin_of_documents|> 搜索结果 <|end_of_documents|> 的检索结果，则标记 if_retrieval=True，否则 False；检索关键词本身不作为实体或关系
5. 特殊处理：需保留推理过程中的假设关系（标记为[假设]），时间关系需转换为医学时间表述（急性/慢性/持续期等），概率性结论需量化为（高/中/低）风险等级
6. 已有知识图谱优先：实体和关系的描述需优先参考已有知识图谱的标准化定义，仅当已有描述无法覆盖时，才引入新描述（例如：已有知识图谱未定义“新型冠状病毒肺炎的后遗症”，则需根据文本内容新增实体“新冠后遗症”）
""".strip()

_EXTRACT_REASONING_KG_EN = """
You are a medical expert.
Given a reasoning process for solving a medical problem in the format of <think>...</think> and <answer>...</answer>. The content within <think>...</think> demonstrates the thought process and may use <|begin_of_query|> search query <|end_of_query|> to mark uncertain knowledge requiring searching. The search system provides relevant information in the format of <|begin_of_documents|> search results <|end_of_documents|>. 
Your task is to extract the important concepts, relationships, and attributes from the given medical reasoning process and represent them in the format of a knowledge graph. 
You should use existing knowledge graph entity and relationship descriptions (prioritizing standard descriptions, and only supplementing with new descriptions when necessary) to ensure consistency in the descriptions.

### Return Format:
```json
[[
    "entity1", "relationship", "entity2", "if_retrieval"]
], ...]
```

Definitions:
1. entity1, relationship, entity2: Clearly extract entities or relationships from the natural language (e.g., "patient", "has symptom", "fever").
2. if_retrieval: A boolean value (True/False) indicating whether the entity or relationship is retrieved from the search results within <|begin_of_documents|>...<|end_of_documents|>.

Processing rules:
1. Overall: Extract only entities and relationships related to solving medical problems, ignoring irrelevant background information. The knowledge graph should be concise and clear, avoiding redundancy, and based on facts, avoiding subjective speculation.
2. Entity extraction: Must be specific medical concepts (diseases, symptoms, drugs, examination items, etc.), excluding vague descriptions (such as "some conditions" or "related factors"). Synonymous expressions should be merged (e.g., "myocardial infarction" instead of "heart attack"). Prioritize the use of terminology from the existing knowledge graph. If the entity is not included in the existing knowledge graph, retain the original term and mark it as a new entity.
3. Relationship definition: Use verb phrases (cause, inhibit, lead to, accompany, etc.) and must have a clear directionality (A->B or B->A should conform to medical logic). Negative relationships should be explicitly marked (e.g., "does not cause", "rule out").
4. Retrieval marking determination: If an entity or relationship is directly from the <|begin_of_documents|> search results <|end_of_documents|>, mark "if_retrieval=True"; otherwise, "if_retrieval=False". The search keywords themselves should not be considered as entities or relationships.
5. Special handling: Retain hypothetical relationships in the reasoning process (marked as [hypothesis]), convert time relationships to medical temporal expressions (acute/chronic/ongoing period, etc.), and quantify probabilistic conclusions as (high/medium/low) risk levels.
6. Prioritize existing knowledge graph: The description of entities and relationships should first refer to the standardized definitions in the existing knowledge graph. Only when the existing descriptions are insufficient, new descriptions should be introduced (for example, if the existing knowledge graph does not define “post-COVID-19 conditions,” a new entity “COVID-19 sequelae” should be added based on the content of the text).
""".strip()

# reasoning prompts for all datasets
VERIFY_SYSTEM_PROMPT_COT_EN = """## Role
Please judge whether the student's reasoning process in answering the question conforms to the reasoning template.

## Task
Based on (1) the question (2) the student's reasoning process (3) the reference reasoning template provided for solving the problem.
Judge the alignment between the student's reasoning process and the reasoning template from aspects such as structural similarity, semantic similarity, and logical soundness, and give a score of 0/1/2/3.
3-point answer criteria:
The reasoning process, although different from the reasoning template, demonstrates superior logical soundness and reasoning.
2-point answer criteria:
The reasoning process aligns with the reasoning template, allowing for differences in details.
1-point answer criteria:
The reasoning process generally matches the reasoning template, permitting certain variations but without containing incorrect information.
0-point answer criteria:
The reasoning process does not align with the reasoning template, or it contains logical contradictions and erroneous information, or includes gibberish, format errors, disorder, extraneous irrelevant information.

During the model's reasoning process, uncertain parts are encapsulated using the <|begin_of_query|> search query <|end_of_query|> tags to facilitate subsequent information retrieval. Once relevant content is retrieved, the model continues its reasoning based on the retrieved information.
It is important to note that the content within <|begin_of_documents|> search results <|end_of_documents|> represents externally retrieved information and is not generated by the model itself. Therefore, this content should not be used to assess the model's logical reasoning ability. Instead, it should be used solely to evaluate the model's capability to process and integrate external information.
Please output the specific score after giving the scoring reason.

## Input
{{
    "question": "question",
    "reasoning_process": "student's reasoning process",
    "reasoning_template": "reasoning template"
}}

## Output Format
Please return in the json format below
```json
{{
    "reason": "scoring reason",
    "score": 0/1/2/3
}}
```
""".strip()

VERIFY_SYSTEM_PROMPT_ZH = """## 角色
请你作为老师判断学生的回答是否为问题的答案

## 任务
请你基于 (1) 问题 (2) 学生的回答 (3) 问题的正确答案 (4) 问题的误导选项，即一些错误答案的候选
判断学生的回答是与正确答案相符，评为 0/1/2 分
2分答案标准:
最终回答与正确答案意思完全一致，允许是正确答案的同义词、缩写等，但不能够包含错误选项
1分答案标准:
最终回答与正确答案意思一致，允许是正确答案的同义词、缩写、子集等，允许在这基础上有一定补充，但不能够包含错误选项
0分答案标准:
最终回答与正确答案不一致，或者包含错误选项，或者包含乱码、格式错误、乱序、多余无关信息
请在给出评分原因后，输出具体分数

## 输入
{{
    "question": "问题",
    "answer": "学生回答",
    "correct_answer": "正确答案",
    "misleading_options": ["误导选项1", "误导选项2", ...]
}}

## 输出格式
请按照下面的json格式返回
```json
{{
    "reason": "评分原因",
    "score": 0/1/2
}}
```
""".strip()

# answer prompt for MedMCQA, MedQA
VERIFY_SYSTEM_PROMPT_EN = """## Role
Please judge whether the student's answer is the answer to the question as a teacher

## Task
Based on (1) the question (2) the student's answer (3) the correct answer to the question (4) misleading options of the question, i.e., some wrong answer candidates
Judge the student's answer is consistent with the correct answer, and give a score of 0/1/2
2-point answer criteria:
The final answer is completely consistent with the correct answer, allowing synonyms, abbreviations, etc., but not including wrong options
1-point answer criteria:
The final answer is consistent with the correct answer, allowing synonyms, abbreviations, subsets, etc., and allowing some supplements on this basis, but not including wrong options
0-point answer criteria:
The final answer is inconsistent with the correct answer, or contains wrong options, or contains garbled characters, format errors, disorder, or irrelevant information
Please output the specific score after giving the scoring reason

## Input
{{
    "question": "question",
    "answer": "student's answer",
    "correct_answer": "correct answer",
    "misleading_options": ["misleading option 1", "misleading option 2", ...]
}}

## Output Format
Please return in the json format below
```json
{{
    "reason": "scoring reason",
    "score": 0/1/2
}}
```
""".strip()

# answer prompt for RareArena_RDC
VERIFY_RAREARENA_SYSTEM_PROMPT_EN = """
# Task

As a distinguished expert in the field of rare diseases, your task is to critically evaluate a student's diagnostic response based on a patient's clinical presentation. You will be provided with two key elements: the student's proposed diagnosis and the reference diagnosis.

Your objective is to:

1. Extract the Diagnostic Disease: Identify the disease proposed by the student from their response.
2. Assess Diagnostic Accuracy: Compare the extracted disease with the reference diagnosis to determine accuracy.
3. Utilize Scoring Rubric: Apply the following scoring rubric to evaluate the student's diagnosis.

# Scoring Criteria

- Score 2: The student's diagnosis must meet ALL the following criteria:
  - Contains only one disease.
  - Explicitly and precisely matches the name of the reference diagnosis.
  - Providing only a description of the disease, without explicitly stating its specific name, is insufficient to earn this score.
  - The response must directly include the exact disease name as given in the reference diagnosis.

- Score 1: The student's diagnosis must meet ALL the following criteria:
  - Contains only one disease.
  - Falls within a broader category that includes the reference diagnosis, indicating partial but not exact alignment.

- Score 0: The student's diagnosis must meet any of the following criteria:
  - Contains more than one disease, even though they contain the reference diagnosis.
  - Don't align with the reference diagnosis or its broader category, failing to meet the criteria for a score of 1 or 2. 

# Output Format

Please provide your evaluation in JSON format, including the following fields:

```json
{
  "extracted_diagnosis": ['disease_n', ..., 'disease_n'],
  "reason": "The rationale for your judgment, explaining why the extracted student's diagnosis received the given score. Be attention when the number of extracted diseases is more than one, the score should be 0.",
  "score": "The corresponding score based on the scoring criteria."
}
```
""".strip()

VERIFY_USER_PROMPT_KG_ZH = """## 思考段落:
{reasoning_process}
## 参考知识图谱:
{reference_kg}

## 请根据思考段落提取知识图谱，以json格式输出:
"""

VERIFY_USER_PROMPT_KG_EN = """## Reasoning Process:
{reasoning_process}
## Reference Knowledge Graph:
{reference_kg}

## Please extract the knowledge graph based on the reasoning process and output in the json format.
"""

VERIFY_USER_PROMPT_KG_LIST_EN = """## Reasoning Process:
{reasoning_process}
## Reference Knowledge Graphs:
{reference_kgs}

## Please extract the knowledge graph based on the reasoning process and output in the json format.
"""

VERIFY_USER_PROMPT_ZH = """## 问题:
{question}
## 学生回答:
{answer}
## 正确答案:
{correct_answer}
## 误导选项:
{misleading_options}

## 请给出评分原因后，输出具体分数，以json格式输出
""".strip()

VERIFY_USER_PROMPT_EN = """## Question:
{question}
## Student's Answer:
{answer}
## Correct Answer:
{correct_answer}
## Misleading Options:
{misleading_options}
""".strip()

VERIFY_RAREARENA_USER_PROMPT_EN = """
# Student's Answer:
{answer}

# Reference Diagnosis:
{correct_answer}
""".strip()

VERIFY_USER_PROMPT_COT_ZH = """## 问题:
{question}
## 学生思考过程:
{reasoning_process}
## 思考模板:
{reasoning_template}

## 请给出评分原因后，输出具体分数，以json格式输出
""".strip()

VERIFY_USER_PROMPT_COT_EN = """## Question:
{question}
## Student's Reasoning Process:
{reasoning_process}
## Reasoning Template:
{reasoning_template}

## Please output the specific score after giving the scoring reason in the json format.
""".strip()

import re
import time
import numpy as np
from tqdm import tqdm
import json
import copy
import requests
import traceback
from typing import Dict, List

from my_reward.api import (
    oneapi_post,
    oneapi_post_by_langchain,
    read_json
)
from my_reward.auxiliary.format_reward import (
    get_think_and_answer
)
from my_reward.contrib.base import RewardActorBase, Reason

import torch
import torch.nn as nn

def find_KG_common_path_1hop(G1, G2): # G1, G2: list
    mcs_score = 0.0
    # for string format
    # G1 = json.loads(G1_str)
    # G2 = json.loads(G2_str)
    set_G1 = {(item[0], item[2]) for item in G1}
    set_G2 = {(item[0], item[2]) for item in G2}
    common = set_G1 & set_G2
    print("common:", len(common), common)
    if len(common) == 0:
        return 0.0
    else:
        # print("common:", len(common), common)
        mcs_score = len(common) / (len(set_G1) + len(set_G2) - len(common))
        return mcs_score
    
def graph_logical_score(G1, G2): # G1, G2: list
    # construct graph dict
    def construct_graph_dict(G):
        graph_dict = {}
        for item in G: # [entity1, relation, entity2, if_retrieval]
            if item[0] not in graph_dict:
                graph_dict[item[0]] = []
            graph_dict[item[0]].append(item + [False]) # add if_visited
        return graph_dict
    G1_dict = construct_graph_dict(G1)
    G2_dict = construct_graph_dict(G2)
    
    def get_start_nodes(G_list):
        start_node_set = {item[0] for item in G_list} - {item[2] for item in G_list}
        for idx, item in enumerate(G_list):
            start_node = item[0]
            rest_entity_set = {item[0] for item in G_list[:idx] + G_list[idx:]} | {item[2] for item in G_list[:idx] + G_list[idx:]}
            if item[0] not in rest_entity_set and item[2] not in rest_entity_set: # isolated relationship
                start_node_set -= {item[0]}
        return start_node_set
    
    def find_next_node(G_dict, start_entity):
        if start_entity not in G_dict: # the current node is not the start node of any relationship
            return None, G_dict, None
        for idx, relation in enumerate(G_dict[start_entity]):
            if relation[-1] == False: # not visited
                G_dict[start_entity][idx][-1] = True # flag: visited
                return relation[2], G_dict, G_dict[start_entity][idx] # next entity, G_dict, relation
            else:
                continue
        return None, G_dict, None # no next node
    
    def construct_multihop_paths(G_dict, G_list):
        total_path_list = []
        start_node_set = get_start_nodes(G_list) # root nodes in the path
        for entity in start_node_set: # entity: str
            entity_path_dict = {
                "1-hop": [[path[0], path[1], path[2]] for path in G_dict[entity]] # [[[entity1, relation, entity2, if_retrieval, if_visited]], [[]], ...]
            }
            
            start_node_info = {}
            for idx, item in enumerate(G_dict[entity]):
                G_dict_item = copy.deepcopy(G_dict)
                G_dict_item[entity][idx][-1] = True # have been visited
                start_node_info[item[2]] = {
                    "prev_path": [item[0], item[1], item[2]], # [[entity1, relation, entity2], [], ...]
                    "G_dict": G_dict_item, # flagging the visited state
                }
            # current state: 1-hop
            terminate = False if len(start_node_info.keys()) > 0 else True
            cur_path_len = 1
            while not terminate: # start from 2-hop
                cur_path_len += 1
                next_start_node_info = {}
                entity_path_dict[f"{str(cur_path_len)}-hop"] = []
                for start_node in start_node_info: # dict
                    next_entity, G_dict_item, next_relation = find_next_node(start_node_info[start_node]["G_dict"], start_node)
                    if next_entity is not None: # found next node
                        next_start_node_info[next_entity] = { # update the next start node info
                            "prev_path": start_node_info[start_node]["prev_path"] + [next_relation[1], next_relation[2]], # add the new path (...prev_paths, next_relation, next_entity)
                            "G_dict": G_dict_item
                        }
                        entity_path_dict[f"{str(cur_path_len)}-hop"].append(next_start_node_info[next_entity]["prev_path"])
                    else:
                        continue
                # update info for iteration
                start_node_info = copy.deepcopy(next_start_node_info)
                terminate = False if len(start_node_info.keys()) > 0 else True
        
            # merge the paths
            for hop_key in entity_path_dict.keys():
                # print(entity_path_dict[hop_key])
                total_path_list += entity_path_dict[hop_key]
        
        total_path_list = set(tuple(path) for path in total_path_list) # 去重 (note: list is not hashable)
        # print(f"total_path_list: {total_path_list}")
        return total_path_list
    
    G1_paths = construct_multihop_paths(G1_dict, G1)
    G2_paths = construct_multihop_paths(G2_dict, G2)
    
    return len(G1_paths & G2_paths) / len(G1_paths | G2_paths) if len(G1_paths | G2_paths) > 0 else 0.0 # Jaccard similarity

def graph_semantic_score(G1_embedding, G2_embedding): # G1, G2: list
    # cosine similarity
    return float(np.dot(G1_embedding, G2_embedding) / (np.linalg.norm(G1_embedding) * np.linalg.norm(G2_embedding))) if (np.linalg.norm(G1_embedding) * np.linalg.norm(G2_embedding)) > 0 else 0.0

def graph_statistic_score(G1, G2): # G1, G2: list
    stats_score = 0.0
    # for string format
    # G1 = json.loads(G1_str)
    # G2 = json.loads(G2_str)
    set_G1 = {item[0] for item in G1} | {item[2] for item in G1} # entity
    set_G2 = {item[0] for item in G2} | {item[2] for item in G2} # entity
    common = set_G1 & set_G2
    union = set_G1 | set_G2
    print("common:", len(common), common)
    if len(common) == 0:
        return 0.0
    else:
        # print("common:", len(common), common)
        # stats_score = len(common) / (len(set_G1) + len(set_G2) - len(common))
        stats_score = len(common) / len(union) if len(union) > 0 else 0.0 # Jaccard similarity
        return stats_score

class RewardActorMedReasoningLogical(RewardActorBase): # combined version of MedMCQA, MedQA, RareArena_RDC
    
    @classmethod
    def normalize_reward(
        cls,
        reward: float,
    ):
        return 1.0 if reward >= 2.0 else (0.4 if reward >= 1.0 else cls.default)

    @classmethod
    def normalize_reasoning_reward(cls, G1: list, G1_embedding: list, G2: list, G2_embedding: list): # G1: {"kg_extraction", "reasoning_embedding"}
        alpha, beta, gamma = 1.0, 1.0, 1.0 # hyper-parameter
        
        score_logic, score_semantic, score_stats = 0.0, 0.0, 0.0
        try:
            score_logic = graph_logical_score(G1, G2)
        except Exception as e:
            traceback.print_stack()
            print(f"########### Error when calculating logical score: {e}")
            
        try:
            score_semantic = graph_semantic_score(G1_embedding, G2_embedding)
        except Exception as e:
            traceback.print_stack()
            print(f"########### Error when calculating semantic score: {e}")
            
        try:
            score_stats = graph_statistic_score(G1, G2)
        except Exception as e:
            traceback.print_stack()
            print(f"########### Error when calculating statistic score: {e}")
        
        reasoning_score = (alpha * score_logic + beta * score_semantic + gamma * score_stats) / (alpha + beta + gamma)
        return reasoning_score
    
        # try:
        #     # G1, G1_embedding = G1_dict["reasoning_KG"], G1_dict["reasoning_embedding"]
        #     score_logic = graph_logical_score(G1, G2)
        #     print("score_logic:", score_logic)
        #     score_semantic = graph_semantic_score(G1_embedding, G2_embedding)
        #     print("score_semantic:", score_semantic)
        #     score_stats = graph_statistic_score(G1, G2)
        #     print("score_stats:", score_stats)
        #     reasoning_score = (alpha * score_logic + beta * score_semantic + gamma * score_stats) / (alpha + beta + gamma)
        #     print("reasoning_score:", reasoning_score)
        #     return reasoning_score
        # except Exception as e:
        #     print(f"########### Error when normalizing score reasoning: {e}")
        #     return 0.0

    @classmethod
    def compute_score(
        cls, 
        params, 
        data_source, 
        prompt_str, 
        response_str, 
        ground_truth, 
        extra_info,
    ):
        # hyper-parameter
        # alpha, beta, gamma = 1.0, 1.0, 0.23 # make sure the whole score >= 0
        alpha_r, beta, gamma = 0.8, 1.0, 0.23

        format_score = cls.compute_format_score(prompt_str, response_str)
        if format_score == 0.0:
            result = {
                "reason": Reason.FORMAT_WRONG.value,
                "reward": cls.default,
            }
        else:
            reasoning_str, answer_str = get_think_and_answer(response_str)
            # answer
            answer_prompt = ""
            system_prompt = ""
            if data_source == "rarearena_rdc": # RareArena_RDC
                answer_prompt = VERIFY_RAREARENA_USER_PROMPT_EN.format(
                    answer=answer_str,
                    correct_answer=extra_info["diagonsis"],
                )
                system_prompt = VERIFY_RAREARENA_SYSTEM_PROMPT_EN
            elif data_source in ["medmcqa", "medqa"]: # MedQA, MedMCQA
                options = extra_info["options"]
                misleading_options = [v for k, v in options.items() if v != ground_truth]
                # answer
                answer_prompt = VERIFY_USER_PROMPT_EN.format(
                    question=extra_info["question"],
                    answer=answer_str,
                    correct_answer=ground_truth,
                    misleading_options=misleading_options
                )
                system_prompt = VERIFY_SYSTEM_PROMPT_EN
            else: # unexpected data source
                raise ValueError(f"Unsupported data source: {data_source}")

            # reasoning (the same prompt template for all datasets)
            reasoning_prompt_list, reference_kgs = [], ""
            # for _, reference_kg in enumerate(extra_info["reasoning_KG"]):
            #     reasoning_kg_prompt = VERIFY_USER_PROMPT_KG_EN.format(
            #         reasoning_process=reasoning_str,
            #         reference_kg=reference_kg
            #     )
            #     reasoning_prompt_list.append(reasoning_kg_prompt)
            reference_kgs = "\n".join(extra_info["reasoning_KG"])
            reasoning_kg_prompt = VERIFY_USER_PROMPT_KG_LIST_EN.format(
                reasoning_process=reasoning_str,
                reference_kgs=reference_kgs
            )
            # KG extraction
            G1 = []
            retries = cls.api_retries
            stt = time.time()
            while retries > 0:
                reasoning_response = oneapi_post(
                    prompt=_EXTRACT_REASONING_KG_EN + "\n" + reasoning_kg_prompt,
                    url=params["url"],
                    model=params["model"],
                    key=params.get("key", "EMPTY"),
                    max_tokens=params.get("max_tokens", 4096),
                    temperature=params.get("temperature", 0.9),
                    top_p=params.get("top_p", 0.6)
                )
                try:
                    G1 = read_json(reasoning_response, default=List)
                except Exception as e:
                    retries -= 1
                    continue
                break
            edt = time.time()

            # reasoning process embedding
            bgem3_embed_url = params.pop("bgem3_embed_url")
            retries = cls.api_retries
            while retries > 0:
                try:
                    reasoning_embedding = requests.post(  
                        bgem3_embed_url,
                        json={"text": reasoning_str},
                        headers={'Content-Type': 'application/json'},
                        timeout=10,
                    )
                    reasoning_embedding = reasoning_embedding.json()
                except Exception as e:
                    print(f"########### Error when getting reasoning embedding: {e}")
                    retries -= 1
                    continue
                break
            if retries == 0:
                result = {
                    "reward": cls.default,
                    "reason": Reason.API_ERROR.value,
                    "exception": "Failed to get reasoning embedding",
                }
            else:
                retries = cls.api_retries
                stt = time.time()
                while retries > 0:
                    answer_response = oneapi_post(
                        prompt=system_prompt + "\n" + answer_prompt,
                        url=params["url"],
                        model=params["model"],
                        key=params.get("key", "EMPTY"),
                        max_tokens=params.get("max_tokens", 4096),
                        temperature=params.get("temperature", 0.9),
                        top_p=params.get("top_p", 0.6)
                    )
                    result = cls.get_final_reward(answer_response)
                    if result.get("exception"):
                        retries -= 1
                        continue
                    break
                print(f"########### Time for verify answer response: {time.time() - stt}")
                if result.get("exception") is None:
                    reasoning_score_list = []
                    for i, reasoning_prompt in enumerate(extra_info["reasoning_KG"]):
                        try:
                            G2 = json.loads(extra_info["reasoning_KG"][i])
                            G1_embedding = reasoning_embedding
                            G2_embedding = extra_info["reasoning_vector"][i]
                            if isinstance(G2_embedding, np.ndarray):
                                G2_embedding = G2_embedding.tolist()         
                            
                            print(f"########### Time for verify reasoning response: {edt - stt}")
                            reasoning_score = cls.normalize_reasoning_reward(G1, G1_embedding, G2, G2_embedding)
                            reasoning_score_list.append(reasoning_score)
                        except Exception as e:
                            print(f"########### Error when calculating reasoning score: {e}")
                            continue
                    reasoning_kg_score = float(max(reasoning_score_list)) if reasoning_score_list else 0.0
                    result["reward"] = max(result["reward"] + alpha_r * reasoning_kg_score, 0.0)

        result["reward"] = cls.add_penalty(result["reward"], prompt_str, response_str, extra_info)
        return result