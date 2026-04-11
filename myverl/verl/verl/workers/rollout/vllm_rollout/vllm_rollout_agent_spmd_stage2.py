# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

# Stage 2: generate global plans
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
import numpy as np
import os
import json
import time
import requests
import copy

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version

from .vllm_rollout_spmd import vLLMRollout
from .search_utils import (
    batch_search, 
    search
)
from .agent_utils import (
    initialize_global_plans, 
    tool_simulation_stage2, 
    global_plan_evaluation
) # 待补充
from .agent_prompts import (
    _DYNAMIC_GLOBAL_PLAN_SELECTION_PROMPT, 
    _AGENT_STAGE2_GLOBAL_PLAN_GEN_PROMPT, 
    _DYNAMIC_GLOBAL_PLAN_SELECTION_STAGE2_PROMPT
)
from .agent_api import (
    oneapi_post_by_langchain, 
    read_json
)

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

class vLLMRolloutWithAgentStage2(vLLMRollout): # reasoning + retrieval
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids'] # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)

        batch_size = idx.size(0)

        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        if is_validate:
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1 # if validate, already repeat in ray_trainer
            }

        n = self.config.n if not is_validate else 1
        print(f"####### generate sequences {n=}")

        # expand to batch_size * n
        idx_list = []
        for i in range(batch_size):
            x = _pre_process_inputs(self.pad_token_id, idx[i])
            for _ in range(n):
                idx_list.append(x.copy())
        idx = idx.repeat_interleave(n, dim=0)
        # idx_list_ori = copy.deepcopy(idx_list)

        env_feedback_latency = 0.0

        input_length_list = [len(x) for x in idx_list]

        agent_bad_format_list = [0] * len(idx_list)
        feedback_count_list = [0] * len(idx_list)
        finish_reason_list = ["stop"] * len(idx_list)
        stop_reason_list = [None] * len(idx_list)
        response_length_list = [0] * len(idx_list)
        pure_response_list = [[] for _ in range(len(idx_list))] # List of List: each element is a list of token ids, which is the response of the agent
        result_mask_list = [[] for _ in range(len(idx_list))]
        
        curr_index_list = list(range(len(idx_list)))

        agent_debug_list = ["" for _ in range(len(idx_list))]
        
        curr_observation_list = ["" for _ in range(len(idx_list))]
        global_plans_chain_list = [[] for _ in range(len(idx_list))] # List: entire global plans, including the initial global plan and the updated global plans
        curr_global_plans_list = ["" for _ in range(len(idx_list))] # List: current global plans, used to generate the thought-action
        execution_step_index_list = [0] * len(idx_list)
        global_plan_score_list = [[] for _ in range(len(idx_list))] # List: the score of the global plans, used to select the best global plan
        dataset_name_list = ["" for _ in range(len(idx_list))]
        
        request_params = {
            "url": "http://qwen-14b-instruct-32k.bcloud.proxy.ml.baichuan-inc.com",
            "model": "",
            "key": "EMPTY", 
            "max_tokens": 4096,
            "temperature": 0.9, 
            "top_p": 1.0, 
            "max_concurrency": 8, 
            "base_model": None,
        }

        with self.update_sampling_params(**kwargs):

            while len(curr_index_list) > 0:

                # print(f"################# {[response_length_list[i] for i in curr_index_list]=}")
                curr_max_tokens = self.config.response_length - min([response_length_list[i] for i in curr_index_list])

                print(f"################## generate {len(curr_index_list)} sequences start")
                print(f"################## {curr_max_tokens=}")

                with self.update_sampling_params(
                    n=1, 
                    stop= ["</action>", "</global_plan>"], # ["</search>", "<document>", "</document>"],
                    detokenize=True, 
                    max_tokens=curr_max_tokens, 
                    min_tokens=0, 
                    include_stop_str_in_output=True
                ):
                    # initialize global plans (from trained models)
                    prompt_token_ids_list = []
                    for i in curr_index_list: # i is "index" in idx_list
                        curr_dataset_name = self.tokenizer.decode(idx_list[i], skip_special_tokens=True).split("</dataset>")[0].split("<dataset>")[-1].strip()
                        dataset_name_list[i] = curr_dataset_name # 其实不用每轮都重新算一遍...
                        curr_prompt_token_ids = idx_list[i] + self.tokenizer.encode(_AGENT_STAGE2_GLOBAL_PLAN_GEN_PROMPT[curr_dataset_name if curr_dataset_name in _AGENT_STAGE2_GLOBAL_PLAN_GEN_PROMPT else "alfworld"].format(
                            execution_step_index=str(execution_step_index_list[i]),
                            observation=curr_observation_list[i],
                            global_plan= curr_global_plans_list[i],
                        ))
                        prompt_token_ids_list.append(curr_prompt_token_ids)
                    # for debugging
                    print(f"prompt_token_ids_list: {len(prompt_token_ids_list)}, {prompt_token_ids_list[:5]}")
                    global_plans_outputs = self.inference_engine.generate(
                        prompts=None, # because we have already convert it to prompt token id
                        sampling_params=self.sampling_params,
                        prompt_token_ids=[prompt_token_ids_list[i] for i in curr_index_list], # batch parallel
                        use_tqdm=True)

                    for i, output in enumerate(global_plans_outputs):
                        global_plans_chain_list[curr_index_list[i]].append(output.outputs[0].text)
                        curr_global_plans_list[curr_index_list[i]] = output.outputs[0].text
                    
                    # thought-action based on the global plans (from trained models)
                    outputs = self.inference_engine.generate(
                        prompts=None, # because we have already convert it to prompt token id
                        sampling_params=self.sampling_params,
                        prompt_token_ids=[idx_list[i] + self.tokenizer.encode(_DYNAMIC_GLOBAL_PLAN_SELECTION_STAGE2_PROMPT[dataset_name_list[i] if dataset_name_list[i] in _DYNAMIC_GLOBAL_PLAN_SELECTION_STAGE2_PROMPT else "alfworld"].format(
                            observation=curr_observation_list[i],
                            global_plans=curr_global_plans_list[i],
                        )) for i in curr_index_list], # batch parallel
                        use_tqdm=True)
                        
                print(f"################## generate {len(curr_index_list)} sequences end")

                curr_text_list = []
                curr_token_ids_list = []
                curr_finish_reason_list = []
                curr_stop_reason_list = []
                for output in outputs:
                    curr_text_list.append(output.outputs[0].text)
                    curr_token_ids_list.append(output.outputs[0].token_ids)
                    curr_finish_reason_list.append(output.outputs[0].finish_reason)
                    curr_stop_reason_list.append(output.outputs[0].stop_reason)

                next_index_list = []

                for i, index in enumerate(curr_index_list):
                    pure_response_list[index] += curr_token_ids_list[i]
                    response_length_list[index] += len(curr_token_ids_list[i])
                    finish_reason_list[index] = curr_finish_reason_list[i]
                    stop_reason_list[index] = curr_stop_reason_list[i]
                    result_mask_list[index] += [1] * len(curr_token_ids_list[i])

                    # only <thought></thought> is not a valid stop reason 
                    if curr_stop_reason_list[i] in ["<thought>", "</thought>", "<global_plan>", "</global_plan>"]:
                        agent_bad_format_list[index] = 1
                        continue

                    if response_length_list[index] > self.config.response_length:
                        # response too long
                        finish_reason_list[index] = "length"
                        stop_reason_list[index] = None
                        continue

                    if curr_stop_reason_list[i] == "</action>":
                        
                        # check if the entire thought-action format is correct
                        if curr_text_list[i].count("<action>") != 1 or curr_text_list[i].count("<thought>") != 1 or curr_text_list[i].count("</thought>") != 1:
                            agent_bad_format_list[index] = 1
                            continue
                        
                        # extract the action part
                        agent_action = curr_text_list[i][curr_text_list[i].rfind("<action>") + len("<action>"):-len("</action>")]
                        agent_action = agent_action.strip()
                        if len(agent_action) == 0:
                            agent_bad_format_list[index] = 1
                            continue
                        
                        agent_action_dict = {}
                        stt = time.time()
                        try: # 调用 tool_simulation
                            # construct the agent action dict
                            curr_instruction = self.tokenizer.decode(idx_list[index], skip_special_tokens=True)[self.tokenizer.decode(idx_list[index], skip_special_tokens=True).rfind("<instruction>") + len("<instruction>"):-len("</instruction>")]
                            curr_instruction = curr_instruction.strip()
                            agent_action_dict.update({
                                "agent_action": agent_action, 
                                "instruction": curr_instruction, 
                                "previous_observation": curr_observation_list[index], 
                                "execution_step_index": str(execution_step_index_list[index])
                            })
                            env_feedback_result = tool_simulation_stage2(agent_action_dict, request_params) # 待修改
                            # search_result = search(self.config.search_url, agent_action)
                            execution_step_index_list[index] += 1 # update the execution step index
                        except Exception as e:
                            print(f"####### environment feedback failed: {e}")
                            env_feedback_result = ""

                        edt = time.time()
                        env_feedback_latency += (edt - stt)
                        
                        feedback_count_list[index] += 1
                        
                        # update the current observation
                        curr_observation_list[index] += (agent_action if agent_action is not None else "" + env_feedback_result if env_feedback_result is not None else "")
                        
                        env_feedback_content_token_ids = self.tokenizer.encode(env_feedback_result)
                        
                        response_length_list[index] += len(env_feedback_content_token_ids)
                        result_mask_list[index] += [0] * len(env_feedback_content_token_ids)
                        pure_response_list[index] += env_feedback_content_token_ids

                        if response_length_list[index] < self.config.response_length and env_feedback_result != "Task Completed!": # 没有达到 max_resp_length && 尚未能完成目标
                            next_index_list.append(index)
                            
                        # verify (score) the global plans
                        agent_action_dict.update({
                            "global_plan": curr_global_plans_list[index],
                            "env_feedback": env_feedback_result
                        })
                        global_plan_eval_result = global_plan_evaluation(agent_action_dict, request_params) # qwen2d5-14b-instruct-32k
                        global_plan_score_list[index].append(global_plan_eval_result) # dict

                curr_index_list = next_index_list
            
        feedback_count = sum(feedback_count_list)
        feedback_average_latency = (env_feedback_latency / (feedback_count + 1e-6)) if feedback_count > 0 else 0.0

        response_list = []
        result_mask_list_padded = []
        for i, (output_ids, result_mask) in enumerate(zip(pure_response_list, result_mask_list)):
            # output_ids = output_ids[input_length_list[i]:]
            assert len(output_ids) == len(result_mask), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"
            response = torch.tensor(output_ids, device=idx.device)
            # cut the response to the max length
            if len(response) > self.config.response_length:
                response = response[:self.config.response_length]
                result_mask = result_mask[:self.config.response_length]
                finish_reason_list[i] = "length"
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            result_mask = torch.tensor(result_mask, device=idx.device)
            result_mask = pad_sequence_to_length(result_mask, self.config.response_length, 0)
            response_list.append(response)
            result_mask_list_padded.append(result_mask)

        response = torch.stack(response_list, dim=0)
        result_mask = torch.stack(result_mask_list_padded, dim=0)
        assert response.size(0) == batch_size * n, f"response size: {response.size()}, batch_size: {batch_size}, n: {n}"
        assert response.size(1) == self.config.response_length, f"response size: {response.size()}, response_length: {self.config.response_length}"

        if n > 1:
            attention_mask = attention_mask.repeat_interleave(n, dim=0)
            position_ids = position_ids.repeat_interleave(n, dim=0)
            batch_size = batch_size * n

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        loss_mask = result_mask * response_attention_mask

        # print(f"############## {position_ids[0].tolist()[2048:3072]=} {position_ids.shape=}")
        # print(f"############## {attention_mask[0].tolist()[2048:3072]=} {attention_mask.shape=}")
        # print(f"############## {loss_mask[0].tolist()[:1024]=} {loss_mask.shape=}")

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx.to(torch.int),
                'responses': response.to(torch.int),
                'input_ids': seq.to(torch.int),  # here input_ids become the whole sentences
                'attention_mask': attention_mask.to(torch.int),
                'loss_mask': loss_mask.to(torch.int),
                'position_ids': position_ids.to(torch.int),
            },
            batch_size=batch_size)

        # free vllm cache engine
        # if self.config.free_cache_engine:
        #     self.inference_engine.free_cache_engine()
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        data =  DataProto(batch=batch)
        data.non_tensor_batch["finish_reason"] = np.array(finish_reason_list, dtype=object)
        data.non_tensor_batch["stop_reason"] = np.array(stop_reason_list, dtype=object)

        data.non_tensor_batch["search_bad_format"] = np.array(agent_bad_format_list, dtype=object)
        data.non_tensor_batch["search_count"] = np.array(feedback_count_list, dtype=object)
        data.non_tensor_batch["document_count"] = np.array(feedback_count_list, dtype=object)
        data.non_tensor_batch["search_debug"] = np.array(agent_debug_list, dtype=object) # 凑数的...
        # data.non_tensor_batch["global_plans_chain"] = np.array(global_plans_chain_list, dtype=object)
        global_plan_avg_score_list = handle_global_plan_score(global_plan_score_list) # [float, float, ...]
        print(f"Rollout stage global_plan_score_list: {len(global_plan_avg_score_list)}, {global_plan_avg_score_list}")
        data.non_tensor_batch["global_plan_score"] = np.array(global_plan_avg_score_list, dtype=object) # [[xxx, xxx, ..., xxx], [xxx, xxx, ...], ...]
        return data
    

# handle the global plan score and return the average score
def handle_global_plan_score(global_plan_score_list):
    """
    1. default: [[], [], ...]
    2. [[{}, {}, ...], [{}, {}, ...], ...]
    """
    if not global_plan_score_list or all(isinstance(item, list) and len(item) == 0 for item in global_plan_score_list):
        return ["0.0"] * len(global_plan_score_list) # return a list of 0.0 with the same length as global_plan_score_list
    else:
        avg_score_list = []
        for curr_global_plan_score_list in global_plan_score_list:
            curr_avg_score_list = []
            for item in curr_global_plan_score_list:
                correctness = float(item["correctness_score"]) if is_number(item["correctness_score"]) else 0.0
                followability = float(item["followability_score"]) if is_number(item["followability_score"]) else 0.0
                standardization = float(item["standardization_score"]) if is_number(item["standardization_score"]) else 0.0
                avg_score = (correctness + followability + standardization) / 3.0
                curr_avg_score_list.append(avg_score)
            curr_avg_score = sum(curr_avg_score_list) / len(curr_avg_score_list) if curr_avg_score_list and len(curr_avg_score_list) > 0 else 0.0
        avg_score_list.append(str(curr_avg_score))
        assert len(avg_score_list) == len(global_plan_score_list), f"avg_score_list: {len(avg_score_list)}, global_plan_score_list: {len(global_plan_score_list)}"
        return avg_score_list


def is_number(element):
    if isinstance(element, (int, float)):
        return True
    elif isinstance(element, str):
        try:
            float(element)
            return True
        except ValueError:
            return False
    else:
        return False