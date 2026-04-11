# Copyright 2024 PRIME team and/or its affiliates
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

import os
import json
import time
import traceback
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.utils.reward_score import diagnosis_verify
from .diagnosis import DiagnosisRewardManager

class DiagnosisWithSearchRewardManager(DiagnosisRewardManager):
    """The reward manager.
    """
    def _compute_score_task(self, args) -> float:
        """线程任务函数"""
        data_source, reward_actor, prompt_str, response_str, ground_truth, extra_info, search_bad_format = args
        if search_bad_format:
            return {
                "reason": "FORMAT_ERROR_SEARCH",
                "reward": -0.1,
                "acc": False,
            }
        elif reward_actor is not None:
            url = os.getenv("DIAGNOSIS_VERIFY_URL", None)
            if url is None:
                raise ValueError("DIAGNOSIS_VERIFY_URL is not set")
            return diagnosis_verify.compute_score(
                url=url,
                data_source=data_source,
                reward_actor=reward_actor,
                prompt_str=prompt_str,
                response_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                params=self.api_params,
            )
        else:
            return self.compute_score(
                data_source=data_source,
                tokenizer=self.tokenizer,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

    def __call__(self, data: DataProto, return_dict: bool = False, save_file_name: str = None):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']
            
        # 准备并发任务
        score_tasks = []
        task_metadata = []  # 存储每个任务的相关信息

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[:-len(eos_token)]

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']
            reward_actor = data_item.non_tensor_batch.get(self.reward_fn_key, None)

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            search_count = data_item.non_tensor_batch.get('search_count', 0)
            document_count = data_item.non_tensor_batch.get('document_count', 0)
            search_bad_format = data_item.non_tensor_batch.get('search_bad_format', False)

            search_debug = data_item.non_tensor_batch.get('search_debug', None)
            if search_debug:
                search_debug = json.loads(search_debug)

            # 提交任务到线程池
            task = self.thread_pool.submit(
                self._compute_score_task, 
                (data_source, reward_actor, prompt_str, response_str, ground_truth, extra_info, search_bad_format)
            )

            score_tasks.append(task)
            task_metadata.append({
                'index': i,
                'valid_response_length': valid_response_length.item(),
                'prompt_str': prompt_str,
                'response_str': response_str,
                'ground_truth': ground_truth,
                'data_source': data_source,
                'extra_info': extra_info,
                'reward_actor': reward_actor,
                'valid_prompt_length': valid_prompt_length.item(),
                'search_count': search_count,
                'document_count': document_count,
                'search_debug': search_debug,
                'search_bad_format': search_bad_format,
            })

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        
        # 获取所有任务的结果
        for task, meta in tqdm(zip(score_tasks, task_metadata)):
            result = task.result()  # 等待任务完成并获取结果
            data_source = meta['data_source']
            response_str = meta['response_str']
            ground_truth = meta['ground_truth']
            valid_response_length = meta['valid_response_length']
            meta['result'] = result
            i = meta['index']
            
            reason = ""
            is_error = False
            is_format_error = False
            # 为了后面能正确统计各项指标，保证不同类型数据 Verify 里必须包括 reward/score、acc
            if isinstance(result, dict):
                if "reward" in result:
                    score = result["reward"]
                elif "score" in result:
                    score = result["score"]
                else:
                    score = 0.0
                    result["exception"] = "score not in result"
                acc = result.get("acc", False)
                reason = result.get("reason", "")
                if result.get("exception"):
                    is_error = True
                if reason and reason.startswith("FORMAT_ERROR"):
                    is_format_error = True
            else:
                score = float(result)
                acc = False

            reward_extra_info["score"].append(score)
            reward_extra_info["acc"].append(float(acc))
            reward_extra_info["format_score"].append(0.0 if is_format_error else 1.0)
            # 保持与 custom 输出格式一致，即 score / acc / pred
            # TODO 非数学的类别取 pred 去算 major 得分的意义不是很大
            reward_extra_info["pred"].append("[INVALID]" if is_error else "")

            reward = score
            overlong_reward = 0.0
            if  self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            meta['score'] = score
            meta['acc'] = acc
            meta['is_error'] = is_error
            meta['is_format_error'] = is_format_error

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[data_source]", data_source)
                print("[reward_actor]", reward_actor)
                print("[prompt_str]", prompt_str)
                print("[response_str]", response_str)
                print("[ground_truth]", ground_truth)
                print("[valid_response_length]", valid_response_length)
                print("[score]", score)
                print("[reward]", reward)
                print("[result]", result)
                print("[search_count]", meta['search_count'])
                print("[document_count]", meta['document_count'])
                print("[search_debug]", meta['search_debug'])
                print("[search_bad_format]", meta['search_bad_format'])

        if self.save_path and save_file_name:
            with open(os.path.join(self.save_path, save_file_name), 'a') as f:
                for meta in task_metadata:
                    if meta.get('extra_info'):
                        extra_info = {k: v for k, v in meta['extra_info'].items() if k not in ["reasoning_templates", "reasoning_KG", "reasoning_vector"]}
                        for key, value in extra_info.items():
                            if isinstance(value, (torch.Tensor, np.ndarray)):
                                extra_info[key] = value.tolist()
                            elif isinstance(value, list):
                                if isinstance(value[0], (torch.Tensor, np.ndarray)):
                                    extra_info[key] = [v.tolist() for v in value]
                    f.write(json.dumps(meta, ensure_ascii=False) + '\n')

        print(f"############## {len(data)} samples with errors: {sum([meta['is_error'] for meta in task_metadata])} ##############")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
