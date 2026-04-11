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
from .search_utils import search

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

class vLLMRolloutWithRetrieval(vLLMRollout): # reasoning + retrieval
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

        search_latency = 0.0
        search_fail = 0

        input_length_list = [len(x) for x in idx_list]

        search_bad_format_list = [0] * len(idx_list)
        search_count_list = [0] * len(idx_list)
        document_count_list = [0] * len(idx_list)
        response_length_list = [0] * len(idx_list)
        result_mask_list = [[] for _ in range(len(idx_list))]
        
        curr_index_list = list(range(len(idx_list)))

        search_debug_list = ["" for _ in range(len(idx_list))]

        with self.update_sampling_params(**kwargs):

            while len(curr_index_list) > 0:

                # print(f"################# {[response_length_list[i] for i in curr_index_list]=}")
                curr_max_tokens = self.config.response_length - min([response_length_list[i] for i in curr_index_list])

                with self.update_sampling_params(
                    n=1, 
                    stop=["</search>", "<document>", "</document>"],
                    detokenize=True, 
                    max_tokens=curr_max_tokens,
                    min_tokens=0, 
                    include_stop_str_in_output=True
                ):
                    outputs = self.inference_engine.generate(
                        prompts=None,  # because we have already convert it to prompt token id
                        sampling_params=self.sampling_params,
                        prompt_token_ids=[idx_list[i] for i in curr_index_list], # batch parallel
                        use_tqdm=True)
                        
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
                    idx_list[index] += curr_token_ids_list[i]
                    response_length_list[index] += len(curr_token_ids_list[i])
                    result_mask_list[index] += [1] * len(curr_token_ids_list[i])

                    if curr_stop_reason_list[i] == "<document>" or curr_stop_reason_list[i] == "</document>":
                        search_bad_format_list[index] = 1
                        continue
                    
                    # response too long
                    if response_length_list[index] > self.config.response_length:
                        continue

                    if curr_stop_reason_list[i] == "</search>":
                        # search too many times
                        if search_count_list[index] >= self.config.get("max_search_count", 4):
                            continue
                        
                        # check if the search format is correct
                        if curr_text_list[i].count("<search>") != 1:
                            search_bad_format_list[index] = 1
                            continue

                        search_query = curr_text_list[i][curr_text_list[i].rfind("<search>") + len("<search>"):-len("</search>")]
                        search_query = search_query.strip()
                        if len(search_query) == 0:
                            search_bad_format_list[index] = 1
                            continue

                        stt = time.time()
                        try:
                            search_result = search(self.config.search_url, search_query)
                        except Exception as e:
                            print(f"####### search failed: {e}")
                            search_result = None

                        edt = time.time()
                        search_latency += (edt - stt)
                        
                        search_count_list[index] += 1

                        if search_result:
                            if isinstance(search_result, (tuple, list)):
                                search_result, search_debug = search_result
                                search_debug_list[index] = json.dumps(search_debug, ensure_ascii=False)
                            document_count_list[index] += search_result.count("<document>")
                        else:
                            search_result = "<document>search failed, no document found</document>"
                            search_fail += 1

                        search_content_token_ids = self.tokenizer.encode(search_result)
                        idx_list[index] += search_content_token_ids
                        response_length_list[index] += len(search_content_token_ids)
                        result_mask_list[index] += [0] * len(search_content_token_ids)
                        
                        if response_length_list[index] < self.config.response_length:
                            next_index_list.append(index)

                curr_index_list = next_index_list
            
        search_count = sum(search_count_list)
        search_average_latency = (search_latency / (search_count + 1e-6)) if search_count > 0 else 0.0
        print(f"############## count={len(search_count_list)} {search_count=} {search_fail=} {search_latency=} {search_average_latency=}")

        response_list = []
        result_mask_list_padded = []
        for i, (output_ids, result_mask) in enumerate(zip(idx_list, result_mask_list)):
            output_ids = output_ids[input_length_list[i]:]
            assert len(output_ids) == len(result_mask), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"
            response = torch.tensor(output_ids, device=idx.device)
            # cut the response to the max length
            if len(response) > self.config.response_length:
                response = response[:self.config.response_length]
                result_mask = result_mask[:self.config.response_length]
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
        data.non_tensor_batch["search_bad_format"] = np.array(search_bad_format_list, dtype=object)
        data.non_tensor_batch["search_count"] = np.array(search_count_list, dtype=object)
        data.non_tensor_batch["document_count"] = np.array(document_count_list, dtype=object)
        data.non_tensor_batch["search_debug"] = np.array(search_debug_list, dtype=object)
        return data