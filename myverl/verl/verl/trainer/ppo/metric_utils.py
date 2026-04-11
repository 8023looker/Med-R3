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
Metrics related to the PPO trainer.
"""

import torch
from typing import Any, Dict, List, Callable
import numpy as np
from verl import DataProto
from collections import Counter, defaultdict
from functools import partial


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def extra_metrics(batch: DataProto, response_length: torch.Tensor) -> Dict[str, Any]:
    extra_metrics = {}
    if "acc" in batch.non_tensor_batch:
        correct_mask = torch.BoolTensor(batch.non_tensor_batch["acc"], device=response_length.device)
        wrong_mask = ~correct_mask
        invalid_ratio = np.count_nonzero(batch.non_tensor_batch["pred"] == "[INVALID]") / len(batch.non_tensor_batch["acc"])
        extra_metrics.update({
            # 统计训练中用到样本的准确率
            'critic/acc/mean(used)':
                torch.mean(correct_mask.float()).detach().item(),
            'critic/acc/std(used)':
                torch.std(correct_mask.float()).detach().item(),
            # 格式错误样本统计
            'critic/invalid_ratio':
                invalid_ratio,
            # 统计所有rollout样本的准确率
            **batch.acc_report(),
            # 正负样本长度统计
            'response_length/correct/mean':
                torch.mean(response_length*correct_mask).detach().item(),
            'response_length/correct/median':
                torch.median(response_length*correct_mask).detach().item(),
            'response_length/wrong/mean':
                torch.mean(response_length*wrong_mask).detach().item(),
            'response_length/wrong/median':
                torch.median(response_length*wrong_mask).detach().item(),
        })
    return extra_metrics

def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        'response_length/median':
            torch.median(response_length).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        'prompt_length/median':
            torch.median(prompt_length).detach().item(),

        **extra_metrics(batch, response_length),
    }
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info['global_token_num'])
    time = timing_raw['step']
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        'perf/total_num_tokens': total_num_tokens,
        'perf/time_per_step': time,
        'perf/throughput': total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(data: list[Any],
                     subset_size: int,
                     reduce_fns: list[Callable[[np.ndarray], float]],
                     n_bootstrap: int = 1000,
                     seed: int = 42) -> list[tuple[float, float]]:
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(data_sources: list[str],
                               sample_inputs: list[str],
                               infos_dict: dict[str, list[Any]],
                               seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """Process validation metrics into a structured format.
    
    Args:
        data_sources: Array of data source identifiers for each sample
        sample_inputs: List of input prompts
        infos_dict: variable name -> list of values for each sample
        
    Returns:
        dict[str, dict[str, dict[str, float]]]: data source -> variable name -> metric value
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue
                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)
                metric[f"std@{n_resps}"] = np.std(var_vals)

                # ns = []
                # n = 2
                # while n < n_resps:
                #     ns.append(n)
                #     n *= 2
                # ns.append(n_resps)

                # for n in ns:
                #     import time
                #     stt = time.time()
                #     # Best/Worst-of-N
                #     [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals,
                #                                                                   subset_size=n,
                #                                                                   reduce_fns=[np.max, np.min],
                #                                                                   seed=seed)
                #     metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                #     metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                #     # Majority voting
                #     if var2vals.get("pred", None) is not None:
                #         vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                #         [(maj_n_mean, maj_n_std)
                #         ] = bootstrap_metric(data=vote_data,
                #                              subset_size=n,
                #                              reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                #                              seed=seed)
                #         metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val

def compute_data_metrics_by_data_source(batch: DataProto) -> Dict[str, Any]:

    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    data_sources = batch.non_tensor_batch['data_source']

    metrics = {}

    for data_source in set(data_sources):
        mask = torch.tensor([data_source == ds for ds in data_sources], dtype=torch.bool)
        data_source_score = torch.masked_select(sequence_score, mask)
        data_source_reward = torch.masked_select(sequence_reward, mask)
        data_source_response_length = torch.masked_select(response_length, mask)
        metrics.update({
            f'critic/score/{data_source}/mean': torch.mean(data_source_score).detach().item(),
            f'critic/rewards/{data_source}/mean': torch.mean(data_source_reward).detach().item(),
            f'response_length/{data_source}/mean': torch.mean(data_source_response_length).detach().item(),
            f'response_length/{data_source}/median': torch.median(data_source_response_length).detach().item(),
        })
    
    return metrics

def calculate_ngram_overlap(response_token_list, ngram_num=3):
    response_ngram_list = []
    for response_token in response_token_list:
        response_ngram = []
        for i in range(len(response_token) - ngram_num + 1):
            response_ngram.append(tuple(response_token[i:i+ngram_num]))
        response_ngram_list.append(response_ngram)
    ngram_overlap_list = []
    for i in range(len(response_ngram_list)):
        for j in range(i+1, len(response_ngram_list)):
            ngram_overlap = len(set(response_ngram_list[i]) & set(response_ngram_list[j])) * 1.0 / len(set(response_ngram_list[i]) | set(response_ngram_list[j]))
            ngram_overlap_list.append(ngram_overlap)
    return sum(ngram_overlap_list) * 1.0 / len(ngram_overlap_list)

def compute_experience_metrics(batch: DataProto, tokenizer, step) -> Dict[str, Any]:

    # batched scoring
    prompt_ids = batch.batch['prompts']
    prompt_length = prompt_ids.shape[-1]
    valid_prompt_length = batch.batch['attention_mask'][:,:prompt_length].sum(dim=-1)

    response_ids = batch.batch['responses']
    valid_response_length = batch.batch['attention_mask'][:,prompt_length:].sum(dim=-1)

    accs = batch.non_tensor_batch.get('acc')

    # data_source
    data_sources = batch.non_tensor_batch.get('data_source')
    prompt_str_to_data_source = {}
    count_by_data_source = defaultdict(int)
    # response token
    response_token_by_prompt_str = defaultdict(list)
    # response acc
    response_acc_by_prompt_str = defaultdict(list)
    # response length when correct
    response_length_when_correct = []
    # response length when wrong
    response_length_when_wrong = []

    for i in range(len(batch)):
        valid_prompt_ids = prompt_ids[i][-valid_prompt_length[i]:]
        valid_response_ids = response_ids[i][:valid_response_length[i]]
        prompt_str = tokenizer.decode(valid_prompt_ids)
        data_source = data_sources[i]
        count_by_data_source[data_source] += 1
        prompt_str_to_data_source[prompt_str] = data_source
        response_token_by_prompt_str[prompt_str].append(valid_response_ids.tolist())
        if accs is not None and float(accs[i]) == 1.0:
            response_acc_by_prompt_str[prompt_str].append(1.0)
            response_length_when_correct.append(float(valid_response_length[i].item()))
        else:
            response_acc_by_prompt_str[prompt_str].append(0.0)
            response_length_when_wrong.append(float(valid_response_length[i].item()))

    # response pass n
    response_pass_n_by_prompt_str = {k: 1.0 if max(v) == 1.0 else 0.0 for k, v in response_acc_by_prompt_str.items()}
    # response accuracy
    response_accuracy_by_prompt_str = {k: np.mean([1.0 if x == 1.0 else 0.0 for x in v]) for k, v in response_acc_by_prompt_str.items()}
    # response zero adv correct
    response_zero_adv_correct_by_prompt_str = {k: 1.0 if all([True if x == 1.0 else False for x in v]) else 0.0 for k, v in response_acc_by_prompt_str.items()}
    # response zero adv wrong
    response_zero_adv_wrong_by_prompt_str = {k: 1.0 if all([True if x == 0.0 else False for x in v]) else 0.0 for k, v in response_acc_by_prompt_str.items()}
    
    result = {}

    for data_source, count in count_by_data_source.items():
        result[f'count/{data_source}'] = count

    response_pass_n_by_data_source = defaultdict(list)
    for prompt_str, v in response_pass_n_by_prompt_str.items():
        if prompt_str not in prompt_str_to_data_source:
            continue
        data_source = prompt_str_to_data_source[prompt_str]
        response_pass_n_by_data_source[data_source].append(v)
    for data_source, v in response_pass_n_by_data_source.items():
        result[f'response/{data_source}/pass_n'] = np.mean(v)
    result['response/pass_n'] = np.mean(list(sum(response_pass_n_by_data_source.values(), [])))

    response_accuracy_by_data_source = defaultdict(list)
    for prompt_str, v in response_accuracy_by_prompt_str.items():
        if prompt_str not in prompt_str_to_data_source:
            continue
        data_source = prompt_str_to_data_source[prompt_str]
        response_accuracy_by_data_source[data_source].append(v)
    for data_source, v in response_accuracy_by_data_source.items():
        result[f'response/{data_source}/accuracy'] = np.mean(v)
    result['response/accuracy'] = np.mean(list(sum(response_accuracy_by_data_source.values(), [])))

    response_zero_adv_correct_by_data_source = defaultdict(list)
    for prompt_str, v in response_zero_adv_correct_by_prompt_str.items():
        if prompt_str not in prompt_str_to_data_source:
            continue
        data_source = prompt_str_to_data_source[prompt_str]
        response_zero_adv_correct_by_data_source[data_source].append(v)
    for data_source, v in response_zero_adv_correct_by_data_source.items():
        result[f'response/{data_source}/correct/zero_adv'] = np.mean(v)
    result['response/correct/zero_adv'] = np.mean(list(sum(response_zero_adv_correct_by_data_source.values(), [])))

    response_zero_adv_wrong_by_data_source = defaultdict(list)
    for prompt_str, v in response_zero_adv_wrong_by_prompt_str.items():
        if prompt_str not in prompt_str_to_data_source:
            continue
        data_source = prompt_str_to_data_source[prompt_str]
        response_zero_adv_wrong_by_data_source[data_source].append(v)
    for data_source, v in response_zero_adv_wrong_by_data_source.items():
        result[f'response/{data_source}/wrong/zero_adv'] = np.mean(v)
    result['response/wrong/zero_adv'] = np.mean(list(sum(response_zero_adv_wrong_by_data_source.values(), [])))    
            
    # TODO 如果速度较慢可以考虑注释掉
    # if step == 0 or step % 8 == 0:
    #     response_ngram_overlap_by_data_source = defaultdict(list)
    #     for prompt_str, response_token_list in response_token_by_prompt_str.items():
    #         if prompt_str not in prompt_str_to_data_source:
    #             continue
    #         data_source = prompt_str_to_data_source[prompt_str]
    #         ngram_overlap_ratio = calculate_ngram_overlap(response_token_list)
    #         response_ngram_overlap_by_data_source[data_source].append(ngram_overlap_ratio)

    #     for data_source, v in response_ngram_overlap_by_data_source.items():
    #         result[f'response/{data_source}/ngram_overlap_score'] = np.mean(v)
    #     result['response/ngram_overlap_score'] = np.mean(list(sum(response_ngram_overlap_by_data_source.values(), [])))
    
    return result

def compute_search_metrics(batch: DataProto) -> Dict[str, Any]:

    data_sources = batch.non_tensor_batch['data_source']

    metrics = {}

    if 'loss_mask' in batch.batch:
        loss_mask = batch.batch['loss_mask']
        loss_length = loss_mask.sum(dim=-1)
        loss_length = loss_length.float()
        metrics["response_length/loss_length/mean"] = torch.mean(loss_length).detach().item()
        metrics["response_length/loss_length/median"] = torch.median(loss_length).detach().item()
        metrics["response_length/loss_length/max"] = torch.max(loss_length).detach().item()
        metrics["response_length/loss_length/min"] = torch.min(loss_length).detach().item()
        for data_source in set(data_sources):
            mask = torch.tensor([data_source == ds for ds in data_sources], dtype=torch.bool)
            data_source_loss_length = torch.masked_select(loss_length, mask)
            metrics[f'response_length/loss_length/{data_source}/mean'] = torch.mean(data_source_loss_length).detach().item()
            metrics[f'response_length/loss_length/{data_source}/median'] = torch.median(data_source_loss_length).detach().item()
            metrics[f'response_length/loss_length/{data_source}/max'] = torch.max(data_source_loss_length).detach().item()
            metrics[f'response_length/loss_length/{data_source}/min'] = torch.min(data_source_loss_length).detach().item()

    if "search_bad_format" in batch.non_tensor_batch:
        metrics["response/search_bad_format"] = np.mean(batch.non_tensor_batch["search_bad_format"])

    if "search_count" in batch.non_tensor_batch:
        metrics["response/search_count/mean"] = np.mean(batch.non_tensor_batch["search_count"])
        metrics["response/search_count/max"] = np.max(batch.non_tensor_batch["search_count"])
        metrics["response/search_count/min"] = np.min(batch.non_tensor_batch["search_count"])
        metrics["response/search_count/atleast_1"] = np.count_nonzero(batch.non_tensor_batch["search_count"] >= 1) / len(batch.non_tensor_batch["search_count"])
        metrics["response/search_count/atleast_2"] = np.count_nonzero(batch.non_tensor_batch["search_count"] >= 2) / len(batch.non_tensor_batch["search_count"])

    if "document_count" in batch.non_tensor_batch:
        metrics["response/document_count/mean"] = np.mean(batch.non_tensor_batch["document_count"])
        metrics["response/document_count/max"] = np.max(batch.non_tensor_batch["document_count"])
        metrics["response/document_count/min"] = np.min(batch.non_tensor_batch["document_count"])
        metrics["response/document_count/atleast_1"] = np.count_nonzero(batch.non_tensor_batch["document_count"] >= 1) / len(batch.non_tensor_batch["document_count"])

    return metrics