import math
from my_reward.api import (
    oneapi_post,
    oneapi_post_by_langchain,
    read_json
)
from my_reward.auxiliary.format_reward import (
    score_think_pattern,
    endswith_think, 
    get_think_and_answer
) 
from my_reward.auxiliary.language_reward import (
    score_language_consistency
)
from enum import Enum

class Reason(Enum):
    DEFAULT = "DEFAULT"
    FORMAT_WRONG = "FORMAT_WRONG"
    CORRECT = "CORRECT"
    WRONG = "WRONG"
    TIMEOUT = "TIMEOUT"
    API_ERROR = "API_ERROR"
    NO_SEARCH = "NO_SEARCH"

class RewardActorBaseWithApi:

    default = 0.1
    api_retries = 3

    @classmethod
    def compute_format_score(
        cls, 
        prompt, 
        response
    ):
        score = score_think_pattern(
            response, 
            not_need_think_at_start=endswith_think(prompt), 
            not_need_answer_tag=("<answer>" not in prompt),
            soft_score=0.0,
        )
        score = float(score)
        return score

    @classmethod
    def compute_language_score(
        cls, 
        response,
        prompt,
    ):
        score = score_language_consistency(prompt, response, split_char_length=200)
        return float(score)

    @classmethod
    def add_think_length_penalty(
        cls,
        reward,
        response_str
    ):
        """
        思考长度相对答案长度，越长得分越高，大于 2 倍以上 clip
        """
        think_str, answer_str = get_think_and_answer(response_str)
        think_str_length = len(think_str)
        answer_str_length = len(answer_str)
        score = 0.0
        if answer_str_length > 0:
            # score = (1.0 - math.exp(- min(think_str_length * 1.0 / answer_str_length, 2))) / (1.0 - math.exp(-2))
            score = (1.0 - math.exp(- min(think_str_length * 1.0 / answer_str_length, 2))) / 0.8646647167
        reward -= (1.0 - score) / 10.0
        return reward

    @classmethod
    def add_language_penalty(
        cls,
        reward,
        question,
        response_str,
    ):
        think_str, answer_str = get_think_and_answer(response_str)
        think_language_score = cls.compute_language_score(response=think_str, prompt=question)
        reward -= (1.0 - think_language_score) / 10.0
        answer_language_score = cls.compute_language_score(response=answer_str, prompt=question)
        reward -= (1.0 - answer_language_score) / 10.0
        return reward
        
    @classmethod
    def add_penalty(
        cls, 
        score, 
        prompt_str, 
        response_str,
        extra_info
    ):
        question = extra_info["question"]
        # format
        format_score = cls.compute_format_score(prompt_str, response_str)
        score -= (1.0 - format_score) * 0.1
        # think length
        score = cls.add_think_length_penalty(score, response_str)
        # language
        score = cls.add_language_penalty(score, question, response_str)
        return score
    
    @classmethod
    def get_verify_prompt(
        cls,
        data_source,
        prompt_str,
        response_str,
        ground_truth,
        extra_info,
        **kwargs
    ) -> str:
        raise NotImplementedError("get_verify_prompt is not implemented in base class")
    
    @classmethod
    def normalize_reward(
        cls,
        reward: float,
    ):
        return max(cls.default, min(1.0, reward))

    @classmethod
    def get_final_reward(
        cls,
        res_str: str,
        **kwargs
    ):
        result = {}
        try:
            res_json = read_json(res_str)
            result["reason"] = res_json["reason"]
            result["reward"] = cls.normalize_reward(float(res_json["score"]))
            result["acc"] = result["reward"] == 1.0
        except Exception as e:
            result["reason"] = f"{res_str}"
            result["reward"] = cls.default
            result["exception"] = str(e)
        return result

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
        format_score = cls.compute_format_score(prompt_str, response_str)
        if format_score == 0.0:
            result = {
                "reason": Reason.FORMAT_WRONG.value,
                "reward": cls.default,
            }
        else:
            prompt = cls.get_verify_prompt(
                data_source=data_source,
                prompt_str=prompt_str,
                response_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info
            )
            if not prompt:
                result = {
                    "reason": Reason.WRONG.value,
                    "reward": cls.default
                }
            else:
                retries = cls.api_retries
                while retries > 0:
                    response = oneapi_post(
                        prompt=prompt,
                        url=params["url"],
                        model=params["model"],
                        key=params.get("key", "EMPTY"),
                        max_tokens=params.get("max_tokens", 4096),
                        temperature=params.get("temperature", 0.9),
                        top_p=params.get("top_p", 0.6),
                    )
                    result = cls.get_final_reward(response)
                    if result.get("exception"):
                        retries -= 1
                        continue
                    break

        result["reward"] = cls.add_penalty(result["reward"], prompt_str, response_str, extra_info)
        return result