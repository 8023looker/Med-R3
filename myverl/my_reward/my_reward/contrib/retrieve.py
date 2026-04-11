from my_reward.contrib.base import Reason
from my_reward.contrib.base_no_api import RewardActorNoApi

class RewardActorRetrieve(RewardActorNoApi):

    @classmethod
    def compute_score(
        cls, 
        params, 
        data_source, 
        prompt_str, 
        response_str, 
        ground_truth, 
        extra_info,
        search_info,
    ):
        format_score = cls.compute_format_score(prompt_str, response_str)
        if format_score == 0.0:
            result = {
                "reason": Reason.FORMAT_WRONG.value,
                "reward": cls.default
            }
        elif not search_info:
            result = {
                "reason": Reason.NO_SEARCH.value,
                "reward": cls.default
            }
        else:
            ground_truth_title_dict = {k: False for k in ground_truth}
            for info in search_info:
                if info["title"] and info["title"] in ground_truth_title_dict:
                    ground_truth_title_dict[info["title"]] = True
            r = sum(ground_truth_title_dict.values()) * 1.0 / len(ground_truth_title_dict)
            result = {
                "reason": Reason.CORRECT.value if r == 1.0 else Reason.WRONG.value,
                "reward": r,
                "acc": r == 1.0
            }
        result["reward"] = cls.add_penalty(result["reward"], prompt_str, response_str, extra_info)
        return result