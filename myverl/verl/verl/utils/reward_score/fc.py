import re
import sys
import string
# sys.path.append('/global_data/sft/cmy/code/self-rag/retrieval_lm/src')
# import normalize_text

def validate_template_format(text: str) -> tuple[bool, str]:
    """
    验证字符串是否符合模板格式要求
    返回: (是否有效, 错误信息)
    """
    # 检查必需的标签
    required_tags = [
        ('<think>', '</think>'),
        ('<answer>', '</answer>')
    ]
    
    for start_tag, end_tag in required_tags:
        # 检查标签出现次数
        start_count = text.count(start_tag)
        end_count = text.count(end_tag)
        
        if start_count == 0 or end_count == 0:
            return False, f"缺少必需的标签对: {start_tag} {end_tag}"
        
        if start_count > 1 or end_count > 1:
            return False, f"标签对 {start_tag} {end_tag} 只能出现一次，但出现了 {max(start_count, end_count)} 次"
    
    # 获取 think 标签内的内容
    think_start = text.find('<think>')
    think_end = text.find('</think>')
    if think_start > think_end:
        return False, "标签顺序错误: <think> 必须在 </think> 之前"
    think_content = text[think_start:think_end + len('</think>')]
    
    # 检查所有的 search 和 result 标签对是否都在 think 内
    if text.count('<search>') != think_content.count('<search>'):
        return False, "存在 <search> 标签不在 <think> 标签内"
    
    if text.count('</search>') != think_content.count('</search>'):
        return False, "存在 </search> 标签不在 <think> 标签内"
    
    if text.count('<result>') != think_content.count('<result>'):
        return False, "存在 <result> 标签不在 <think> 标签内"
    
    if text.count('</result>') != think_content.count('</result>'):
        return False, "存在 </result> 标签不在 <think> 标签内"
    
    if text.count('<search>') != text.count('<result>'):
        return False, "search 和 result 标签数量不匹配"
    
    # 检查每对 search/result 标签的顺序
    current_pos = 0
    while True:
        search_pos = think_content.find('<search>', current_pos)
        if search_pos == -1:
            break
            
        result_pos = think_content.find('<result>', search_pos)
        search_end_pos = think_content.find('</search>', search_pos)
        result_end_pos = think_content.find('</result>', result_pos)
        
        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False, "search/result 标签不完整"
            
        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False, "search/result 标签嵌套顺序错误"
            
        current_pos = result_end_pos
    
    # 检查答案中是否包含 \boxed{}
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "标签顺序错误: <answer> 必须在 </answer> 之前"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "答案中缺少 \\boxed{} 格式"
    
    return True, "格式正确"

def extract_content(text: str):
    text = text.strip()

    if not (text.count("<think>") == 1 and 
            text.count("</think>") == 1 and
            text.count("<answer>") == 1 and
            text.count("</answer>") == 1):
        return None

    pattern = r"^<think>(.*?)</think>\s*<answer>(.*?)</answer>$"
    match = re.match(pattern, text, re.DOTALL)
    if not match:
        return None
    
    return match.group(1), match.group(2)

def compute_score_with_format(tokenizer, solution_str, ground_truth) -> float:
    solution_str_split = solution_str.split("Assistant:")
    valid_template, _ = validate_template_format(solution_str)
    if valid_template:
        response = solution_str_split[1]
    else:
        return 0  # bad format

    if response.endswith(tokenizer.eos_token):
        response = response[:-len(tokenizer.eos_token)]
    else:
        return 0  # over length

    think_answer = extract_content(response)
    if think_answer is not None:
        think, answer = think_answer
    else:
        return 0  # bad format

    # if normalize_text.normalize_text(answer).lower() \
    #     == normalize_text.normalize_text(ground_truth).lower():
    if answer.lower() == ground_truth.lower():
        return 1  # correct answer
    else:
        return 0.1  # wrong answer but good format

def validate_template_format_2(text: str) -> tuple[bool, str]:
    """
    验证字符串是否符合模板格式要求
    返回: (是否有效, 错误信息)
    """
    # 检查<think></think>是否成对出现
    if text.count('<think>') != text.count('</think>'):
        return False, "存在 <think> </think> 不成对出现"
    
    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "缺少 <think> 或 </think> 标签"
    
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> 或 </answer> 出现次数不为1"        
    
    # 检查每对 search/result 标签的顺序
    current_pos = 0
    while True:
        search_pos = text.find('<search>', current_pos)
        if search_pos == -1:
            break
            
        result_pos = text.find('<result>', search_pos)
        search_end_pos = text.find('</search>', search_pos)
        result_end_pos = text.find('</result>', result_pos)
        
        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False, "search/result 标签不完整"
            
        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False, "search/result 标签嵌套顺序错误"
            
        current_pos = result_end_pos
    
    # 检查答案中是否包含 \boxed{}
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "标签顺序错误: <answer> 必须在 </answer> 之前"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "答案中缺少 \\boxed{} 格式"
    
    return True, "格式正确"

def extract_answer(text: str):
    text = text.strip()

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    
    return match.group(1)

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def compute_score_with_format_2(tokenizer, solution_str, ground_truth) -> float:
    if "<|im_start|>assistant\n" in solution_str:
        solution_str_split = solution_str.split("<|im_start|>assistant\n")
    else:
        solution_str_split = solution_str.split("Assistant:")
    
    response = solution_str_split[1]
    valid_template, reason = validate_template_format_2(response)
    if not valid_template:
        return 0, f'bad format: {reason}'

    if response.endswith(tokenizer.eos_token):
        response = response[:-len(tokenizer.eos_token)]
    else:
        return 0, f'over length'

    answer_part = extract_answer(response)
    if answer_part is not None:
        try:
            answer = remove_boxed(last_boxed_only_string(answer_part))
        except Exception as e:
            return 0, f'find box error: {e}'
    else:
        return 0, f'cannot extract answer'

    # if normalize_text.normalize(answer).strip().lower() \
    #     == normalize_text.normalize(ground_truth).strip().lower():
    # if answer.lower() == ground_truth.lower():
    f1_score = get_f1_score(answer, ground_truth)
    if f1_score > 0:
        return f1_score, f'correct answer, get f1 score: {f1_score}'
    else:
        return 0.1, f'wrong answer but good format: {answer}'

# ## f1 score

from typing import Union, List
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_f1_score(prediction: str, ground_truths: Union[str, List[str]]):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])
    
    return final_metric['f1']

def validate_template_format_3(text: str) -> tuple[bool, str]:
    """
    验证字符串是否符合模板格式要求
    返回: (是否有效, 错误信息)
    """
    # 提取所有assistant回复
    assistant_responses = []
    current_pos = 0
    while True:
        start_pos = text.find("<|im_start|>assistant\n", current_pos)
        if start_pos == -1:
            break
        end_pos = text.find("<|im_end|>", start_pos)
        if end_pos == -1:
            break
        response = text[start_pos + len("<|im_start|>assistant\n"):end_pos].strip()
        assistant_responses.append(response)
        current_pos = end_pos + len("<|im_end|>")

    if not assistant_responses:
        return False, "未找到任何assistant回复"

    # 检查最后一个回复
    for response in assistant_responses:
        # 1. 检查<think>和</think>成对出现
        think_count = response.count("<think>")
        think_end_count = response.count("</think>")
        if think_count != think_end_count:
            return False, f"<think>和</think>不成对出现: think={think_count}, think_end={think_end_count}"
        if think_count == 0:
            return False, "缺少<think>标签"

        # 2. 检查<tool_call>和</tool_call>成对出现
        tool_call_count = response.count("<tool_call>")
        tool_call_end_count = response.count("</tool_call>")
        if tool_call_count != tool_call_end_count:
            return False, f"<tool_call>和</tool_call>不成对出现: tool_call={tool_call_count}, tool_call_end={tool_call_end_count}"

        # 3. 检查每个tool_call的内容是否可以被json解析
        current_pos = 0
        while True:
            tool_call_start = response.find("<tool_call>", current_pos)
            if tool_call_start == -1:
                break
            tool_call_end = response.find("</tool_call>", tool_call_start)
            if tool_call_end == -1:
                break
            
            tool_call_content = response[tool_call_start + len("<tool_call>"):tool_call_end].strip()
            
            # 检查是否包含name和arguments
            if '"name"' not in tool_call_content or '"arguments"' not in tool_call_content:
                return False, "tool_call缺少name或arguments字段"
            
            try:
                import json
                json.loads(tool_call_content)
            except json.JSONDecodeError:
                return False, f"tool_call内容不是有效的JSON格式: {tool_call_content}"
            
            current_pos = tool_call_end + len("</tool_call>")

    # 4. 检查最后一个回复是否包含\box
    if "\\box" not in assistant_responses[-1]:
        return False, "最后一个回复中缺少\\box"

    return True, assistant_responses[-1]

def compute_score_with_format_3(tokenizer, solution_str, ground_truth) -> tuple[float, str]:
    if not solution_str.endswith(tokenizer.eos_token):
        return 0, f'not end with eos token'
    
    valid_template, reason = validate_template_format_3(solution_str)
    if not valid_template:
        return 0, f'bad format: {reason}'
    else:
        response = reason

    try:
        answer = remove_boxed(last_boxed_only_string(response))
    except Exception as e:
        return 0, f'find box error: {e}'

    f1_score = get_f1_score(answer, ground_truth)
    if f1_score > 0:
        return f1_score, f'correct answer, get f1 score: {f1_score}'
    else:
        return 0.1, f'wrong answer but good format: {answer}'