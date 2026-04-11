from itertools import islice, zip_longest
from collections import defaultdict
import re

def score_search_pattern(s: str):
    # 检查是否满足<search>...</search><document>...</document>格式
    pattern = ["<search>", "</search>", "<document>", "</document>"]
    for i in range(len(pattern)):
        index = s.find(pattern[i])
        if index == -1:
            if i == 0:
                return 1.0
            else:
                return 0.0
        prefix = s[:index]
        for j in range(i+1, len(pattern)):
            if prefix.count(pattern[j]) > 0:
                return 0.0
        s = s[index + len(pattern[i]):]
    return 1.0
    
def clean_search_pattern(s: str):
    while True:
        start_index = s.find("<search>")
        end_index = s.find("</search>")
        if start_index != -1 and end_index != -1 and start_index < end_index:
            s = s[:start_index] + s[end_index + len("</search>"):]
        else:
            break

    while True:
        start_index = s.find("<document>")
        end_index = s.find("</document>")
        if start_index != -1 and end_index != -1 and start_index < end_index:
            s = s[:start_index] + s[end_index + len("</document>"):]
        else:
            break
    
    return s.replace("<search>", "").replace("</search>", "").replace("<document>", "").replace("</document>", "").strip()

def score_think_pattern(s: str, not_need_think_at_start: bool = False, not_need_answer_tag: bool = False, soft_score = 0.0):
    """
    是否满足 <think>...</think> <answer>...</answer> 格式
    分值区间 [0, 1]
    not_need_think_at_start: 开头是否需要 <think> 标签
    not_need_answer_tag: 是否需要 <answer> 标签
    soft_score: 对格式放松要求，只要保证有明确的 think + answer 结构，就不会给最低分
    """
    for x in ["</s>", "<|im_end|>", "<|endoftext|>", "<|end_of_sentence|>"]:
        s = s.replace(x, "")
    s = s.strip()

    if not_need_think_at_start:
        s = "<think>" + s

    think_end = s.rfind("</think>")
    # 1. 至少包含 </think>
    if think_end == -1:
        return 0.0
    # 1.2 <think> ... </think> 之间不能包含 <answer> </answer>
    if "<answer>" in s[:think_end] or "</answer>" in s[:think_end]:
        return 0.0
    # 1.3 answer 内容不能为空
    answer_content = s[think_end + len("</think>"):]
    answer_content = answer_content.strip()
    if len(answer_content.replace("<answer>", "").replace("</answer>", "").strip()) == 0:
        return 0.0
    # 2.1 是否以 <think> 开头
    if not s.startswith("<think>"):
        return soft_score
    # 2.2 是否包含 <think>...</think> 且只包含一次
    if s.count("<think>") != 1 or s.count("</think>") != 1:
        return soft_score
    # 2.3 是否包含 <answer>...</answer> 且只包含一次
    if not not_need_answer_tag:
        if answer_content.count("<answer>") != 1 or answer_content.count("</answer>") != 1:
            return soft_score
    # 2.4 </answer>之后是否还有多余内容
        if len(answer_content[answer_content.rfind("</answer>") + len("</answer>"):].strip()) != 0:
            return soft_score
    # 2.5 </think> 与 <answer> 之间是否有多余内容
        if len(s[think_end + len("</think>"):answer_content.find("<answer>")].strip()) != 0:
            return soft_score
    return 1.0

def endswith_think(s: str):
    for x in ["</s>", "<|im_end|>", "<|endoftext|>", "<|end_of_sentence|>"]:
        s = s.replace(x, "")
    s = s.strip()
    return s.endswith("<think>")

def get_think_and_answer(s: str, clean_search: bool = True):
    think_content = ""
    answer_content = ""
    if s.startswith("<think>"):
        s = s[len("<think>"):]
    if "</think>" in s:
        think_content = s[:s.rfind("</think>")]
        answer_content = s[s.rfind("</think>") + len("</think>"):]
    else:
        answer_content = s
    if "<answer>" in answer_content:
        answer_content = answer_content[answer_content.find("<answer>") + len("<answer>"):]
    if "</answer>" in answer_content:
        answer_content = answer_content[:answer_content.rfind("</answer>")]
    if clean_search:
        think_content = clean_search_pattern(think_content)
        answer_content = clean_search_pattern(answer_content)
    think_content = think_content.strip()
    answer_content = answer_content.strip()
    return think_content, answer_content

def score_repeatness(s: str):
    """
    计算重复分数，分数越高表示包含重复子串比例越高
    分值区间 [0, 1]
    """

    def ranks(l):
        index = {v: i for i, v in enumerate(sorted(set(l)))}
        return [index[v] for v in l]

    def suffixArray(s):
        # 倍增算法计算后缀数组
        line = ranks(s)
        # print(line)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            # print(line, k)
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        # sa数组表示排名为i的数组的后缀的起始位置
        # ans数组，对应 rk 数组，表示位置是i的后缀的排名
        return ans, sa

    def lcp(arr, suffixArr, inv_suff):
        # suffixArr 表示 sa
        # inv_suff 表示 rk
        n, ans, k = len(arr), [0] * len(arr), 0
        # ans 对应 LCP(i, i+1)
        # 表示 suffix[sa[i]] 和 suffix[sa[i+1]] 的最长公共前缀
        # 即第i名和第i+1名的后缀的最长公共前缀
        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue
            # j, 当前第i位置的下一个排名的后缀的起始位置
            # 即比较两个相邻排名的后缀得到最长公共前缀
            j = suffixArr[inv_suff[i] + 1]
            # 统计最长公共前缀的长度
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1
            # 记录当前第i位置排名的结果
            ans[inv_suff[i]] = k
            # 从下一个位置开始，已知该位置在上一个前缀位置的进一位
            # 如果上一个最大公共前缀是k，那么下一个最大公共前缀至少是k-1，除非是最后一个位置
            if k > 0:
                k -= 1

        return ans

    arr = [ord(i) for i in s]
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffixArray(arr)
    cnt = sum(lcp(arr, sa, c))
    # 每个后缀的最长公共前缀和 除以 所有后缀的长度和
    return cnt * 2 / (n * (n + 1))

def score_reflection_pattern(s: str):
    """
    回答中包含反思词汇的数量，非归一化值
    """
    # TODO: may need to add more pattern
    reflection_pattern_words = [
        r"wait,",
        r"recheck[,\s]",
        r"retry",
        r"alternatively,",
        r"however,",
        r"therefore,",
        r"given that",

    ]
    s = s.lower()
    res = defaultdict(int)
    for word in reflection_pattern_words:
        # can only be followed by a comma or a space
        res[word] = len(re.findall(word, s))
    return sum(res.values())