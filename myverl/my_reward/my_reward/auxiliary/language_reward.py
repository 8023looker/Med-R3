import re

def score_language_consistency(s1: str, s2: str, split_char_length: int = 200):
    """
    语言一致性分值
    分值区间 [0, 1]
    s2 分段并评估语言，最终用 target 语言占比作为分值
    只区分是否包含中文

    同时要避免模型趋于生成尽可能短的文本来逃避惩罚
    """
    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    lang1 = bool(zh_pattern.search(s1))
    
    lang2_list = []
    for i in range(0, len(s2), split_char_length):
        lang2 = bool(zh_pattern.search(s2[i:i+split_char_length]))
        lang2_list.append(lang2)
    
    if len(lang2_list) == 0:
        return 0.0
    
    return lang2_list.count(lang1) * 1.0 / len(lang2_list)
