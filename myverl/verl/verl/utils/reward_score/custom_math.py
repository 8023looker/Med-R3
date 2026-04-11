from sympy import sympify, latex, symbols, simplify
import re
from typing import List, Dict
import os

TEMPLATE_NAME = os.environ.get("TEMPLATE_NAME", "r1")


class AnswerExtract:
    def __init__(self, template_name: str, UNK: str = "[INVALID]"):
        self.template = template_name
        self.UNK = UNK
    
    def remove_boxed(self, s: str):
        if "\\boxed " in s:
            left = "\\boxed "
            assert s[:len(left)] == left
            return s[len(left):]

        left = "\\boxed{"

        if s[:len(left)] != left:
            return self.UNK
        if s[-1] != "}":
            return self.UNK

        return s[len(left):-1]

    def last_boxed_only_string(self, string):
        idx = string.rfind("\\boxed")
        if "\\boxed " in string:
            return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return self.UNK

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
            retval = self.UNK
        else:
            retval = string[idx:right_brace_idx + 1]

        return retval
    
    def r1_style(self, text: str) -> str:
        text = text.strip()

        if not (text.count("<think>") == 1 and 
                text.count("</think>") == 1):
            return self.UNK
        if not (text.count("<answer>") == 1 and
                text.count("</answer>") == 1):
            return self.UNK
        pattern = r"^<think>(.*?)</think>\s*<answer>(.*?)</answer>$"
        match = re.match(pattern, text, re.DOTALL)
        if not match:
            return self.UNK
        else:
            return self.remove_boxed(self.last_boxed_only_string(match.group(2)))
        
    
    def r1_no_ans_style(self, text: str) -> str:
        text = text.strip()
        if not (text.count("<think>") == 1 and 
                text.count("</think>") == 1):
            return self.UNK
        pattern = r"^<think>(.*?)</think>(.*?)$"
        match = re.match(pattern, text, re.DOTALL)
        if not match:
            return self.UNK
        else:
            return self.remove_boxed(self.last_boxed_only_string(match.group(2)))
        
    def dapo_style(self, text: str) -> str:
        text = text.strip()[-300:]
        pattern = r"(?i)Answer\s*:\s*([^\n]+)"
        match = re.search(pattern, text)
        if not match:
            return self.UNK
        else:
            return match.group(1)

    def extract_answer(self, solution_str: str) -> str:
        if self.template == "r1":
            return self.r1_style(solution_str)
        elif self.template == "r1_no_ans":
            return self.r1_no_ans_style(solution_str)
        elif self.template == "dapo":
            return self.dapo_style(solution_str)
        else:
            return self.UNK


class CustomMathVerify:
    def __init__(self):
        pass

    @staticmethod
    def python_penalty(solution_str: str):
        if "```python" in solution_str:
            penalty = 0.1
        else:
            penalty = 0.0
        return penalty

    def clean_answer(self, answer):
        answer = answer.strip()
        # 如果答案以美元符号包裹，则去除这些符号, 示例: "$123$" -> "123"
        if answer.startswith("$") and answer.endswith("$"):
            answer = answer[1:-1]
        
        # 如果答案以转义的美元符号开头，并且只出现一次，则去除, 示例: "\$123" -> "123"
        if answer.startswith(r"\$") and answer.count(r"\$") == 1:
            answer = answer.replace(r"\$", "")
        
        # 将 \dfrac 替换为 \frac, 示例: "\dfrac{1}{2}" -> "\frac{1}{2}"
        answer = answer.replace(r"\dfrac", r"\frac")
        
        # 移除 LaTeX 中的 \text{} 命令, 示例: "\text{abc}" -> "abc"
        answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", answer)
        
        # 移除 LaTeX 中的 \textbf{} 命令, 示例: "\textbf{abc}" -> "abc"
        answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", answer)
        
        # 移除 LaTeX 中的 \overline{} 命令, 示例: "\overline{abc}" -> "abc"
        answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", answer)
        
        # 规范化简写的平方根表达式, 示例: "sqrta" -> "sqrt{a}"
        answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", answer)
        
        # 移除所有美元符号, 示例: "$123$" -> "123"
        answer = answer.replace("$", "")
        
        # 移除逗号以规范化数字, 示例: "1,000" -> "1000"
        if answer.replace(",", "").isdigit():
            answer = answer.replace(",", "")
        
        return answer
    
    def sympify_answer(self, expression):
        try:
            return latex(sympify(expression))
        except:
            return "None"

    def are_expressions_equivalent(self, expr1, expr2):
        # 包含连续的括号，eval会抛出异常
        if re.match("\(.*?\)\(.*?\)", expr1) or re.match("\(.*?\)\(.*?\)", expr2):
            return False
        try: 
            # 提取表达式中的所有字母  
            variables = set(re.findall(r'[a-zA-Z_]\w*', expr1 + expr2))  
            # 定义这些字母为符号变量  
            symbol_dict = {var: symbols(var) for var in variables}  
            
            # 替换表达式中的变量为符号变量  
            expr1 = eval(expr1, {}, symbol_dict)  
            expr2 = eval(expr2, {}, symbol_dict)  
            # 简化表达式的差  
            simplified_diff = simplify(expr1 - expr2)  
            # 如果差为 0，则说明等价  
            return simplified_diff == 0
        except:
            return False

    @staticmethod
    def verify_math(gold, answer):
        instance = CustomMathVerify()
        gold = instance.clean_answer(gold)
        answer = instance.clean_answer(answer)
        if re.sub(r'\s+', '', gold) == re.sub(r'\s+', '', answer):
            # Badcase: (a + 5)(b + 2)   (a+5)(b+2)
            result = True
        elif re.sub(r'\s+', '', gold) == re.sub(r'\s+', '', f"({answer})"):
            result = True
        elif re.sub(r'\s+', '', gold) == re.sub(r'\s+', '', f"({answer})").replace("sqrt", "\sqrt"):
            result = True
        elif instance.are_expressions_equivalent(gold, answer):
            result = True
        elif instance.sympify_answer(answer) == gold:
            result = True
        else:
            result = False
            # 直接调用 math_verify 模块的函数
            # from math_verify import parse, verify
            # result = verify(
            #     parse(gold), 
            #     parse(answer)
            # )
        return result


def compute_score(solution_str: str,
                  ground_truth: str,
                  template_name: str = TEMPLATE_NAME) -> Dict:
    """Compute the reward score for a solution.
    
    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        
    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Limit solution length for efficiency
    extractor = AnswerExtract(template_name)
    pred = extractor.extract_answer(solution_str)

    if pred == extractor.UNK:
        return {
            "score": -1.0,
            "acc": False,
            "pred": pred,
            "template_name": template_name,
        }
    else:
        acc = CustomMathVerify.verify_math(ground_truth, pred)
        penalty = CustomMathVerify.python_penalty(solution_str)
        reward = 1.0 - penalty if acc else -1.0
        return {
            "score": reward,
            "acc": acc,
            "pred": pred,
            "template_name": template_name,
        }