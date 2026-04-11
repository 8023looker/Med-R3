import math
from my_reward.auxiliary.format_reward import (
    score_think_pattern,
    endswith_think, 
    get_think_and_answer,
    clean_search_pattern
) 
from my_reward.contrib.base import RewardActorBase

class RewardActorMedReasoningLogical2(RewardActorBase):

    @classmethod
    def compute_format_score(
        cls, 
        prompt, 
        response
    ):
        think_str, answer_str = get_think_and_answer(response)

        result = score_think_pattern(
            response, 
            not_need_think_at_start=endswith_think(prompt), 
            not_need_answer_tag=("<answer>" not in prompt),
            soft_score=0.5 if "<search>" in prompt and "</search>" in prompt else 0.0,
        )
        result = float(result)

        if "<search>" in prompt and "</search>" in prompt:

            if response.count("<search>") != response.count("</search>"):
                return min(result, 0.0)
            
            if response.count("<document>") != response.count("</document>"):
                return min(result, 0.5)
            
            if "<search>" in answer_str \
                or "</search>" in answer_str \
                or "<document>" in answer_str \
                or "</document>" in answer_str:
                return min(result, 0.5)
            
        return result

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
        think_str = clean_search_pattern(think_str)
        answer_str = clean_search_pattern(answer_str)
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
        think_str = clean_search_pattern(think_str)
        think_language_score = cls.compute_language_score(response=think_str, prompt=question)
        reward -= (1.0 - think_language_score) / 10.0
        answer_str = clean_search_pattern(answer_str)
        answer_language_score = cls.compute_language_score(response=answer_str, prompt=question)
        reward -= (1.0 - answer_language_score) / 10.0
        return reward

if __name__ == "__main__":
    prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nA conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind (invisible to the User) and then provides the User with the answer. During thinking, the assistant can invoke a medical search tool to search the fact information about specific medical topics if needed. The step-by-step reasoning process is enclosed within <think> </think> and followed by the wrap-up answer. Assistant can invoke search tool by specify a search query enclosed within <search> and </search>, and the search result will be placed between <document> and </document>. The search tool can only used in thinking section, and the number of times the search tool can be used is unlimited. For example, <think> reasoning process here <search> search query </search> <document> search result </document> reasoning process </think> answer here. The reasoning process and the answer should be in the same language as the question, but there's no limit for the search query and search result.\nA 67-year-old, 60-kg homeless man has been in the intensive care unit (ICU) for a week after an emergency laparotomy and sigmoid resection for perforated diverticulitis. His serum albumin is 1.1 g/dL. He was just weaned from mechanical ventilation. His colostomy is not functioning. You start total parenteral nutrition (TPN) to deliver 1800 kcal/24 h. Two days later, the patient is in respiratory distress and requires reintubation and mechanical ventilation. What serum level should you check in this patient?<|im_end|>\n<|im_start|>assistant\n"
    response = "<think> reasoning process here <search> pulmonary function test </search><document>Pulmonary Function Testing (PFT)Pulmonary function tests measure the lungs' capacity to hold air, to move air in and out, and to absorb oxygen.\n\nPulmonary function tests are better at detecting the general type and severity of lung disorder than at defining the specific cause of problems; however, these tests can be used to diagnose some specific disorders, such as asthma and chronic obstructive pulmonary disease (COPD).\nasthma\nchronic obstructive pulmonary disease (COPD)\n(See also Medical History and Physical Examination for Lung Disorders and Respiratory System.)\nMedical History and Physical Examination for Lung Disorders\nRespiratory System\nLung flow rate measurements\nThe assessment of a lung disorder often involves testing how much air the lungs can hold (lung volume) as well as how much and how quickly air can be exhaled (airflow). Airflow measurements are made with a spirometer, which consists of a mouthpiece and tubing connected to a recording device. The person’s lips should be held tightly around the mouthpiece, and nose clips should be worn to ensure that all the air inhaled or exhaled goes through the mouth. A person inhales deeply, then exhales forcefully as quickly as possible through the tubing while measurements are taken. The volume of air inhaled and exhaled and the length of time each breath takes are recorded and analyzed. This measurement is repeated several times to be sure the results are consistent. Often, the tests are repeated after a person takes a drug that opens the airways of the lungs (bronchodilator). In disorders such as asthma and chronic obstructive pulmonary disease (COPD), the ability to exhale quickly is impaired.\nasthma\nchronic obstructive pulmonary disease\nA spirometer consists of a mouthpiece, tubing, and a recording device.</document><document>TestsofPulmonaryFunction(PFT)Pulmonary function tests provide measures of airflow, lung volumes, gas exchange, response to bronchodilators, and respiratory muscle function.\nairflow\nlung volumes\ngas exchange\nrespiratory muscle function\nBasic pulmonary function tests available in the ambulatory setting include\nSpirometry\nPulse oximetry\nSpirometry and pulse oximetry provide physiologic measures of pulmonary function and can be used to quickly narrow a differential diagnosis and suggest a subsequent strategy of additional testing or therapy. More complicated testing includes\npulse oximetry\nMeasurement of lung volumes\nLung, chest wall, and respiratory system compliance\nComplete cardiopulmonary exercise testing\ncardiopulmonary exercise testing\nThese tests provide a more detailed description of physiologic abnormalities and the likely underlying pathology. The choice and sequence of testing are guided by information taken from the history and physical examination.\nhistory and physical examination</document><document>Pulmonary function testing (PFT) includes evaluation of total pulmonary capacity (TPC), forced vital capacity (fVC) in sitting and lying positions, maximum inspiratory\nand expiratory pressures, sniff test, and peak cough flow. Assessments of inspiratory and expiratory muscle strength and the sniff test are the most sensitive tests for detecting alveolar hypoventilation.\n\nThe findings indicating neuromuscular respiratory damage are as follows:\n\n- $\\mathrm{fVC}<80 \\%$ when seated.\n\n- fVC reduction of more than $20 \\%$ in the supine position in favor of a diaphragmatic dysfunction.\n\n- Inspiratory muscle strength $<80 \\mathrm{cmH}_{2} \\mathrm{O}$ in males and $<60 \\mathrm{cmH}_{2} \\mathrm{O}$ in females and sniff nasal inspiratory pressure $(\\mathrm{SNIP})<60 \\%$ of the theoretical values.\n\n- Peak cough flow $<270 \\mathrm{~L} / \\mathrm{min}$, which indicates the need for assisted coughing techniques, or $<180 \\mathrm{~L} / \\mathrm{min}$, which indicates the need for mechanical-assisted coughing techniques.\n\nBlood gas analyses are performed only if PFT shows abnormal results. Hypercapnia $\\left(\\mathrm{pCO}_{2}>45 \\mathrm{mmHg}\\right)$ indicates daytime alveolar hypoventilation and requires ventilation. These analyses can also be performed in patients showing respiratory symptoms unexplained by PFT according to the opinion of the pulmonologist.\n\n</document> </document> reasoning process </think>\n\nThe nurse is worried there is dysfunction of the respiratory muscles, so the first thing you need to check is forced vital capacity (FVC). However, since there is no information available about it, you do not have enough information and cannot perform any further investigation.<|im_end|>"