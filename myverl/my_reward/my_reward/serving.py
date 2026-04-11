import os
import json
import time
from fastapi import FastAPI
import argparse
from pydantic import BaseModel
import my_reward.contrib
from typing import Dict, Optional, List

def timestamp():
    # 获取当前时间戳并转换成字符串
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

from my_reward.contrib import *

class QueryRequest(BaseModel):
    reward_actor: str
    data_source: str
    prompt_str: str
    response_str: str
    ground_truth: str
    extra_info: Dict = None
    global_plan_score: str = "0.0"

class Response(BaseModel):
    reward: float

app = FastAPI()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
    }

@app.post("/compute_score", response_model=Response)
def compute_score(request: QueryRequest):
    reward_actor = request.reward_actor
    data_source = request.data_source
    prompt_str = request.prompt_str
    response_str = request.response_str
    ground_truth = request.ground_truth
    extra_info = request.extra_info
    global_plan_score = request.global_plan_score
    if extra_info:
        extra_info = json.loads(extra_info)
    else:
        extra_info = {}
    
    reward_actor_cls = eval(f"my_reward.contrib.{reward_actor}")
    reward_actor_instance = reward_actor_cls()
    try:
        stt = time.time()
        if global_plan_score:
            reward = reward_actor_instance.compute_score(data_source, prompt_str, response_str, ground_truth, extra_info, global_plan_score)
        else:
            reward = reward_actor_instance.compute_score(data_source, prompt_str, response_str, ground_truth, extra_info)
        edt = time.time()
        print(f"{timestamp()} ################### {reward_actor} compute_score time: {edt - stt}: reward: {reward}")
        return Response(reward=reward)
    except Exception as e:
        edt = time.time()
        print(f"{timestamp()} ################### {reward_actor} failed: {e}, compute_score time: {edt - stt}")
        return Response(reward=reward_actor_instance.default)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", 
        type=int, 
        default=80,
        help="port to use for the serving"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of workers to use"
    )
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=args.workers)