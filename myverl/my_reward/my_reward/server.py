import uvicorn
import time
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
from inspect import signature

from my_reward.contrib import *

PARAMS = {}

app = FastAPI()

class VerifyRequest(BaseModel):
    params: Optional[Dict] = None
    data_source: str
    reward_actor: str
    prompt_str: str
    response_str: str
    ground_truth: str
    extra_info: Optional[Dict] = None
    search_info: Optional[Dict] = None
    global_plan_score: Optional[str] = "0.0"

@app.post("/verify")
async def verify(request: VerifyRequest):
    try:
        stt = time.time()
        reward_actor = request.reward_actor
        reward_actor = eval(reward_actor)
        compute_score_params = signature(reward_actor.compute_score).parameters
        kwargs = {}
        for key in [
            "data_source",
            "prompt_str",
            "response_str",
            "ground_truth",
            "extra_info",
            "search_info",
            "global_plan_score",
        ]:
            if key in compute_score_params:
                kwargs[key] = getattr(request, key)
        kwargs["params"] = PARAMS
        if request.params:
            for key in request.params:
                kwargs["params"][key] = request.params[key]
        result = reward_actor.compute_score(**kwargs)
        if "acc" not in result:
            result["acc"] = False
        edt = time.time()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{current_time}] Source: {request.data_source}, Gold: {request.ground_truth}, Result: {result}, Latency: {edt - stt:.2f}s")
        if "exception" in result:
            print(f"############## ERROR IN {request.reward_actor}: {result['exception']}")
        return result
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='diagnosis verify server')
    parser.add_argument('--port', type=int, default=30000, help='port')
    parser.add_argument('--url', type=str, help='verify url')
    parser.add_argument('--model', type=str, help='verify model')
    parser.add_argument('--key', type=str, help='verify key')
    parser.add_argument('--max_tokens', type=int, default=4096, help='verify max_tokens')
    parser.add_argument('--temperature', type=float, default=0.6, help='verify temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='verify top_p')
    args = parser.parse_args()
    PARAMS.update({
        "url": args.url,
        "model": args.model,
        "key": args.key,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p
    })
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port
    )