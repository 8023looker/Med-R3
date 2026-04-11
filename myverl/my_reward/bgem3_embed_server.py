import uvicorn
import time
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from inspect import signature

import torch
from FlagEmbedding import BGEM3FlagModel

bgem3_model = BGEM3FlagModel("./models/bge-m3/", use_fp16=True, device="cuda:0")
for _ in range(10):
    stt = time.time()
    bgem3_model.encode("根据提供的搜索结果，用户提到的Uvicorn是一个ASGI服务器，通常用于运行FastAPI或类似的Python应用。而Nginx作为反向代理和负载均衡器，可以将流量分发到多个后端服务器", max_length=16384)
    edt = time.time()
    print(f"Latency: {edt - stt:.2f}s")

app = FastAPI()

class EmbeddingRequest(BaseModel):
    text: str

@app.post("/embedding")
async def embedding(request: EmbeddingRequest):
    try:
        stt = time.time()
        text = request.text
        result = bgem3_model.encode(text, max_length=16384)["dense_vecs"].tolist()
        edt = time.time()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{current_time}] Latency: {edt - stt:.2f}s")
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
    parser = argparse.ArgumentParser(description='bgem3 embedding server')
    parser.add_argument('--port', type=int, default=30000, help='port')
    args = parser.parse_args()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port
    )