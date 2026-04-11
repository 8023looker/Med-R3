import uvicorn
import time
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from inspect import signature

import torch
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./model/bge-reranker-v2-m3', trust_remote_code=True)
bgem3_model = AutoModelForSequenceClassification.from_pretrained('./model/bge-reranker-v2-m3')
bgem3_model.cuda()
bgem3_model.eval()

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    for _ in range(10):
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=16384)
        inputs = inputs.to(bgem3_model.device)
        stt = time.time()
        result = bgem3_model(**inputs, return_dict=True).logits.view(-1,).float().cpu().numpy().tolist()
        edt = time.time()
        print(f"Result: {result}, Latency: {edt - stt:.2f}s")

app = FastAPI()

class RankRequest(BaseModel):
    query: str
    document: str

@app.post("/rank")
async def rank(request: RankRequest):
    try:
        stt = time.time()
        query = request.query
        document = request.document
        with torch.no_grad():
            inputs = tokenizer([[query, document]], padding=True, truncation=True, return_tensors='pt', max_length=16384)
            inputs = inputs.to(bgem3_model.device)
            result = bgem3_model(**inputs, return_dict=True).logits.view(-1,).float().cpu().numpy().tolist()[0]
        edt = time.time()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{current_time}] Result: {result}, Latency: {edt - stt:.2f}s")
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