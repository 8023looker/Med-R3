import requests
import json
from concurrent.futures import ThreadPoolExecutor

from typing import List, Dict, Any

def batch_search(url: str, queries: List[str], max_workers: int = 8) -> Any:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for query in queries:
            future = executor.submit(search, url, query)
            futures.append(future)
        results = []
        for i, future in enumerate(futures):
            result = future.result()
            results.append(result)
    return results
        
def search(url: str, query: str, top_n=5):
    if not query:
        return None
        
    url = f'{url}/search'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer c4ca4238a0b923820dcc509a6f75849b',
    }
    query_data = {
        "request_id": "keerlutest",
        "query": query, 
        "top_n": top_n,
        "control": {
            "hit_size": 20,
            "timeout": 2000,
        # "debug_level": 1
        }
    }
    retry_count = 0
    while retry_count < 5:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(query_data), timeout=20)
        except Exception as e:
            print(f"Request failed with exception: {e=}, {retry_count=}")
            retry_count += 1
            continue

        if response.status_code == 200:
            response = response.json()
            try:
                debug = []
                response = response["result"]["hits"]
                retrieval_text = ''
                for line in response[:top_n]:
                    chunk = line['fields']['chunk']
                    chunk = chunk.replace("<search>", "").replace("</search>", "").replace("<document>", "").replace("</document>", "")
                    retrieval_text += f"<document>{chunk}</document>"
                    debug.append({
                        "query": query,
                        "chunk": line['fields'].get('chunk'),
                        "id": line['fields'].get('id'),
                        "source_id": line['fields'].get('source_id'),
                        "title": line['fields'].get('title'),
                        "url": line['fields'].get('url'),
                        "score": line.get('score'),
                    })
                retrieval_text = retrieval_text.strip()
                return retrieval_text, debug
            except Exception as e:
                print(f"Response parsing failed with exception: {e=}, {response=}, {retry_count=}")
                retry_count += 1
                continue
        else:
            print(f"Request failed with status code: {response.status_code}")
            return None
    
    return None