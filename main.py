from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimilarityRequest(BaseModel):
    docs: list[str]
    query: str

def embed(text):
    res = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(res.data[0].embedding)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/similarity")
async def similarity(request: SimilarityRequest):
    query_vec = embed(request.query)
    scored = [(cosine_sim(query_vec, embed(doc)), doc) for doc in request.docs]
    scored.sort(reverse=True)
    return {"matches": [doc for _, doc in scored[:3]]}
