from fastapi import FastAPI
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

app = FastAPI()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

llm = OllamaLLM(model="qwen2.5:3b")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]

@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "cpu-rag",
                "object": "model",
                "owned_by": "local"
            }
        ]
    }

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):

    question = req.messages[-1].content

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Answer ONLY using this context.

Context:

{context}

Question:

{question}

Answer:
"""

    answer = llm.invoke(prompt)

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }
        ]
    }