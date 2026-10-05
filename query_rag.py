from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

CHROMA_PATH = "chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model
)

query = input("Ask your question: ")

results = db.similarity_search(query, k=3)

context = "\n\n".join([doc.page_content for doc in results])

prompt = PromptTemplate.from_template("""
You are an expert in CPU Architecture.

Answer ONLY using the context below.

Context:
{context}

Question:
{question}

Answer:
""")

final_prompt = prompt.format(
    context=context,
    question=query
)

llm = OllamaLLM(
    model="qwen2.5:3b"
)

response = llm.invoke(final_prompt)

print("\n==========================")
print(response)
print("==========================")