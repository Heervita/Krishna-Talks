import os
import pandas as pd
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

# 1. Load Environment
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# CRITICAL FIX: Explicitly set the API version to 'v1'
client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1'})

# 2. Universal Embedding Function
class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        # 'text-embedding-004' is the stable 2026 standard on v1
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=input
        )
        return [item.values for item in response.embeddings]

# 3. Setup Vector Database
db_client = chromadb.PersistentClient(path="./krishna_db")
custom_ef = GeminiEmbeddingFunction()

# Use 'v4' to ensure we bypass any old, corrupted metadata
collection = db_client.get_or_create_collection(
    name="gita_wisdom_v4", 
    embedding_function=custom_ef
)

def load_and_ingest_gita():
    if collection.count() > 0:
        print("Wisdom already loaded in 'v4'.")
        return

    print("Loading Gita...")
    file_path = 'gita_data.csv' if os.path.exists('gita_data.csv') else 'gita_data.csv'
    
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()
        df = df.fillna('')

        documents, metadatas, ids = [], [], []
        for index, row in df.iterrows():
            content = f"Verse {row.get('verse', index)}: {row.get('text', '')}\nCommentary: {row.get('commentary', '')}"
            documents.append(content)
            metadatas.append({"chapter": int(row.get('chapter', 0))})
            ids.append(f"id_{index}")

        # Batching: Process in chunks of 100 for the API limit
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            print(f"Ingesting batch {i//batch_size + 1}...")
            batch_docs = documents[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            
            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
        print("Ingestion complete! Krishna is ready for the birthday surprise.")
    except Exception as e:
        print(f"Error during ingestion: {e}")

load_and_ingest_gita()

# 4. FastAPI Setup
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str

@app.post("/ask")
async def ask_krishna(request: ChatRequest):
    results = collection.query(query_texts=[request.message], n_results=2)
    context = "\n\n".join(results['documents'][0])
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"As Lord Krishna, guide your friend based on this: {context}\n\nUser: {request.message}"
    )
    return {"response": response.text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)