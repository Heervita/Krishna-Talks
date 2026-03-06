import os
import time
import pandas as pd
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

# 1. Setup & Environment
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# 1. Update the Client to force 'v1'
client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1'})

# 2. Local Embedding (Fast & Reliable)
class LocalEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return self.model.encode(input).tolist()

db_client = chromadb.PersistentClient(path="./krishna_db")
custom_ef = LocalEmbeddingFunction()
collection = db_client.get_or_create_collection(name="gita_v5", embedding_function=custom_ef)

# 3. FastAPI App Setup
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str

@app.post("/ask")
async def ask_krishna(request: ChatRequest):
    try:
        # Search Local DB
        results = collection.query(query_texts=[request.message], n_results=2)
        context = "\n\n".join(results['documents'][0])
        
        # Personalized Prompt for Vids
        prompt = f"""
        You are Lord Krishna speaking to your dear friend Vids (Vidisha).
        Vids always strives to be a better version of herself.
        Context: {context}
        User says: {request.message}
        Instruction: Address her as Vids. Be compassionate and encouraging.
        """
        
        # 2. In your @app.post("/ask") section:
        # Change the model name to exactly "gemini-1.5-flash"
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # 2.0 is the definitive stable version for v1
            contents=prompt
        )
        return {"response": response.text}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"response": "Vids, the connection is flickering, but I am here. Try again in a heartbeat."}

# 4. THE CRITICAL FIX: This ensures the server actually stays running
if __name__ == "__main__":
    import uvicorn
    print("Divine connection established. Starting server on Port 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)