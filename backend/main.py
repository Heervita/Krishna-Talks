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
import asyncio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 1. Load Environment
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini Client for the "Chatting" part
client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1'})

# 2. Local Embedding Function (Solves all API rate limit issues!)
class LocalEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self):
        # This model runs on your CPU, so no more 404/400 errors from Google
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        embeddings = self.model.encode(input)
        return embeddings.tolist()

# 3. Setup Vector Database
db_client = chromadb.PersistentClient(path="./krishna_db")
# We use the local function here
custom_ef = LocalEmbeddingFunction()

# Use 'v5' to ensure a completely fresh start with local embeddings
collection = db_client.get_or_create_collection(
    name="gita_wisdom_v5", 
    embedding_function=custom_ef
)

def load_and_ingest_gita():
    if collection.count() > 0:
        print("Wisdom already loaded in 'v5'.")
        return

    print("Loading Gita...")
    # Flexible path check
    file_path = 'gita_data.csv' if os.path.exists('gita_data.csv') else 'backend/gita_data.csv'
    
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

        # Batching: Since it's local, this will be very fast
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
            # time.sleep is no longer strictly needed for local, 
            # but we'll keep a tiny bit (0.1) just for stability.
            time.sleep(0.1)

        print("Ingestion complete! Krishna is ready.")
    except Exception as e:
        print(f"Error during ingestion: {e}")

# Run ingestion when the server starts
load_and_ingest_gita()

# 4. FastAPI Setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    message: str

@app.post("/ask")
async def ask_krishna(request: ChatRequest):
    try:
        # 1. Search the Gita wisdom locally
        results = collection.query(query_texts=[request.message], n_results=2)
        context = "\n\n".join(results['documents'][0])
        
        # 2. The Personalized System Prompt
        # We tell the AI exactly who it is talking to and how to treat her
        prompt = f"""
        Role: You are Lord Krishna, the divine guide.
        Recipient: Your dear friend Vidisha (whom you always affectionately call 'Vids').
        
        Character Insight: Vids is a soul who is constantly striving to be a better version of herself. 
        She values growth, duty, and inner peace.
        
        Gita Context to use: 
        {context}
        
        User's Struggle: "{request.message}"
        
        Instruction: 
        1. Address her as 'Vids' at least once in your response.
        2. Acknowledge her constant effort to improve herself.
        3. Use a tone that is serene, compassionate, and encouraging.
        4. Provide guidance based on the provided Gita context, explaining it simply.
        """
        
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=prompt
        )
        return {"response": response.text}
    
    except Exception as e:
        print(f"Error: {e}")
        if "429" in str(e):
            return {"response": "My dear Vids, even the heavens take a moment to breathe. Please wait a heartbeat and ask me again."}
        return {"response": "Vids, I am here, but the connection is flickering. Let us try again."}
