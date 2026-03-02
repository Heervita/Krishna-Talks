import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Setup the Vector Database
# This creates a folder called 'krishna_db' to store the "brain"
client = chromadb.PersistentClient(path="./krishna_db")

# 2. Use Google's Embedding Function (since you have the API key)
gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=os.getenv("GOOGLE_API_KEY")
)

collection = client.get_or_create_collection(
    name="gita_wisdom", 
    embedding_function=gemini_ef
)

def ingest_data(file_path):
    df = pd.read_csv(file_path)
    
    # Prepare the data for the vector store
    documents = []
    metadatas = []
    ids = []

    for index, row in df.iterrows():
        # Combine Verse and Commentary for better "context"
        content = f"Verse {row['verse']}: {row['text']} \nCommentary: {row['commentary']}"
        documents.append(content)
        
        # Metadata helps the LLM know exactly which Chapter/Verse it's looking at
        metadatas.append({"chapter": int(row['chapter']), "verse": int(row['verse'])})
        ids.append(f"id_{index}")

    # Add to ChromaDB (This might take a minute the first time)
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print("Wisdom successfully stored in the vector database!")

def query_wisdom(user_query):
    # This finds the top 2 most relevant verses based on the 'meaning' of the query
    results = collection.query(
        query_texts=[user_query],
        n_results=2
    )
    return results['documents'][0]