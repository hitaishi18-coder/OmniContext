from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Import our custom services
from services.ingestion import process_drive_folder
from services.chat import get_answer

# We'll use these later for the Unbiased AI decision loop
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

load_dotenv()

app = FastAPI(title="OmniContext AI Engine", description="Unbiased Drive-RAG API")

# --- DATA MODELS ---
class IngestRequest(BaseModel):
    folder_id: str

class ChatRequest(BaseModel):
    question: str

# --- HEALTH ROUTES ---
@app.get("/")
async def root():
    return {"status": "online", "message": "LangGraph/FastAPI engine is running successfully."}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# --- INGESTION ROUTE ---
@app.post("/api/ingest")
async def ingest_document(request: IngestRequest):
    """
    Triggers the download and chunking of a Google Drive Folder.
    """
    try:
        chunks = process_drive_folder(request.folder_id)
        
        if not chunks:
            raise HTTPException(status_code=404, detail="Could not load folder. Check Folder ID and Permissions.")

        return {
            "status": "success",
            "message": f"Successfully processed and split folder into {len(chunks)} chunks.",
            "preview_chunk": chunks[0].page_content[:200] + "..." if chunks else ""
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW: CHAT ROUTE ---
@app.post("/api/chat")
async def chat_with_omni(request: ChatRequest):
    """
    Sends a question to the Unbiased Drive-RAG engine.
    """
    try:
        result = get_answer(request.question)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return {
            "status": "success",
            "answer": result["answer"],
            "sources": result["sources"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))