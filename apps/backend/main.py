from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Import the ingestion service we just created
from services.ingestion import process_drive_file

# We'll use these later for the Unbiased AI decision loop
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

load_dotenv()

app = FastAPI(title="OmniContext AI Engine", description="Unbiased Drive-RAG API")

# Define the expected data format using Pydantic
class IngestRequest(BaseModel):
    file_id: str

@app.get("/")
async def root():
    return {"status": "online", "message": "LangGraph/FastAPI engine is running successfully."}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# --- NEW ENDPOINT ---
@app.post("/api/ingest")
async def ingest_document(request: IngestRequest):
    """
    Triggers the download and chunking of a Google Drive PDF.
    """
    try:
        chunks = process_drive_file(request.file_id)
        
        if not chunks:
            raise HTTPException(status_code=404, detail="Could not load document. Check File ID and Permissions.")

        return {
            "status": "success",
            "message": f"Successfully processed and split document into {len(chunks)} chunks.",
            # Return a preview of the first chunk so we know it worked!
            "preview_chunk": chunks[0].page_content[:200] + "..." 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))