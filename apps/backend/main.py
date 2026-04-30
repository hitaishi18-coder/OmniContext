from fastapi import FastAPI
from dotenv import load_dotenv
import os

# We'll use these later for the Unbiased AI decision loop
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

load_dotenv()

app = FastAPI(title="OmniContext AI Engine", description="Unbiased Drive-RAG API")

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "OmniContext Backend",
        "message": "LangGraph/FastAPI engine is running successfully."
    }

@app.get("/health")
async def health_check():
    # In the future, we will check Redis and DB connections here
    return {"status": "healthy"}