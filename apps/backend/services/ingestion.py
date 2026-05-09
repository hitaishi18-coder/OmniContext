# apps/backend/services/ingestion.py

import os
from langchain_community.document_loaders import GoogleDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def process_drive_file(file_id: str):
    """
    Downloads a file from Google Drive, extracts the text, and splits it into chunks.
    """
    print(f"Starting ingestion for file ID: {file_id}")
    
    # 1. Load the document directly from Google Drive
    # Note: This requires a credentials.json file from Google Cloud
    loader = GoogleDriveLoader(
        file_ids=[file_id],
        recursive=False
    )
    
    try:
        docs = loader.load()
        print(f"Successfully loaded {len(docs)} document(s).")
    except Exception as e:
        print(f"Error loading document from Drive: {e}")
        return None

    # 2. Chunk the document to prevent LLM hallucination and fit context windows
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(docs)
    print(f"Split document into {len(chunks)} chunks.")
    
    # 3. Prepare the Gemini Embeddings model (We will use this in the next step to push to a Vector DB)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # For now, let's just return the chunks to verify it works
    return chunks