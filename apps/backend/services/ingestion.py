# apps/backend/services/ingestion.py

import io
import tempfile
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# NEW IMPORTS FOR MEMORY (Vector DB)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 1. Authenticate our Robot with Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
creds = service_account.Credentials.from_service_account_file(
    'credentials.json', scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)

def process_drive_folder(folder_id: str):
    """
    Sweeps a Google Drive Folder, downloads PDFs into secure temp files, 
    chunks them, and saves the vectors to ChromaDB.
    """
    print(f"Starting Omni-Router ingestion for Folder ID: {folder_id}")
    
    # Safety Check: Ensure Gemini API key is loaded
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY is missing from the environment.")
        return None

    all_chunks = []
    
    try:
        # 2. Query Google Drive for all files inside this specific folder
        query = f"'{folder_id}' in parents and trashed=false"
        results = drive_service.files().list(
            q=query, 
            fields="files(id, name, mimeType)",
            pageSize=100
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            print("No files found. The folder is either empty or not shared properly.")
            return None
            
        print(f"Found {len(files)} files. Beginning extraction sequence...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        
        # 3. Loop through every file in the folder
        for file in files:
            file_id = file.get('id')
            file_name = file.get('name')
            mime_type = file.get('mimeType')
            
            print(f"Routing: {file_name} ({mime_type})")
            
            if mime_type != 'application/pdf':
                print(f" -> Skipping {file_name} (Not a PDF).")
                continue
                
            # 4. Download the binary bytes
            request = drive_service.files().get_media(fileId=file_id)
            file_bytes = io.BytesIO()
            downloader = MediaIoBaseDownload(file_bytes, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                
            # 5. Save to self-deleting temp file and chunk it
            with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
                temp_pdf.write(file_bytes.getvalue())
                temp_pdf.flush()
                
                loader = PyPDFLoader(temp_pdf.name)
                docs = loader.load()
                
                # Tag the metadata so Gemini knows which file this text came from
                for doc in docs:
                    doc.metadata['source'] = file_name
                    doc.metadata['drive_id'] = file_id
                
                chunks = text_splitter.split_documents(docs)
                all_chunks.extend(chunks)
                print(f" -> Success: Added {len(chunks)} chunks.")

        # 6. --- NEW CODE: SAVE TO VECTOR DATABASE ---
        if all_chunks:
            print("Converting chunks to Gemini Vectors and saving to DB...")
            
            # CHANGE THIS LINE: 
            # From: model="models/text-embedding-004"
            # To:   model="models/gemini-embedding-001"
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            
            vectorstore = Chroma.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            print("Successfully saved to ChromaDB!")

        return all_chunks

    except Exception as e:
        print(f"Error during folder ingestion: {e}")
        return None