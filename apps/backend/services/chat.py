# apps/backend/services/chat.py

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

def get_answer(question: str):
    """
    Searches the Vector DB and forces Gemini to answer strictly based on the context.
    """
    print(f"Incoming question: {question}")
    
    if not os.environ.get("GOOGLE_API_KEY"):
        return {"error": "Google API Key missing"}

    # 1. Connect to our long-term memory (ChromaDB)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    try:
        vectorstore = Chroma(
            persist_directory="./chroma_db", 
            embedding_function=embeddings
        )
    except Exception as e:
        return {"error": f"Database not found. Did you ingest files first? {e}"}

    # Pull the top 3 most mathematically relevant chunks to the user's question
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    
    if not docs:
        return {"answer": "I don't have any context to answer that.", "sources": []}

    # 2. Extract the text and the metadata (for citations!)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get('source', 'Unknown') for doc in docs]))

    # 3. Set up the strict "Unbiased" Prompt
    template = """
    You are OmniContext, an unbiased enterprise AI. 
    You must answer the user's question strictly using ONLY the provided context below.
    If the answer is not in the context, you must reply exactly with: "I cannot answer this based on the provided documents."
    Do NOT use outside knowledge or hallucinate.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    # 4. Ask Gemini 1.5 Flash (Fast and accurate for Hackathons)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) 
    final_prompt = prompt.format(context=context_text, question=question)
    
    print("Thinking...")
    response = llm.invoke(final_prompt)
    
    return {
        "answer": response.content,
        "sources": sources
    }