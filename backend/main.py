from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from rag_pipeline import SimpleRAG
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Simple RAG API")
rag_system = None

class VideoRequest(BaseModel):
    video_id: str
    google_api_key: str

class TextRequest(BaseModel):
    text: str
    google_api_key: str

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    conversation_history: List[Dict[str, str]]

@app.get("/")
async def root():
    return {"message": "Simple RAG API is running!"}

@app.post("/setup/youtube")
async def setup_with_youtube(request: VideoRequest):
    global rag_system
    
    try:
        rag_system = SimpleRAG(google_api_key=request.google_api_key)
        
        transcript = rag_system.get_youtube_transcript(request.video_id)
        rag_system.setup_vectorstore(transcript)
        
        return {
            "message": "RAG system setup successfully with YouTube video",
            "video_id": request.video_id,
            "transcript_length": len(transcript)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error setting up RAG system: {str(e)}")

@app.post("/setup/text")
async def setup_with_text(request: TextRequest):
    global rag_system
    
    try:
        rag_system = SimpleRAG(google_api_key=request.google_api_key)
        
        rag_system.setup_vectorstore(request.text)
        
        return {
            "message": "RAG system setup successfully with custom text",
            "text_length": len(request.text)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error setting up RAG system: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    global rag_system
    
    if rag_system is None:
        raise HTTPException(
            status_code=400, 
            detail="RAG system not initialized. Please setup first using /setup/youtube or /setup/text"
        )
    
    try:
        answer = rag_system.query(request.question)
        return QueryResponse(
            answer=answer,
            conversation_history=rag_system.get_memory()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying RAG system: {str(e)}")

@app.get("/memory")
async def get_memory():
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=400, detail="RAG system not initialized")
    
    return {
        "conversation_history": rag_system.get_memory(),
        "total_exchanges": len(rag_system.get_memory())
    }

@app.delete("/memory")
async def clear_memory():
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=400, detail="RAG system not initialized")
    
    rag_system.clear_memory()
    return {"message": "Conversation history cleared successfully"}

@app.get("/status")
async def get_status():
    global rag_system
    
    return {
        "rag_initialized": rag_system is not None,
        "has_vectorstore": rag_system.vectorstore is not None if rag_system else False,
        "memory_size": len(rag_system.get_memory()) if rag_system else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)