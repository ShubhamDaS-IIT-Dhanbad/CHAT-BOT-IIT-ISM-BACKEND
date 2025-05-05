from pydantic import BaseModel
from typing import List, Dict, Any

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    message: str
    query: str
    source_documents: List[Dict[str, Any]]
    context_missing: bool