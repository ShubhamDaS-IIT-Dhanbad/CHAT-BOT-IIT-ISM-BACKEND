from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user query")

class ChatResponse(BaseModel):
    message: str
    query: str
    source_documents: list[dict] | None = None