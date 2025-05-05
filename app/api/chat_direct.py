from fastapi import APIRouter, HTTPException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from app.core.config import settings

chat_direct_router = APIRouter(prefix="/testing", tags=["testing"])

# Request model
class ChatRequest(BaseModel):
    message: str

# System prompt message
prompt_message = (
    "In brief, you are a helpful and concise chatbot designed to assist student of iit ism dhanbad. "
    "Answer user questions clearly and efficiently."
)

# Route: post /direct
@chat_direct_router.post("/")
async def direct_chat(request: ChatRequest):
    try:
        # Initialize GPT-4o-mini model using LangChain
        llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model="gpt-3.5-turbo",
        )
        # Run chat completion with system and user messages
        response = llm.invoke([
            {"role": "system", "content": prompt_message},
            {"role": "user", "content": request.message}
        ])
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
