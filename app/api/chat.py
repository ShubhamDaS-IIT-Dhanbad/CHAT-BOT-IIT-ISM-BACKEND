from fastapi import APIRouter, HTTPException, Depends
from app.models.chat import ChatRequest, ChatResponse
from app.services.langchain_service import LangChainService

router = APIRouter(prefix="/chat", tags=["chat"])

async def get_qa_chain():
    return LangChainService.get_qa_chain()

@router.get("/", response_model=ChatResponse)
async def chat(request: ChatRequest, qa_chain=Depends(get_qa_chain)):
    try:
        result = qa_chain.invoke({"query": request.query})
        return ChatResponse(
            message=result["result"],
            query=request.query,
            source_documents=[
                {"text": doc.page_content, "metadata": doc.metadata}
                for doc in result.get("source_documents", [])
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")