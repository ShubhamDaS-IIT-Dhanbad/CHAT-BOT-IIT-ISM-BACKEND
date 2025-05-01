from app.services.langchain_service import LangChainService

def get_qa_chain():
    return LangChainService.get_qa_chain()