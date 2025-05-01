from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from app.core.config import settings
from app.services.pinecone_service import PineconeService

class LangChainService:
    _qa_chain = None

    @classmethod
    def get_qa_chain(cls):
        if cls._qa_chain is None:
            # Initialize LLM
            llm = ChatOpenAI(
                openai_api_key=settings.openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.7,
            )
            # Get retriever from Pinecone
            retriever = PineconeService.get_retriever()
            # Setup RAG chain
            cls._qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )
        return cls._qa_chain