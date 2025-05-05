from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from app.core.config import settings
from app.services.pinecone_service import PineconeService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainService:
    _qa_chain = None

    @classmethod
    def get_qa_chain(cls):
        if cls._qa_chain is None:
            try:
                llm = ChatOpenAI(
                    openai_api_key=settings.openai_api_key,
                    model="gpt-3.5-turbo",
                    temperature=0.7,
                )
                retriever = PineconeService.get_retriever(search_kwargs={"k": 4})
                cls._qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                )
                logger.info("Initialized RetrievalQA chain")
            except Exception as e:
                logger.error(f"QA chain initialization failed: {str(e)}")
                raise ValueError(f"QA chain initialization failed: {str(e)}")
        return cls._qa_chain