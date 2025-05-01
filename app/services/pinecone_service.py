from pinecone import Pinecone as PineconeClient
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings

class PineconeService:
    _vectorstore = None

    @classmethod
    def get_vectorstore(cls):
        if cls._vectorstore is None:
            # Initialize Pinecone client
            pc = PineconeClient(api_key=settings.pinecone_api_key)
            index = pc.Index(settings.pinecone_index_name)

            # Setup embeddings and vector store
            embed = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
            cls._vectorstore = PineconeVectorStore(index, embed, text_key="text")
        return cls._vectorstore

    @classmethod
    def get_retriever(cls, search_kwargs=None):
        search_kwargs = search_kwargs or {"k": 4}
        return cls.get_vectorstore().as_retriever(search_kwargs=search_kwargs)