from pinecone import Pinecone as PineconeClient
from pinecone.openapi_support.exceptions import NotFoundException
from openai import OpenAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from app.core.config import settings
from pydantic import Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIEmbedder:
    def __init__(self, api_key, model="text-embedding-3-small", dimension=512):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimension = dimension

    def embed_query(self, text):
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model,
                dimensions=self.dimension
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            raise ValueError(f"Embedding failed: {str(e)}")

class PineconeService:
    _vectorstore = None
    _embedder = OpenAIEmbedder(api_key=settings.openai_api_key, model="text-embedding-3-small", dimension=512)

    @classmethod
    def get_vectorstore(cls):
        if cls._vectorstore is None:
            try:
                pc = PineconeClient(api_key=settings.pinecone_api_key)
                index = pc.Index(settings.pinecone_index_name)
                cls._vectorstore = index
                logger.info(f"Initialized Pinecone index: {settings.pinecone_index_name}")
            except NotFoundException as e:
                logger.error(f"Pinecone index '{settings.pinecone_index_name}' not found: {str(e)}")
                raise ValueError(f"Pinecone index '{settings.pinecone_index_name}' not found. Verify the index name or create it.")
            except Exception as e:
                logger.error(f"Pinecone initialization failed: {str(e)}")
                raise ValueError(f"Pinecone initialization failed: {str(e)}")
        return cls._vectorstore

    @classmethod
    def get_retriever(cls, search_kwargs=None):
        index = cls.get_vectorstore()
        k = (search_kwargs or {}).get("k", 4)

        class PineconeRetriever(BaseRetriever):
            index: object = Field(...)
            embedder: OpenAIEmbedder = Field(...)
            k: int = Field(...)

            def __init__(self, index, embedder, k):
                super().__init__(index=index, embedder=embedder, k=k)

            def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
                try:
                    vector = self.embedder.embed_query(query)
                    result = self.index.query(vector=vector, top_k=self.k, include_metadata=True)
                    return [
                        Document(
                            page_content=match["metadata"].get("text", ""),
                            metadata=match["metadata"]
                        )
                        for match in result["matches"]
                    ]
                except Exception as e:
                    logger.error(f"Document retrieval failed: {str(e)}")
                    raise ValueError(f"Document retrieval failed: {str(e)}")

            async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
                try:
                    vector = self.embedder.embed_query(query)
                    result = self.index.query(vector=vector, top_k=self.k, include_metadata=True)
                    return [
                        Document(
                            page_content=match["metadata"].get("text", ""),
                            metadata=match["metadata"]
                        )
                        for match in result["matches"]
                    ]
                except Exception as e:
                    logger.error(f"Async document retrieval failed: {str(e)}")
                    raise ValueError(f"Async document retrieval failed: {str(e)}")

        return PineconeRetriever(index, cls._embedder, k)