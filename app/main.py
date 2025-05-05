from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import router
from app.api.chat_direct import chat_direct_router
from app.core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate settings at startup
    if not all([settings.openai_api_key, settings.pinecone_api_key, settings.pinecone_env]):
        raise ValueError("Missing required environment variables")
    yield
    # Cleanup if needed

app = FastAPI(lifespan=lifespan)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://127.0.0.1:8000","https://your-frontend-domain.com"],  # Adjust for your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(chat_direct_router)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}
