
import os
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
class Config:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    ALLOWED_FILE_TYPES = ["pdf"]
    CHUNK_SIZE = 5000
    CHUNK_OVERLAP = 200
    GEMINI_MODEL = "gemini-2.0-flash"
    EMBEDDINGS_MODEL = "models/embedding-001"

# Validate configuration
if not Config.GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Application state management
class AppState:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDINGS_MODEL,
            google_api_key=Config.GOOGLE_API_KEY
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.docs: List[Document] = []
        self.final_documents: List[Document] = []
        self.vectors = None
        self.processed_files: List[str] = []

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.app_state = AppState()
    logger.info("Application startup complete")
    yield
    logger.info("Application shutdown initiated")

# Create FastAPI app
app = FastAPI(
    title="PDF QA ChatBot API (Gemini)",
    description="API for PDF-based question answering using Google Gemini",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=Config.GEMINI_MODEL,
    google_api_key=Config.GOOGLE_API_KEY,
    temperature=0.3
)

# Prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
Answer the questions based strictly on the provided context.
Follow these rules:
1. Be concise and accurate
2. If the answer isn't in the context, say "I couldn't find this information in the documents"
3. Never hallucinate or make up information

Context:
{context}

Question: {input}
"""
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class FileProcessResponse(BaseModel):
    filename: str
    status: str
    preview: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str
    status: str
    sources: Optional[List[Dict[str, Any]]] = None

class ProcessedFilesResponse(BaseModel):
    files: List[str]
    count: int

class HealthCheckResponse(BaseModel):
    status: str
    model: str
    processed_files: int

# Helpers
def validate_file_type(filename: str) -> bool:
    extension = filename.split(".")[-1].lower()
    return extension in Config.ALLOWED_FILE_TYPES

async def process_pdf(file: UploadFile) -> List[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_file_path)
        return loader.load()
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# API Endpoints
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return {
        "status": "healthy",
        "model": Config.GEMINI_MODEL,
        "processed_files": len(app.state.app_state.processed_files)
    }

@app.post("/upload/", response_model=FileProcessResponse)
async def process_file(file: UploadFile = File(...)):
    try:
        if not validate_file_type(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Allowed types: {', '.join(Config.ALLOWED_FILE_TYPES)}"
            )

        logger.info(f"Processing PDF file: {file.filename}")
        docs = await process_pdf(file)
        file_preview = "\n".join([doc.page_content[:200] for doc in docs])

        final_documents = app.state.app_state.text_splitter.split_documents(docs)
        app.state.app_state.docs.extend(docs)
        app.state.app_state.final_documents.extend(final_documents)

        if app.state.app_state.vectors is None:
            app.state.app_state.vectors = FAISS.from_documents(
                app.state.app_state.final_documents,
                app.state.app_state.embeddings
            )
        else:
            app.state.app_state.vectors.add_documents(final_documents)

        app.state.app_state.processed_files.append(file.filename)

        return FileProcessResponse(
            filename=file.filename,
            status="success",
            preview=file_preview,
            message="PDF processed successfully"
        )

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        return FileProcessResponse(
            filename=file.filename,
            status="error",
            error=str(e),
            message="Failed to process PDF"
        )

@app.post("/ask/", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        if not app.state.app_state.vectors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No PDF files have been processed yet"
            )

        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = app.state.app_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({"input": request.question})
        
        return AnswerResponse(
            answer=response['answer'],
            status="success",
            sources=[{"page_content": doc.page_content[:200]} for doc in response.get("context", [])]
        )

    except Exception as e:
        logger.error(f"Error answering question: {str(e)}", exc_info=True)
        return AnswerResponse(
            answer="Please upload PDF files first.",
            status="error"
        )

@app.get("/files/", response_model=ProcessedFilesResponse)
async def get_processed_files():
    return ProcessedFilesResponse(
        files=app.state.app_state.processed_files,
        count=len(app.state.app_state.processed_files)
    )

@app.post("/reset/")
async def reset_session():
    try:
        app.state.app_state.docs = []
        app.state.app_state.final_documents = []
        app.state.app_state.vectors = None
        app.state.app_state.processed_files = []
        return {"status": "success", "message": "Session reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting session: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset session"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test:app", host="0.0.0.0", port=8001, reload=True)
