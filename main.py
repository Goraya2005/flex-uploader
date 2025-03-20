from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import shutil
import tempfile
import logging
import pickle

# ---------------------------
# 1. Configuration and Setup
# ---------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Configure CORS (Allow Frontend to Access Backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 50))  # Default 50MB
INDEX_FILE = "index.pkl"  # Persistent index storage

if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY is missing in environment variables!")
    raise RuntimeError("GOOGLE_API_KEY is required. Please set it in Render.com environment variables.")

# Initialize Google Generative AI LLM
try:
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    logger.info("Google Generative AI initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Google Generative AI: {e}")
    raise RuntimeError("Failed to initialize AI model.")

# Load or Initialize Index
def load_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "rb") as f:
            return pickle.load(f)
    return None

def save_index(index):
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(index, f)

index = load_index()

# ---------------------------
# 2. Define Pydantic Models
# ---------------------------

class QueryRequest(BaseModel):
    prompt: str
    output_format: str = "text"

# ---------------------------
# 3. API Endpoints
# ---------------------------

@app.get("/health/")
async def health_check():
    """Check if API is running."""
    return {"status": "API is running"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Upload file, process text, create embeddings, and build the index."""
    global index
    tmp_path = None
    
    try:
        # Check file size limit
        file_size = file.file.seek(0, 2)  # Get file size
        file.file.seek(0)  # Reset pointer
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE_MB}MB limit.")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        # Load file content
        loader = TextLoader(tmp_path)
        text = loader.load()
        if not text:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Create text splitter and embeddings
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

        # Create index
        index_creator = VectorstoreIndexCreator(embedding=embedding, text_splitter=text_splitter)
        index = index_creator.from_loaders([loader])

        # Save index
        save_index(index)

        return {"message": "File uploaded and index created successfully"}
    
    except Exception as e:
        logger.error(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/query/")
async def query_index(request: QueryRequest):
    """Process a query and return AI-generated response."""
    global index
    if index is None:
        raise HTTPException(status_code=400, detail="Index not initialized. Upload a file first.")

    try:
        response = index.query(request.prompt, llm=llm)
        return {"response": response}
    
    except Exception as e:
        logger.error(f"Query Error: {e}")
        raise HTTPException(status_code=500, detail="Error processing query.")
