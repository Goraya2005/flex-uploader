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
import uvicorn

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

# Configure CORS to allow requests from Vercel frontend
ALLOWED_ORIGINS = [
    "https://data-flex-psi.vercel.app",  # ✅ Your deployed frontend
    "http://localhost:3000",             # ✅ Local dev frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PORT = int(os.getenv("PORT", 8000))  # Render assigns this in deployment

if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY is missing!")
    raise RuntimeError("Please set GOOGLE_API_KEY in environment variables.")

# Initialize Gemini Pro LLM
try:
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    logger.info("Google Generative AI initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Google Generative AI: {e}")
    raise RuntimeError("LLM initialization failed.")

# Global index variable (in-memory)
index = None

# ---------------------------
# 2. Pydantic Models
# ---------------------------

class QueryRequest(BaseModel):
    prompt: str
    output_format: str = "text"

# ---------------------------
# 3. API Endpoints
# ---------------------------

@app.get("/health/")
async def health_check():
    return {"status": "API is running"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a file, reads its content, splits it, embeds it, and creates an in-memory index.
    """
    global index
    tmp_path = None

    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Load file using TextLoader
        loader = TextLoader(tmp_path)
        docs = loader.load()

        if not docs or len(docs) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty or unreadable.")

        # Create text splitter and embedding model
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

        # Create index from document loader
        index_creator = VectorstoreIndexCreator(embedding=embedding, text_splitter=text_splitter)
        index = index_creator.from_loaders([loader])

        logger.info("Index created and stored in memory.")

        return {"message": "File uploaded and index created successfully"}

    except Exception as e:
        logger.error(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/query/")
async def query_index(request: QueryRequest):
    """
    Accepts a prompt, queries the in-memory index, and returns the response.
    """
    global index
    if index is None:
        raise HTTPException(status_code=400, detail="Index not initialized. Upload a file first.")

    try:
        response = index.query(request.prompt, llm=llm)
        return {"response": response}
    except Exception as e:
        logger.error(f"Query Error: {e}")
        raise HTTPException(status_code=500, detail="Error processing query.")

# ---------------------------
# 4. Run on Render (0.0.0.0 + $PORT)
# ---------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
