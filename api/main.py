# ------------------ IMPORTS ------------------

# Import FastAPI and useful classes for building APIs
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware   # Middleware for handling CORS
from fastapi.staticfiles import StaticFiles          # For serving static files (CSS, JS, images)
from fastapi.templating import Jinja2Templates       # For rendering HTML templates (Jinja2 engine)
from typing import Dict, Any, List, Optional         # Type hints for function return values
import os

# Import project-specific modules for document ingestion, analysis, comparison, and chat
from src.document_ingestion.data_ingestion import FaissManager, DocHandler, DocumentCompare, ChatIngestor
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_compare import DocumentComparerLLM
from src.document_chat.retrieval import ConversationalRag


# ------------------ APP CONFIG ------------------

# Environment variables → where data and FAISS indexes are stored
FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")

# Create FastAPI application instance with metadata
app = FastAPI(title="Document Portal API", version="0.1")

# Add CORS middleware → allows frontend (React/Angular/Vue) to talk to backend
# Without this, browser may block requests due to CORS policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # "*" means allow all domains (⚠️ in production use your frontend domain)
    allow_credentials=True,    # Allow cookies / auth headers
    allow_methods=["*"],       # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"]        # Allow all headers
)

# Mount static file directory so frontend assets (CSS, JS, images) can be served
# Example: http://localhost:8080/static/style.css
app.mount("/static", StaticFiles(directory="static"), name="static")


# Tell FastAPI where to look for HTML templates (e.g. index.html)
templates = Jinja2Templates(directory="templates")


# ------------------ HELPER CLASS ------------------
class FastAPIFileAdapter:
    """
    Adapter class to wrap FastAPI's UploadFile object 
    into a file-like object compatible with our ingestion pipeline.
    """

    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename
    def read(self) -> bytes:
        """Sync wrapper around UploadFile.file.read()."""
        self._uf.file.seek(0)  # Ensure reading from start
        return self._uf.file.read()

    def getbuffer(self) -> bytes:
        """Alias for read(), for compatibility."""
        return self.read()


# ------------------ ROUTES ------------------

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """
    Root route → serves the frontend HTML page (index.html).
    Useful if you’re building a UI along with backend.
    """
    response = templates.TemplateResponse("index.html", {"request": request})
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/health")
def health() -> Dict[str, str]:
    """
    Health check endpoint → lets you (or monitoring system) verify 
    if the API is running.
    """
    return {"status": "ok", "service": "document-portal"}


# ------------------ UTILITY FUNCTION ------------------

def _read_pdf_via_handler(handler: DocHandler, path: str) -> str:
    """
    Reads PDF content using a given DocHandler instance.

    - Checks if handler implements `read_pdf` (preferred) or `read_` (fallback).
    - Returns extracted text as a string.
    - Raises RuntimeError if neither method is available.
    """
    if hasattr(handler, "read_pdf"):
        return handler.read_pdf(path)  # type: ignore

    if hasattr(handler, "read_"):
        return handler.read_(path)  # type: ignore

    raise RuntimeError("DocHandler has neither read_pdf nor read_ method.")


# ------------------ DOCUMENT ANALYSIS ------------------

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Any:
    try:
        # Construct adapter
        wrapped = FastAPIFileAdapter(file)
        dh = DocHandler()
        saved_path = dh.save_pdf(wrapped)
        text = _read_pdf_via_handler(dh, saved_path)
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)
        
        return JSONResponse(content=result)

    except Exception as e:
        print("EXCEPTION CAUGHT:", repr(e))
        raise HTTPException(status_code=500, detail=f"Analysis Failed: {e}")


# ------------------ DOCUMENT COMPARISON ------------------

@app.post("/compare")
async def compare_document(reference: UploadFile = File(...), actual: UploadFile = File(...)) -> Any:
    """
    Upload two documents (reference + actual) and compare them.

    - Saves both documents to disk.
    - Combines their contents.
    - Uses LLM-powered DocumentComparerLLM to compare.
    - Returns structured comparison results as JSON.
    """
    try:
        dc = DocumentCompare()
        ref_path, actual_path = dc.save_uploaded_files(
            FastAPIFileAdapter(reference), FastAPIFileAdapter(actual)
        )

        # Load and merge document texts
        combined_text = dc.combine_documents()

        # Compare documents using LLM
        comp = DocumentComparerLLM()
        df = comp.compare_documents(combined_docs=combined_text)

        return {
            "rows": df.to_dict(orient="records"),
            "session_id": dc.session_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparing documents Failed: {e}")


# ------------------ CHAT (RAG SYSTEM) ------------------

@app.post("/chat/index")
async def chat_build_index(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5)
) -> Any:
    """
    Builds a FAISS vector index for uploaded documents (for RAG/chat).

    - Accepts one or multiple documents.
    - Splits text into chunks (configurable size + overlap).
    - Stores embeddings in FAISS index (per-session or global).
    - Prepares index for querying later.
    """
    try:
        wrapped = [FastAPIFileAdapter(f) for f in files]

        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None
        )

        ci.build_retriever(
            wrapped,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k
        )

        return {
            "session_id": session_id,
            "k": k,
            "use_session_dir": use_session_dirs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing Failed: {e}")


@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    use_session_dirs: bool = Form(True),
    session_id: Optional[str] = Form(None),
    k: int = Form(5)
) -> Any:
    """
    Query previously built FAISS index (RAG system).

    - Accepts a question as input.
    - Loads session-specific or global FAISS index.
    - Retrieves most relevant document chunks.
    - Uses ConversationalRag to generate an answer.
    - Returns the AI-generated answer.
    """
    try:
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="Session id is required if use_session_dirs is True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=464, detail=f"FAISS index not found at: {index_dir}")

        rag = ConversationalRag(session_id=session_id)
        rag.load_retriever_from_faiss(index_path=index_dir)

        response = rag.invoke(question, chat_history=[])

        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query Failed: {e}")
