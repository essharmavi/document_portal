# ------------------------------
# IMPORTS
# ------------------------------
from __future__ import annotations  # allows forward references in type hints
import os
import sys
import shutil
import json
import uuid
import hashlib
import fitz   # PyMuPDF for reading PDF text
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Iterable

# LangChain imports for document processing
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Custom utility imports from your project
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from utils.file_io import _session_id, save_uploaded_files
from utils.document_ops import load_documents, concat_for_analysis, concat_for_comparison


# ------------------------------
# SUPPORTED FILE TYPES
# ------------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}  # Only allow these formats


# ===================================================================
# CLASS 1: FaissManager (Handles FAISS index creation & updates)
# ===================================================================
class FaissManager:
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        # Create directory for storing FAISS index
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file to track added docs (avoid duplicates)
        self.meta_path = self.index_dir / "ingested_data.json"
        self._meta: Dict[str, Any] = {"rows": {}}

        # Load metadata if it already exists
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}}
            except Exception:
                self._meta = {"rows": {}}

        # Initialize embedding model (used to convert text → vector)
        self.model_loader = model_loader or ModelLoader()
        self.embedding_model = self.model_loader.load_embeddings()

        # Placeholder for FAISS vector store
        self.vector_store: Optional[FAISS] = None

    def _exists(self) -> bool:
        """Check if FAISS index files already exist in directory"""
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    @staticmethod
    def _fingerprint( text: str, metadata: Dict[str, Any]) -> str:
        """Create a unique key for each document chunk to avoid duplicates"""
        src = metadata.get("source") or metadata.get("file_path")
        rid = metadata.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save_metadata(self):
        """Save metadata JSON (list of added documents)"""
        self.meta_path.write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add_documents(self, docs: List[Document]):
        """Add new documents to FAISS index"""
        if self.vector_store is None:
            raise RuntimeError("Call load_or_create() before adding documents")

        new_docs: List[Document] = []
        for doc in docs:
            # Generate unique fingerprint for this chunk
            key = self._fingerprint(text=doc.page_content, metadata=doc.metadata or {})

            # Skip if already exists
            if key in self._meta["rows"]:
                continue

            # Add to metadata & queue for indexing
            self._meta["rows"][key] = True
            new_docs.append(doc)

        # If there are new docs, index and save
        if new_docs:
            self.vector_store.add_documents(new_docs)
            self.vector_store.save_local(self.index_dir)
            self._save_metadata()

    def load_or_create(self, text: Optional[List[str]]= None, metadata: Optional[List[Dict]]=None):
        """Load existing FAISS index, or create new one if it doesn’t exist"""
        if self._exists():
            # Load existing FAISS index
            self.vector_store = FAISS.load_local(
                folder_path=str(self.index_dir),
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True,
            )
            return self.vector_store
        
        if not text:
            raise DocumentPortalException("No existing FAISS index. Need to create one", sys)
        
        # Create new FAISS index
        self.vector_store = FAISS.from_texts(
            texts=text,
            embedding=self.embedding_model,
            metadatas=metadata or []
        )
        self.vector_store.save_local(folder_path=str(self.index_dir))
        return self.vector_store



# ===================================================================
# CLASS 2: ChatIngestor (Handles uploading, splitting, indexing)
# ===================================================================
class ChatIngestor:
    def __init__(self, temp_base: str = "data/uploads", faiss_base: str = "faiss_index", use_session_dirs: bool = True, session_id: Optional[str] = None):
        try:
            self.logger = CustomLogger().get_logger(__name__)
            self.model_loader = ModelLoader()

            # Manage sessions
            self.use_session = use_session_dirs
            self.session_id = session_id or _session_id()

            # Setup directories
            self.temp_base = Path(temp_base)
            self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base)
            self.faiss_base.mkdir(parents=True, exist_ok=True)

            # Resolve session-specific directories
            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            self.logger.info(
                "Chat Ingestor initialized",
                session_id= session_id,
                temp_dir = str(self.temp_dir),
                faiss_dir = str(self.faiss_dir),
                sessionised = self.use_session
            )
        except Exception as e:
            self.logger.error ("Failed to initialize ChatIngestor", error=str(e))
            DocumentPortalException ("Initialization error in ChatIngestor", sys)

    def _resolve_dir(self, base: Path):
        """Return directory path (session-specific if enabled)"""
        if self.use_session:
            d = base / self.session_id
            d.mkdir(parents=True, exist_ok=True)
            return d
        return base

    def _split(self, docs: List[Document], chunk_size =1000, chunk_overlap = 200) -> List[Document]:
        """Split documents into smaller chunks for embeddings"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size= chunk_size,
            chunk_overlap= chunk_overlap
        )
        chunks = splitter.split_documents(documents=docs)
        self.logger.info("Document Split completed",chunk_size= chunk_size, chunk_overlap= chunk_overlap) 
        return chunks
    

    def build_retriever(
        self,
        uploaded_files: Iterable,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,
    ):
        """Build FAISS retriever from uploaded documents"""
        try:
            # STEP 1: Save uploaded files locally
            paths = save_uploaded_files(uploaded_files, self.temp_dir)

            # STEP 2: Load documents (convert to LangChain docs)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid document")

            # STEP 3: Split documents into chunks
            chunks = self._split(docs, chunk_overlap=chunk_overlap, chunk_size=chunk_size)

            # STEP 4: Initialize FAISS Manager
            fm = FaissManager(index_dir=self.faiss_dir, model_loader=self.model_loader)

            # STEP 5: Extract text + metadata for indexing
            texts = [c.page_content for c in chunks]
            meta_data = [c.metadata for c in chunks]

            # STEP 6: Load or create FAISS vector store
            vector_store  = fm.load_or_create(text=texts, metadata= meta_data)

            # STEP 7: Add new chunks to FAISS
            fm.add_documents(chunks)

            self.logger.info("FAISS index updated", index_path= str(self.faiss_dir))

            # STEP 8: Return retriever object (used to search documents)
            return vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k":k})

        except Exception as e:
            self.logger.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e

class DocHandler:
    def __init__(
        self, data_dir: Optional[str] = None, session_id: Optional[str] = None
    ):
        self.logger = CustomLogger().get_logger(__name__)
        self.data_dir = (
            data_dir
            or os.getenv("DATA_STORAGE_PATH")
            or os.path.join(os.getcwd(), "data", "document_analysis")
        )
        self.session_id = session_id or _session_id("session")
        self.session_path = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        self.logger.info(
            "DocHandler Initialized",
            session_id=self.session_id,
            session_path=self.session_path,
        )

    def save_pdf(self, uploaded_file) -> str:
        try:
            filename = os.path.basename(uploaded_file.name)
            if not filename.lower().endswith(".pdf"):
                raise ValueError("Invalid File type. Only PDFs are allowed")
            save_path = os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())

            self.logger.info(
                "PDF saved successfully",
                file=filename,
                save_path=save_path,
                session_id=self.session_id,
            )
            return save_path

        except Exception as e:
            self.logger.error(
                "Failed to save PDF", error=str(e), session_id=self.session_id
            )
            DocumentPortalException(f"Failed to save PDF: {str(e)}", sys)

    def read_pdf(self, pdf_path):
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text_chunks.append(
                        f"\n---Page {page_num + 1} ---\n{page.get_text()}"
                    )
                    text = "\n".join(text_chunks)
            # type: ignore
            self.logger.info(
                "PDF read successfully",
                pdf_path=pdf_path,
                session_id=self.session_id,
                pages=len(text_chunks),
            )
            return text
        except Exception as e:
            self.logger.error(
                "Failed to read PDF",
                error=str(e),
                pdf_path=pdf_path,
                session_id=self.session_id,
            )
            raise DocumentPortalException(f"Could not process PDF: {pdf_path}", sys)


class DocumentCompare:
    """
    Handles saving, reading, and comparing multiple documents (mainly PDFs).
    Stores each session in its own directory.
    """

    def __init__(self, base_dir="data/document_compare", session_id: Optional[str] = None):
        # Initialize logger
        self.logger = CustomLogger().get_logger(__name__)

        # Set base directory and session folder
        self.base_dir = Path(base_dir)
        self.session_id = session_id or _session_id()
        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)

        # Log initialization
        self.logger.info(
            "DocumentCompare Initialized",
            session_id=self.session_id,
            session_path=str(self.session_path),
        )

    def save_uploaded_files(self, reference_file, actual_file):
        """
        Save two uploaded PDFs (reference and actual) into session folder.
        Returns tuple of saved file paths.
        """
        try:
            ref_path = self.session_path / reference_file.name
            act_path = self.session_path / actual_file.name

            # Iterate over both files and save them
            for fobj, out in ((reference_file, ref_path), (actual_file, act_path)):
                if not fobj.name.lower().endswith(".pdf"):
                    raise ValueError("Only PDF files are allowed.")

                with open(out, "wb") as f:
                    if hasattr(fobj, "read"):  # handle file-like object
                        f.write(fobj.read())
                    else:  # handle memory buffer
                        f.write(fobj.getbuffer())

            # Log success
            self.logger.info("Files saved", reference=str(ref_path), actual=str(act_path), session=self.session_id)
            return ref_path, act_path

        except Exception as e:
            self.logger.error("Error saving PDF files", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error saving files", session=self.session_id) from e

    def read_pdf(self, pdf_path: Path):
        """
        Extract text content from a PDF file.
        Returns full text (page-wise).
        """
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError(f"PDF is encrypted: {pdf_path.name}")

                parts = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()

                    if text.strip():  # skip empty pages
                        parts.append(f"\n--- Page {page_num + 1} ---\n{text}")

                self.logger.info("PDF read successfully", file=str(pdf_path), pages=len(parts))
                return "\n".join(parts)

        except Exception as e:
            self.logger.error("Error reading PDF", file=str(pdf_path), error=str(e))
            raise DocumentPortalException("Error reading PDF", sys)

    def combine_documents(self):
        """
        Reads all PDFs in session folder and combines their text into a single string.
        Useful for comparing or indexing.
        """
        try:
            doc_parts = []

            # Collect all PDF content in folder
            for file in sorted(self.session_path.iterdir()):
                if file.is_file() and file.suffix.lower() == ".pdf":
                    content = self.read_pdf(file)
                    doc_parts.append(f"Document: {file.name}\n{content}")

            combined = "\n\n".join(doc_parts)

            self.logger.info("Documents combined", count=len(doc_parts), session=self.session_id)
            return combined

        except Exception as e:
            self.logger.error("Error combining documents", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error combining documents", sys)

    def clean_old_sessions(self, keep_latest: int = 3):
        """
        Keeps only the latest N session folders. Deletes older sessions.
        """
        try:
            sessions = sorted([f for f in self.base_dir.iterdir() if f.is_dir()], reverse=True)

            for folder in sessions[keep_latest:]:  # delete older sessions
                shutil.rmtree(folder, ignore_errors=True)
                self.logger.info("Old session folder deleted", path=str(folder))

        except Exception as e:
            self.logger.error("Error cleaning old sessions", error=str(e))
            raise DocumentPortalException("Error cleaning old sessions", sys)
