from __future__ import annotations
import os
import sys
import json
import uuid
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Dict, Any

# External dependencies
import fitz  # PyMuPDF (for working with PDFs, though not directly used here)
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,      # PDF ingestion
    Docx2txtLoader,   # Word DOCX ingestion
    TextLoader        # TXT ingestion
)
from langchain_community.vectorstores import FAISS  # Vector database for embeddings

# Internal utilities
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

# Setup logger for this module
log = CustomLogger().get_logger(__name__)

# Define supported file extensions for ingestion
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# ------------------------------------------------ #
# Document Loading
# ------------------------------------------------ #
def load_documents(paths: Iterable[Path]) -> List[Document]:
    """
    Load multiple documents from given file paths using appropriate loaders.

    Args:
        paths (Iterable[Path]): File paths to documents.

    Returns:
        List[Document]: A list of LangChain Document objects, each containing:
                        - `page_content`: the extracted text
                        - `metadata`: info about the file (like source path, page numbers)

    Raises:
        DocumentPortalException: If document loading fails at any point.
    """
    docs: List[Document] = []
    try:
        for p in paths:
            ext = p.suffix.lower()

            # Choose loader based on file extension
            if ext == ".pdf":
                loader = PyPDFLoader(str(p))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(p))
            elif ext == ".txt":
                loader = TextLoader(str(p), encoding="utf-8")
            else:
                log.warning("Unsupported extension skipped", path=str(p))
                continue

            # Load and add docs (PDF/DOCX may return multiple docs for multiple pages)
            docs.extend(loader.load())

        log.info("Documents loaded", count=len(docs))
        return docs

    except Exception as e:
        log.error("Failed loading documents", error=str(e))
        raise DocumentPortalException("Error loading documents", e) from e


# ------------------------------------------------ #
# Document Concatenation Helpers
# ------------------------------------------------ #
def concat_for_analysis(docs: List[Document]) -> str:
    """
    Concatenate documents into a single string for downstream analysis.

    Each document is prefixed with its SOURCE metadata so you can trace
    back where the content came from.

    Args:
        docs (List[Document]): LangChain Document objects.

    Returns:
        str: A formatted string combining all document text with metadata.
    """
    parts = []
    for d in docs:
        # Try multiple metadata fields for source info
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"

        # Format: Add source header + content
        parts.append(f"\n--- SOURCE: {src} ---\n{d.page_content}")

    return "\n".join(parts)


def concat_for_comparison(ref_docs: List[Document], act_docs: List[Document]) -> str:
    """
    Prepare a combined string to compare two sets of documents
    (e.g. reference vs actual).

    Args:
        ref_docs (List[Document]): Reference documents.
        act_docs (List[Document]): Actual documents.

    Returns:
        str: A structured string showing both sets clearly separated.
    """
    # Concatenate each group with source labels
    left = concat_for_analysis(ref_docs)
    right = concat_for_analysis(act_docs)

    # Wrap them with explicit section markers
    return f"<<REFERENCE_DOCUMENTS>>\n{left}\n\n<<ACTUAL_DOCUMENTS>>\n{right}"
