from __future__ import annotations
import uuid

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Iterable, List, Optional, Dict, Any

from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

# Global logger instance for this module
log = CustomLogger().get_logger(__name__)

# Allowed file formats for ingestion
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# ----------------------------- #
# Helpers (file I/O + loading)  #
# ----------------------------- #

def _session_id(prefix: str = "session") -> str:
    """
    Generate a unique session ID string using:
      - prefix (e.g. 'session')
      - current UTC datetime (YYYYMMDD_HHMMSS format)
      - random UUID hex suffix (8 chars)

    Example: session_20250821_103512_ab12cd34
    """
    ist = ZoneInfo("Asia/Kolkata")
    return f"{prefix}_{datetime.now(ist).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def save_uploaded_files(uploaded_files: Iterable, target_dir: Path) -> List[Path]:
    """
    Save uploaded files to a local directory.

    Args:
        uploaded_files (Iterable): Iterable of uploaded files (could be from Streamlit, FastAPI, etc.).
                                   Each item should have `.name` and either `.read()` or `.getbuffer()`.
        target_dir (Path): Directory where files will be saved.

    Returns:
        List[Path]: List of saved file paths.

    Raises:
        DocumentPortalException: If saving fails at any point.
    """
    try:
        # Ensure target directory exists
        target_dir.mkdir(parents=True, exist_ok=True)

        saved: List[Path] = []

        # Process each uploaded file
        for uf in uploaded_files:
            # Get original filename and extension
            name = getattr(uf, "name", "file")
            ext = Path(name).suffix.lower()

            # Skip unsupported extensions
            if ext not in SUPPORTED_EXTENSIONS:
                log.warning("Unsupported file skipped", filename=name)
                continue

            # Create a unique filename (random UUID-based) to avoid collisions
            fname = f"{uuid.uuid4().hex[:8]}{ext}"
            out = target_dir / fname

            # Write file content to disk
            with open(out, "wb") as f:
                if hasattr(uf, "read"):
                    f.write(uf.read())  # typical file-like object
                else:
                    f.write(uf.getbuffer())  # e.g., from Streamlit upload

            # Track saved file
            saved.append(out)

            # Log successful save
            log.info("File saved for ingestion", uploaded=name, saved_as=str(out))

        return saved

    except Exception as e:
        # Log + raise wrapped exception for better error traceability
        log.error("Failed to save uploaded files", error=str(e), dir=str(target_dir))
        raise DocumentPortalException("Failed to save uploaded files", e) from e
