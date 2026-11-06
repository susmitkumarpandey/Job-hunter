import io
from fastapi import HTTPException, UploadFile, status
import pdfplumber
import docx


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts raw text from a PDF byte stream."""
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extracts raw text from a DOCX byte stream."""
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


def extract_resume_text(file: UploadFile, content: bytes) -> str:
    """
    Validates file format and extracts clean plain text.
    Supports PDF, DOCX, and TXT files.
    """
    filename = (file.filename or "").lower()
    content_type = file.content_type or ""

    if filename.endswith(".pdf") or content_type == "application/pdf":
        text = extract_text_from_pdf(content)
    elif filename.endswith(".docx") or "wordprocessingml" in content_type:
        text = extract_text_from_docx(content)
    elif filename.endswith(".txt") or content_type == "text/plain":
        text = content.decode("utf-8", errors="ignore").strip()
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload a .pdf, .docx, or .txt file.",
        )

    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The uploaded document contains no readable text.",
        )

    return text