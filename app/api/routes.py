import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from ..core.config import settings
from ..core import rag
from ..models.schemas import (
    QuestionRequest,
    AnswerResponse,
    SourceChunk,
    DocumentInfo,
    UploadResponse,
    DeleteResponse,
)

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


@router.post("/ask", response_model=AnswerResponse)
def ask_question(body: QuestionRequest):
    try:
        result = rag.ask(body.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AnswerResponse(
        answer=result["answer"],
        sources=[SourceChunk(**s) for s in result["sources"]],
    )


@router.post("/documents/upload", response_model=UploadResponse, status_code=201)
def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=422, detail="Μη αποδεκτός τύπος αρχείου.")

    doc_id = str(uuid.uuid4())
    dest = Path(settings.upload_dir) / f"{doc_id}{suffix}"

    size = 0
    with dest.open("wb") as f:
        while chunk := file.file.read(65536):
            size += len(chunk)
            if size > settings.max_file_size_bytes:
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Το αρχείο είναι πολύ μεγάλο.")
            f.write(chunk)

    try:
        n = rag.ingest_document(str(dest), doc_id, file.filename)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        chunks_indexed=n,
        message=f"Αποθηκεύτηκε σε {n} τμήματα.",
    )


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents():
    return [DocumentInfo(**d) for d in rag.list_documents()]


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str):
    deleted = rag.delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Δεν βρέθηκε.")

    for ext in ALLOWED_EXTENSIONS:
        Path(settings.upload_dir, f"{doc_id}{ext}").unlink(missing_ok=True)

    return DeleteResponse(doc_id=doc_id, chunks_deleted=deleted, message="Διαγράφηκε.")
