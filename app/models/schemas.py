from pydantic import BaseModel
from typing import Optional


class QuestionRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    doc_id: str
    source_file: str
    page: Optional[int] = None
    excerpt: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class DocumentInfo(BaseModel):
    doc_id: str
    source_file: str


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_indexed: int
    message: str


class DeleteResponse(BaseModel):
    doc_id: str
    chunks_deleted: int
    message: str
