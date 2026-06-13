from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from .config import settings

# lazy load για να μην αργεί η εκκίνηση
_embeddings = None
_vectorstore = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key,
        )
    return _embeddings


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=get_embeddings(),
            collection_name="study_docs",
        )
    return _vectorstore


def ingest_document(file_path: str, doc_id: str, original_filename: str = None) -> int:
    p = Path(file_path)

    if p.suffix.lower() == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()
    for d in docs:
        d.metadata["doc_id"] = doc_id
        d.metadata["source_file"] = original_filename or p.name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    get_vectorstore().add_documents(chunks)
    return len(chunks)


def delete_document(doc_id: str) -> int:
    vs = get_vectorstore()
    res = vs.get(where={"doc_id": doc_id})
    ids = res.get("ids", [])
    if ids:
        vs.delete(ids=ids)
    return len(ids)


def list_documents():
    vs = get_vectorstore()
    res = vs.get(include=["metadatas"])
    seen = {}
    for meta in res.get("metadatas", []):
        did = meta.get("doc_id", "unknown")
        if did not in seen:
            seen[did] = {"doc_id": did, "source_file": meta.get("source_file", "")}
    return list(seen.values())


_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Είσαι βοηθός μελέτης. Απάντησε βασιζόμενος ΜΟΝΟ στα παρακάτω κείμενα.
Αν δεν βρίσκεις την απάντηση, πες ότι δεν έχεις αρκετές πληροφορίες.

Κείμενα:
{context}

Ερώτηση: {question}

Απάντηση:""",
)


def ask(question: str) -> dict:
    vs = get_vectorstore()

    llm = ChatOpenAI(
        model=settings.openai_model,
        openai_api_key=settings.openai_api_key,
        max_tokens=2048,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vs.as_retriever(search_kwargs={"k": settings.top_k_results}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": _prompt},
    )

    result = chain.invoke({"query": question})

    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "doc_id": doc.metadata.get("doc_id", ""),
            "source_file": doc.metadata.get("source_file", ""),
            "page": doc.metadata.get("page"),
            "excerpt": doc.page_content[:200],
        })

    return {"answer": result["result"], "sources": sources}
