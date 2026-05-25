"""RAG (Retrieval-Augmented Generation) knowledge system."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ai_agent.core import get_logger, get_settings
from ai_agent.memory.manager import EmbeddingEngine, VectorStore, MemoryEntry, MemoryType

logger = get_logger(__name__)


class Document:
    """A chunked document for RAG."""
    __slots__ = ("id", "content", "metadata", "embedding")

    def __init__(self, content: str, metadata: dict[str, Any] | None = None):
        self.id = hashlib.md5(content.encode()).hexdigest()[:12]
        self.content = content
        self.metadata = metadata or {}
        self.embedding: list[float] | None = None


class RAGPipeline:
    """Ingest documents, chunk, embed, retrieve for augmented generation."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._settings = get_settings()
        self._persist_dir = persist_dir or self._settings.data_dir / "rag"
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = EmbeddingEngine()
        self._store = VectorStore()
        self._documents: list[Document] = []
        self._load()

    def ingest_file(self, path: str | Path, chunk_size: int = 500, overlap: int = 50) -> int:
        """Ingest a file into the RAG knowledge base."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = p.read_text(encoding="utf-8", errors="ignore")
        chunks = self._chunk_text(content, chunk_size, overlap)

        count = 0
        for i, chunk in enumerate(chunks):
            doc = Document(content=chunk, metadata={"source": str(p), "chunk": i, "filename": p.name})
            doc.embedding = self._embedder.embed(chunk)
            self._documents.append(doc)
            entry = MemoryEntry(
                id=doc.id, content=chunk, memory_type=MemoryType.SEMANTIC,
                embedding=doc.embedding, metadata=doc.metadata,
            )
            self._store.add(entry)
            count += 1

        self._persist()
        logger.info("rag_ingested", file=str(p), chunks=count)
        return count

    def ingest_directory(self, path: str | Path, extensions: list[str] | None = None) -> int:
        """Ingest all files in a directory."""
        p = Path(path).expanduser().resolve()
        exts = extensions or [".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".rs", ".go", ".js", ".ts"]
        ignore = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
        total = 0
        for fp in p.rglob("*"):
            if any(part in ignore for part in fp.parts):
                continue
            if fp.is_file() and fp.suffix in exts:
                try:
                    total += self.ingest_file(fp)
                except Exception as e:
                    logger.warning("rag_ingest_failed", file=str(fp), error=str(e))
        return total

    def query(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant documents for a question."""
        q_embedding = self._embedder.embed(question)
        results = self._store.search(q_embedding, limit=top_k, min_score=0.2)
        return [
            {"content": r.content, "source": r.metadata.get("source", ""), "score": r.score}
            for r in results
        ]

    def query_with_context(self, question: str, top_k: int = 5) -> str:
        """Get formatted context string for LLM augmentation."""
        results = self.query(question, top_k)
        if not results:
            return ""
        context_parts = []
        for i, r in enumerate(results, 1):
            source = Path(r["source"]).name if r["source"] else "unknown"
            context_parts.append(f"[{i}] ({source}):\n{r['content']}")
        return "\n\n".join(context_parts)

    @property
    def stats(self) -> dict[str, Any]:
        sources = set()
        for doc in self._documents:
            sources.add(doc.metadata.get("source", ""))
        return {"documents": len(self._documents), "sources": len(sources)}

    def clear(self) -> None:
        self._documents.clear()
        self._store = VectorStore()
        index_path = self._persist_dir / "rag_index.json"
        index_path.unlink(missing_ok=True)

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """Split text into overlapping chunks."""
        lines = text.splitlines(keepends=True)
        chunks = []
        current: list[str] = []
        current_len = 0

        for line in lines:
            current.append(line)
            current_len += len(line)
            if current_len >= chunk_size:
                chunks.append("".join(current))
                # Keep overlap
                overlap_lines: list[str] = []
                overlap_len = 0
                for l in reversed(current):
                    if overlap_len + len(l) > overlap:
                        break
                    overlap_lines.insert(0, l)
                    overlap_len += len(l)
                current = overlap_lines
                current_len = overlap_len

        if current:
            chunks.append("".join(current))
        return chunks

    def _persist(self) -> None:
        data = [{"id": d.id, "content": d.content, "metadata": d.metadata} for d in self._documents]
        (self._persist_dir / "rag_index.json").write_text(json.dumps(data))

    def _load(self) -> None:
        path = self._persist_dir / "rag_index.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for item in data:
                doc = Document(content=item["content"], metadata=item.get("metadata", {}))
                doc.id = item["id"]
                doc.embedding = self._embedder.embed(item["content"])
                self._documents.append(doc)
                entry = MemoryEntry(
                    id=doc.id, content=doc.content, memory_type=MemoryType.SEMANTIC,
                    embedding=doc.embedding, metadata=doc.metadata,
                )
                self._store.add(entry)
        except Exception as e:
            logger.warning("rag_load_failed", error=str(e))
