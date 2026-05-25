"""ChromaDB-backed vector store for production use."""

from typing import Any

from ai_agent.core import get_logger, get_settings
from ai_agent.memory.manager import EmbeddingEngine, MemoryEntry, MemoryType

logger = get_logger(__name__)


class ChromaVectorStore:
    """Production vector store using ChromaDB."""

    def __init__(self, collection_name: str = "agent_memory") -> None:
        self._settings = get_settings()
        self._collection_name = collection_name
        self._client: Any = None
        self._collection: Any = None
        self._embedder = EmbeddingEngine()

    def _ensure_client(self) -> None:
        if self._client is None:
            try:
                import chromadb
                self._client = chromadb.HttpClient(
                    host=self._settings.chroma_host,
                    port=self._settings.chroma_port,
                )
                self._collection = self._client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                logger.warning("chromadb_unavailable", error=str(e))
                raise

    def add(self, entry: MemoryEntry) -> None:
        self._ensure_client()
        embedding = entry.embedding or self._embedder.embed(entry.content)
        self._collection.add(
            ids=[entry.id],
            embeddings=[embedding],
            documents=[entry.content],
            metadatas=[{
                "memory_type": entry.memory_type.value,
                "score": entry.score,
                "tags": ",".join(entry.tags),
                **{k: str(v) for k, v in entry.metadata.items()},
            }],
        )

    def search(self, query: str, limit: int = 5, where: dict[str, Any] | None = None) -> list[MemoryEntry]:
        self._ensure_client()
        query_embedding = self._embedder.embed(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
        )

        entries = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            entries.append(MemoryEntry(
                id=doc_id,
                content=results["documents"][0][i],
                memory_type=MemoryType(meta.get("memory_type", "long_term")),
                metadata=meta,
                tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
            ))
        return entries

    def delete(self, entry_id: str) -> None:
        self._ensure_client()
        self._collection.delete(ids=[entry_id])

    @property
    def count(self) -> int:
        self._ensure_client()
        return self._collection.count()
