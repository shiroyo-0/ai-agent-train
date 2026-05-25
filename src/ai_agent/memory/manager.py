"""Memory system - short-term, long-term, episodic, and vector memory."""

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from ai_agent.core import get_logger, get_settings

logger = get_logger(__name__)


class MemoryType(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class MemoryEntry:
    id: str
    content: str
    memory_type: MemoryType
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 1.0
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)


class EmbeddingEngine:
    """Generate embeddings using sentence-transformers."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or get_settings().embedding_model
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
            except ImportError:
                logger.warning("sentence-transformers not available, using hash-based embeddings")
                self._model = "fallback"
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._load_model()
        if model == "fallback":
            return self._hash_embed(text)
        return model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        if model == "fallback":
            return [self._hash_embed(t) for t in texts]
        return model.encode(texts, normalize_embeddings=True).tolist()

    @staticmethod
    def _hash_embed(text: str, dim: int = 384) -> list[float]:
        """Deterministic fallback embedding using hashing."""
        h = hashlib.sha512(text.encode()).digest()
        np.random.seed(int.from_bytes(h[:4], "big"))
        return np.random.randn(dim).astype(np.float32).tolist()


class VectorStore:
    """In-memory vector store with cosine similarity search."""

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)

    def search(self, query_embedding: list[float], limit: int = 5, min_score: float = 0.0) -> list[MemoryEntry]:
        if not self._entries or not query_embedding:
            return []

        q = np.array(query_embedding)
        scored = []
        for entry in self._entries:
            if entry.embedding is None:
                continue
            e = np.array(entry.embedding)
            sim = float(np.dot(q, e) / (np.linalg.norm(q) * np.linalg.norm(e) + 1e-8))
            if sim >= min_score:
                scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    def remove(self, entry_id: str) -> bool:
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.id != entry_id]
        return len(self._entries) < before

    @property
    def size(self) -> int:
        return len(self._entries)


class MemoryManager:
    """Unified memory system with short-term, long-term, and episodic memory."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._settings = get_settings()
        self._persist_dir = persist_dir or self._settings.data_dir / "memory"
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = EmbeddingEngine()
        self._vector_store = VectorStore()
        self._short_term: list[MemoryEntry] = []
        self._max_short_term = 50
        self._load_persisted()

    def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.LONG_TERM,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Store a memory entry."""
        entry_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:12]
        embedding = self._embedder.embed(content)

        entry = MemoryEntry(
            id=entry_id,
            content=content,
            memory_type=memory_type,
            embedding=embedding,
            metadata=metadata or {},
            tags=tags or [],
        )

        if memory_type == MemoryType.SHORT_TERM:
            self._short_term.append(entry)
            if len(self._short_term) > self._max_short_term:
                # Promote oldest to long-term or discard
                oldest = self._short_term.pop(0)
                if oldest.access_count > 2:
                    oldest.memory_type = MemoryType.LONG_TERM
                    self._vector_store.add(oldest)
        else:
            self._vector_store.add(entry)

        return entry_id

    def recall(self, query: str, limit: int = 5, memory_type: MemoryType | None = None) -> list[MemoryEntry]:
        """Retrieve relevant memories using semantic search."""
        query_embedding = self._embedder.embed(query)
        results = self._vector_store.search(query_embedding, limit=limit * 2)

        # Also check short-term memory
        if not memory_type or memory_type == MemoryType.SHORT_TERM:
            for entry in self._short_term:
                if entry.embedding:
                    q = np.array(query_embedding)
                    e = np.array(entry.embedding)
                    sim = float(np.dot(q, e) / (np.linalg.norm(q) * np.linalg.norm(e) + 1e-8))
                    if sim > 0.3:
                        results.append(entry)

        # Filter by type if specified
        if memory_type:
            results = [r for r in results if r.memory_type == memory_type]

        # Score and rank
        for entry in results:
            entry.access_count += 1
            entry.last_accessed = time.time()
            # Decay score based on age
            age_hours = (time.time() - entry.created_at) / 3600
            entry.score = max(0.1, 1.0 - (age_hours * 0.01)) * (1 + entry.access_count * 0.1)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def store_episode(self, task: str, actions: list[str], outcome: str, success: bool) -> str:
        """Store an episodic memory (task execution record)."""
        content = f"Task: {task}\nActions: {json.dumps(actions)}\nOutcome: {outcome}\nSuccess: {success}"
        return self.store(
            content,
            memory_type=MemoryType.EPISODIC,
            metadata={"task": task, "success": success, "action_count": len(actions)},
            tags=["episode", "success" if success else "failure"],
        )

    def compress(self, entries: list[MemoryEntry]) -> str:
        """Compress multiple memories into a summary."""
        contents = [e.content for e in entries]
        return f"[Compressed {len(entries)} memories]\n" + "\n---\n".join(contents[:5])

    def persist(self) -> None:
        """Save memory state to disk."""
        data = {
            "entries": [
                {
                    "id": e.id, "content": e.content, "memory_type": e.memory_type.value,
                    "metadata": e.metadata, "score": e.score, "access_count": e.access_count,
                    "created_at": e.created_at, "last_accessed": e.last_accessed, "tags": e.tags,
                }
                for e in self._vector_store._entries + self._short_term
            ]
        }
        (self._persist_dir / "memory.json").write_text(json.dumps(data, indent=2))

    def _load_persisted(self) -> None:
        """Load persisted memory."""
        path = self._persist_dir / "memory.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for item in data.get("entries", []):
                entry = MemoryEntry(
                    id=item["id"],
                    content=item["content"],
                    memory_type=MemoryType(item["memory_type"]),
                    embedding=self._embedder.embed(item["content"]),
                    metadata=item.get("metadata", {}),
                    score=item.get("score", 1.0),
                    access_count=item.get("access_count", 0),
                    created_at=item.get("created_at", time.time()),
                    last_accessed=item.get("last_accessed", time.time()),
                    tags=item.get("tags", []),
                )
                if entry.memory_type == MemoryType.SHORT_TERM:
                    self._short_term.append(entry)
                else:
                    self._vector_store.add(entry)
        except Exception as e:
            logger.warning("failed_to_load_memory", error=str(e))

    @property
    def stats(self) -> dict[str, int]:
        return {
            "short_term": len(self._short_term),
            "long_term": self._vector_store.size,
            "total": len(self._short_term) + self._vector_store.size,
        }
