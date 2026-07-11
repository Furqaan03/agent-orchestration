"""Long-term semantic memory: embed task learnings, retrieve for future planning.

Importance scoring, consolidation, and decay are deterministic and tested;
embedder is injected (ChromaDB in production)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Memory:
    id: str
    user_id: str
    content: str
    embedding: np.ndarray
    importance: float = 1.0     # grows with access
    access_count: int = 0
    created_at: float = 0.0


@dataclass
class LongTermMemory:
    embedder: Callable[[str], np.ndarray]
    memories: list[Memory] = field(default_factory=list)
    decay_rate: float = 0.01

    def store(self, memory_id: str, user_id: str, content: str, now: float) -> Memory:
        m = Memory(id=memory_id, user_id=user_id, content=content, embedding=self.embedder(content), created_at=now)
        self.memories.append(m)
        return m

    def retrieve(self, query: str, user_id: str, k: int = 3) -> list[Memory]:
        qvec = self.embedder(query)
        scored = []
        for m in self.memories:
            if m.user_id != user_id:
                continue
            denom = np.linalg.norm(qvec) * np.linalg.norm(m.embedding)
            sim = float(np.dot(qvec, m.embedding) / denom) if denom else 0.0
            # importance boosts frequently-accessed memories in ranking.
            scored.append((sim * m.importance, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [m for _, m in scored[:k]]
        for m in top:
            m.access_count += 1
            m.importance += 0.1   # accessed memories become more important
        return top

    def decay(self, now: float) -> None:
        """Stale, rarely-accessed memories lose importance over time."""
        for m in self.memories:
            age = now - m.created_at
            m.importance = max(0.1, m.importance - self.decay_rate * age * (1.0 / (1 + m.access_count)))

    def forget_user(self, user_id: str) -> int:
        """GDPR-style delete of all memories for a user."""
        before = len(self.memories)
        self.memories = [m for m in self.memories if m.user_id != user_id]
        return before - len(self.memories)
