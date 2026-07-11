import hashlib

import numpy as np

from src.memory.longterm import LongTermMemory
from src.memory.working import WorkingMemory


def _fake_embedder(dim=32):
    def embed(text):
        vec = np.zeros(dim)
        for tok in text.lower().split():
            vec[int(hashlib.sha256(tok.encode()).hexdigest()[:6], 16) % dim] += 1
        n = np.linalg.norm(vec)
        return vec / n if n else vec
    return embed


def test_working_memory_scoped_and_cleared():
    wm = WorkingMemory()
    wm.record_output("a", "result-a")
    assert wm.get_output("a") == "result-a"
    wm.clear()
    assert wm.get_output("a") is None


def test_longterm_retrieval_by_similarity():
    ltm = LongTermMemory(embedder=_fake_embedder())
    ltm.store("m1", "user1", "how to reset a password", now=0.0)
    ltm.store("m2", "user1", "recipe for chocolate cake", now=0.0)
    results = ltm.retrieve("password reset help", "user1", k=1)
    assert results[0].id == "m1"


def test_longterm_user_isolation():
    ltm = LongTermMemory(embedder=_fake_embedder())
    ltm.store("m1", "user1", "user one secret", now=0.0)
    ltm.store("m2", "user2", "user two secret", now=0.0)
    results = ltm.retrieve("secret", "user1")
    assert all(m.user_id == "user1" for m in results)


def test_access_increases_importance():
    ltm = LongTermMemory(embedder=_fake_embedder())
    m = ltm.store("m1", "user1", "important fact", now=0.0)
    before = m.importance
    ltm.retrieve("important fact", "user1")
    assert m.importance > before


def test_forget_user_deletes_memories():
    ltm = LongTermMemory(embedder=_fake_embedder())
    ltm.store("m1", "user1", "x", now=0.0)
    ltm.store("m2", "user2", "y", now=0.0)
    removed = ltm.forget_user("user1")
    assert removed == 1
    assert all(m.user_id != "user1" for m in ltm.memories)
