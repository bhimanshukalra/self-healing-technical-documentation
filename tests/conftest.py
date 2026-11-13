import hashlib
import math
import re

import pytest


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder; texts sharing words are similar."""

    dim = 256

    def __init__(self):
        self.calls: list[int] = []

    def embed(self, texts):
        self.calls.append(len(texts))
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in re.findall(r"[a-z]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


@pytest.fixture
def embedder():
    return FakeEmbedder()
