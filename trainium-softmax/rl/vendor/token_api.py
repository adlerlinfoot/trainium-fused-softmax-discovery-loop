from dataclasses import dataclass
from types import SimpleNamespace
@dataclass
class EncodedTextChunk:
    tokens: list[int]
@dataclass
class ModelInput:
    chunks: list[EncodedTextChunk]
    @property
    def length(self): return sum(len(c.tokens) for c in self.chunks)
@dataclass
class SamplingParams:
    stop: list
    max_tokens: int
    temperature: float = 1.0
types=SimpleNamespace(EncodedTextChunk=EncodedTextChunk)
