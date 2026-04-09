import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder:
    def __init__(self):
        self._vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"[a-zA-Z0-9_]+")
        self._fitted = False

    def _serialize_signature(self, sig: dict[str, str]) -> str:
        return " ".join(f"{k} {v}" for k, v in sorted(sig.items()))

    def fit(self, signatures: list[dict[str, str]]):
        texts = [self._serialize_signature(s) for s in signatures]
        self._vectorizer.fit(texts)
        self._fitted = True

    def embed(self, signature: dict[str, str]) -> list[float]:
        if not self._fitted:
            raise RuntimeError("Embedder must be fit before embedding")
        text = self._serialize_signature(signature)
        vec = self._vectorizer.transform([text]).toarray()[0]
        return vec.tolist()

    def embed_text(self, text: str) -> list[float]:
        if not self._fitted:
            raise RuntimeError("Embedder must be fit before embedding")
        vec = self._vectorizer.transform([text]).toarray()[0]
        return vec.tolist()

    @staticmethod
    def cosine_distance(a: list[float], b: list[float]) -> float:
        a_arr = np.array(a)
        b_arr = np.array(b)
        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - (dot / (norm_a * norm_b))

    @staticmethod
    def compute_residual(sig_a: dict[str, str], sig_b: dict[str, str]) -> dict[str, tuple[str, str]]:
        all_keys = set(sig_a.keys()) | set(sig_b.keys())
        residual = {}
        for key in all_keys:
            val_a = sig_a.get(key)
            val_b = sig_b.get(key)
            if val_a != val_b:
                residual[key] = (val_a or "(absent)", val_b or "(absent)")
        return residual
