import pytest
from prism.embeddings import Embedder


class TestEmbedder:
    def test_embed_single_signature(self):
        embedder = Embedder()
        sig = {"symmetry_type": "broken", "dimensionality": "temporal", "self_reference": "false"}
        embedder.fit([sig])
        vec = embedder.embed(sig)
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_similar_signatures_have_closer_embeddings(self):
        embedder = Embedder()
        sig_a = {"symmetry_type": "broken", "dimensionality": "temporal", "boundary": "open"}
        sig_b = {"symmetry_type": "broken", "dimensionality": "temporal", "boundary": "closed"}
        sig_c = {"symmetry_type": "rotational", "dimensionality": "spatial", "boundary": "none"}
        embedder.fit([sig_a, sig_b, sig_c])
        vec_a = embedder.embed(sig_a)
        vec_b = embedder.embed(sig_b)
        vec_c = embedder.embed(sig_c)
        dist_ab = embedder.cosine_distance(vec_a, vec_b)
        dist_ac = embedder.cosine_distance(vec_a, vec_c)
        assert dist_ab < dist_ac  # a and b are more similar

    def test_embed_description_for_drop_mode(self):
        embedder = Embedder()
        sigs = [
            {"symmetry_type": "broken", "dimensionality": "temporal"},
            {"symmetry_type": "rotational", "dimensionality": "spatial"},
        ]
        embedder.fit(sigs)
        vec = embedder.embed_text("broken temporal symmetry")
        assert isinstance(vec, list)
        assert len(vec) > 0

    def test_refit_with_new_signatures(self):
        embedder = Embedder()
        embedder.fit([{"a": "1"}])
        dim1 = len(embedder.embed({"a": "1"}))
        embedder.fit([{"a": "1"}, {"b": "2", "c": "3"}])
        dim2 = len(embedder.embed({"a": "1"}))
        assert dim1 > 0
        assert dim2 > 0

    def test_compute_residual(self):
        sig_a = {"symmetry_type": "broken", "dimensionality": "temporal", "boundary": "open"}
        sig_b = {"symmetry_type": "broken", "dimensionality": "spatial", "boundary": "open"}
        residual = Embedder.compute_residual(sig_a, sig_b)
        assert "dimensionality" in residual
        assert residual["dimensionality"] == ("temporal", "spatial")
        assert "symmetry_type" not in residual
        assert "boundary" not in residual
