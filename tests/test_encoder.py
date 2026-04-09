# tests/test_encoder.py
import pytest
import tempfile
import os
from prism.db import PrismDB
from prism.models import Pattern
from prism.embeddings import Embedder
from prism.resonance import ResonanceEngine
from prism.encoder import Encoder


@pytest.fixture
def env():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = PrismDB(path)
    embedder = Embedder()
    engine = ResonanceEngine(db, embedder)
    encoder = Encoder(db, engine)
    db.insert_pattern(Pattern(id="recursion", name="Recursion", description="Structure that contains itself"))
    db.insert_pattern(Pattern(id="symmetry-breaking", name="Symmetry Breaking", description="Uniform becomes differentiated"))
    yield encoder, db
    db.close()
    os.unlink(path)


class TestEncoder:
    def test_encode_instance(self, env):
        encoder, db = env
        inst = encoder.encode(
            instance_id="test-1",
            domain="grammar",
            description="Irregular verbs resist regularization",
            structural_signature={"symmetry_type": "broken", "dimensionality": "temporal"},
            pattern_ids=["symmetry-breaking"],
            rationale="Fundamental structures resist simplification over time",
        )
        assert inst.id == "test-1"
        stored = db.get_instance("test-1")
        assert stored.domain == "grammar"
        assert stored.encoding_rationale == "Fundamental structures resist simplification over time"

    def test_encode_returns_speculative_links(self, env):
        encoder, db = env
        encoder.encode(
            instance_id="a",
            domain="grammar",
            description="test a",
            structural_signature={"symmetry_type": "broken", "dimensionality": "temporal"},
            pattern_ids=["symmetry-breaking"],
        )
        inst, links = encoder.encode_with_links(
            instance_id="b",
            domain="physics",
            description="test b",
            structural_signature={"symmetry_type": "broken", "dimensionality": "temporal"},
            pattern_ids=["symmetry-breaking"],
        )
        assert isinstance(links, list)

    def test_encode_validates_pattern_exists(self, env):
        encoder, db = env
        with pytest.raises(KeyError, match="Pattern not found"):
            encoder.encode(
                instance_id="test",
                domain="grammar",
                description="test",
                structural_signature={"x": "1"},
                pattern_ids=["nonexistent-pattern"],
            )
