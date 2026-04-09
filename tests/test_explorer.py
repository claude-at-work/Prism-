# tests/test_explorer.py
import pytest
import tempfile
import os
from prism.db import PrismDB
from prism.models import Pattern, Instance, Link, LinkType, LinkState
from prism.embeddings import Embedder
from prism.resonance import ResonanceEngine
from prism.encoder import Encoder
from prism.explorer import Explorer


@pytest.fixture
def populated_env():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = PrismDB(path)
    embedder = Embedder()
    engine = ResonanceEngine(db, embedder)
    encoder = Encoder(db, engine)

    db.insert_pattern(Pattern(id="symmetry-breaking", name="Symmetry Breaking", description="Uniform becomes differentiated"))
    db.insert_pattern(Pattern(id="recursion", name="Recursion", description="Structure that contains itself"))

    encoder.encode(instance_id="gram-1", domain="grammar", description="Irregular verbs resist regularization",
                   structural_signature={"symmetry_type": "broken", "dimensionality": "temporal", "scope": "fundamental"},
                   pattern_ids=["symmetry-breaking"])
    encoder.encode(instance_id="phys-1", domain="physics", description="Hydrogen has complex spectra despite simplicity",
                   structural_signature={"symmetry_type": "broken", "dimensionality": "energetic", "scope": "fundamental"},
                   pattern_ids=["symmetry-breaking"])
    encoder.encode(instance_id="cog-1", domain="cognition", description="Self-referential thoughts create loops",
                   structural_signature={"symmetry_type": "recursive", "dimensionality": "abstract", "scope": "local"},
                   pattern_ids=["recursion"])

    explorer = Explorer(db, embedder, engine)
    yield explorer, db
    db.close()
    os.unlink(path)


class TestWander:
    def test_wander_from_instance(self, populated_env):
        explorer, db = populated_env
        result = explorer.wander("gram-1")
        assert result["node"].id == "gram-1"
        assert "links" in result
        assert isinstance(result["links"], list)

    def test_wander_from_pattern(self, populated_env):
        explorer, db = populated_env
        result = explorer.wander_pattern("symmetry-breaking")
        assert result["pattern"].id == "symmetry-breaking"
        assert "instances" in result
        assert len(result["instances"]) >= 1


class TestDrop:
    def test_drop_finds_neighbors(self, populated_env):
        explorer, db = populated_env
        result = explorer.drop("something fundamental that resists simplification")
        assert "neighbors" in result
        assert isinstance(result["neighbors"], list)


class TestDrift:
    def test_drift_returns_speculative_link(self, populated_env):
        explorer, db = populated_env
        result = explorer.drift()
        if result is not None:
            assert "link" in result
            assert "source" in result
            assert "target" in result
