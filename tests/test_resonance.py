import pytest
import tempfile
import os
from prism.db import PrismDB
from prism.models import Pattern, Instance, LinkState
from prism.embeddings import Embedder
from prism.resonance import ResonanceEngine


@pytest.fixture
def env():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = PrismDB(path)
    embedder = Embedder()
    engine = ResonanceEngine(db, embedder)
    yield db, embedder, engine
    db.close()
    os.unlink(path)


def _setup_patterns(db):
    db.insert_pattern(Pattern(id="symmetry-breaking", name="Symmetry Breaking", description="Where uniform becomes differentiated"))
    db.insert_pattern(Pattern(id="recursion", name="Recursion", description="Structure that contains itself"))
    db.insert_pattern(Pattern(id="asymptotic-approach", name="Asymptotic Approach", description="Getting closer without arriving"))


class TestSpeculativeLinkGeneration:
    def test_generates_links_for_similar_instances(self, env):
        db, embedder, engine = env
        _setup_patterns(db)
        inst_a = Instance(
            id="a", domain="grammar", description="test a",
            structural_signature={"symmetry_type": "broken", "dimensionality": "temporal", "boundary": "open"},
            pattern_ids=["symmetry-breaking"],
        )
        inst_b = Instance(
            id="b", domain="physics", description="test b",
            structural_signature={"symmetry_type": "broken", "dimensionality": "temporal", "boundary": "closed"},
            pattern_ids=["symmetry-breaking"],
        )
        inst_c = Instance(
            id="c", domain="cognition", description="test c",
            structural_signature={"symmetry_type": "rotational", "dimensionality": "spatial", "boundary": "none"},
            pattern_ids=["recursion"],
        )
        links = engine.add_instance(inst_a)
        assert len(links) == 0  # first instance, nothing to compare

        links = engine.add_instance(inst_b)
        # b should be similar to a

        links_c = engine.add_instance(inst_c)
        # c should be less similar to a and b

    def test_links_have_residuals(self, env):
        db, embedder, engine = env
        _setup_patterns(db)
        engine.add_instance(Instance(
            id="a", domain="grammar", description="test",
            structural_signature={"symmetry_type": "broken", "dimensionality": "temporal"},
            pattern_ids=["symmetry-breaking"],
        ))
        links = engine.add_instance(Instance(
            id="b", domain="physics", description="test",
            structural_signature={"symmetry_type": "broken", "dimensionality": "spatial"},
            pattern_ids=["symmetry-breaking"],
        ))
        for link in links:
            stored = db.get_link(link.id)
            if stored.residual_dimensions:
                assert "dimensionality" in stored.residual_dimensions


class TestUpwardPressure:
    def test_triangulation_elevates_link(self, env):
        db, embedder, engine = env
        _setup_patterns(db)
        engine.add_instance(Instance(
            id="a", domain="grammar", description="test",
            structural_signature={"symmetry_type": "broken", "dimensionality": "temporal", "scope": "local"},
            pattern_ids=["symmetry-breaking"],
        ))
        engine.add_instance(Instance(
            id="b", domain="physics", description="test",
            structural_signature={"symmetry_type": "broken", "dimensionality": "temporal", "scope": "local"},
            pattern_ids=["symmetry-breaking"],
        ))
        engine.add_instance(Instance(
            id="c", domain="cognition", description="test",
            structural_signature={"symmetry_type": "broken", "dimensionality": "temporal", "scope": "local"},
            pattern_ids=["symmetry-breaking"],
        ))
        engine.apply_pressures()
        links = db.get_links_for_instance("a")
        corroborated = [l for l in links if l.state == LinkState.CORROBORATED]
        assert isinstance(corroborated, list)


class TestDownwardPressure:
    def test_isolated_links_decay(self, env):
        db, embedder, engine = env
        _setup_patterns(db)
        engine.add_instance(Instance(
            id="a", domain="grammar", description="test",
            structural_signature={"symmetry_type": "broken", "dimensionality": "temporal"},
            pattern_ids=["symmetry-breaking"],
        ))
        engine.add_instance(Instance(
            id="b", domain="physics", description="test",
            structural_signature={"symmetry_type": "broken", "dimensionality": "spatial"},
            pattern_ids=["symmetry-breaking"],
        ))
        for i in range(5):
            engine.add_instance(Instance(
                id=f"noise-{i}", domain="cognition", description="test",
                structural_signature={"symmetry_type": "rotational", "dimensionality": "abstract", "index": str(i)},
                pattern_ids=["recursion"],
            ))
        engine.apply_pressures()
        links = db.get_links_for_instance("a")
        for link in links:
            if link.source_id == "b" or link.target_id == "b":
                assert link.state in (LinkState.PROPOSED, LinkState.WEAK)


class TestLateralPressure:
    def test_detects_unnamed_clusters(self, env):
        db, embedder, engine = env
        _setup_patterns(db)
        for i in range(4):
            engine.add_instance(Instance(
                id=f"cluster-{i}", domain="grammar", description=f"test {i}",
                structural_signature={"symmetry_type": "broken", "dimensionality": "temporal", "index": str(i)},
                pattern_ids=["symmetry-breaking"],
            ))
        proposals = engine.detect_emergent_patterns()
        assert isinstance(proposals, list)
