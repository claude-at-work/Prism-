import pytest
import tempfile
import os
from prism.db import PrismDB
from prism.embeddings import Embedder
from prism.resonance import ResonanceEngine
from prism.encoder import Encoder
from prism.seed import seed_atlas, SEED_PATTERNS, SEED_INSTANCES


class TestSeedData:
    def test_seed_patterns_are_valid(self):
        assert len(SEED_PATTERNS) >= 7
        for p in SEED_PATTERNS:
            assert p.id
            assert p.name
            assert p.description

    def test_seed_instances_cover_four_domains(self):
        domains = {inst["domain"] for inst in SEED_INSTANCES}
        assert "grammar" in domains
        assert "observer-reality" in domains
        assert "cognition" in domains
        assert "prediction" in domains

    def test_seed_instances_have_valid_patterns(self):
        pattern_ids = {p.id for p in SEED_PATTERNS}
        for inst in SEED_INSTANCES:
            for pid in inst["pattern_ids"]:
                assert pid in pattern_ids, f"Instance {inst['id']} references unknown pattern {pid}"

    def test_seed_instances_minimum_count(self):
        assert len(SEED_INSTANCES) >= 20


class TestSeeding:
    def test_seed_populates_database(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db = PrismDB(path)
        embedder = Embedder()
        engine = ResonanceEngine(db, embedder)
        encoder = Encoder(db, engine)

        seed_atlas(db, encoder)

        patterns = db.list_patterns()
        assert len(patterns) >= 7
        instances = db.list_instances()
        assert len(instances) >= 20

        db.close()
        os.unlink(path)
