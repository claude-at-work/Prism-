# tests/test_db.py
import pytest
import tempfile
import os
from prism.db import PrismDB
from prism.models import Pattern, Instance, Link, LinkType, LinkState


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = PrismDB(path)
    yield database
    database.close()
    os.unlink(path)


class TestSchema:
    def test_tables_created(self, db):
        tables = db.list_tables()
        assert "patterns" in tables
        assert "instances" in tables
        assert "links" in tables
        assert "instance_patterns" in tables

    def test_vec_table_created(self, db):
        tables = db.list_tables()
        assert "instance_embeddings" in tables


class TestPatterns:
    def test_insert_and_get_pattern(self, db):
        p = Pattern(id="recursion", name="Recursion", description="Structure that contains itself")
        db.insert_pattern(p)
        result = db.get_pattern("recursion")
        assert result.name == "Recursion"

    def test_list_patterns(self, db):
        db.insert_pattern(Pattern(id="a", name="A", description="first"))
        db.insert_pattern(Pattern(id="b", name="B", description="second"))
        patterns = db.list_patterns()
        assert len(patterns) == 2


class TestInstances:
    def test_insert_and_get_instance(self, db):
        db.insert_pattern(Pattern(id="recursion", name="Recursion", description="test"))
        inst = Instance(
            id="test-inst",
            domain="grammar",
            description="A test instance",
            structural_signature={"symmetry": "broken"},
            pattern_ids=["recursion"],
            embedding=[0.1, 0.2, 0.3],
        )
        db.insert_instance(inst)
        result = db.get_instance("test-inst")
        assert result.domain == "grammar"
        assert result.pattern_ids == ["recursion"]

    def test_list_instances_by_domain(self, db):
        db.insert_pattern(Pattern(id="r", name="R", description="test"))
        db.insert_instance(Instance(id="a", domain="grammar", description="a", structural_signature={"x": "1"}, pattern_ids=["r"], embedding=[0.1]))
        db.insert_instance(Instance(id="b", domain="physics", description="b", structural_signature={"x": "2"}, pattern_ids=["r"], embedding=[0.2]))
        db.insert_instance(Instance(id="c", domain="grammar", description="c", structural_signature={"x": "3"}, pattern_ids=["r"], embedding=[0.3]))
        grammar_instances = db.list_instances(domain="grammar")
        assert len(grammar_instances) == 2

    def test_list_instances_by_pattern(self, db):
        db.insert_pattern(Pattern(id="r", name="R", description="test"))
        db.insert_pattern(Pattern(id="s", name="S", description="test"))
        db.insert_instance(Instance(id="a", domain="grammar", description="a", structural_signature={"x": "1"}, pattern_ids=["r"], embedding=[0.1]))
        db.insert_instance(Instance(id="b", domain="physics", description="b", structural_signature={"x": "2"}, pattern_ids=["r", "s"], embedding=[0.2]))
        r_instances = db.list_instances(pattern_id="r")
        assert len(r_instances) == 2
        s_instances = db.list_instances(pattern_id="s")
        assert len(s_instances) == 1


class TestLinks:
    def test_insert_and_get_link(self, db):
        link = Link(
            id="link-1",
            source_id="a",
            target_id="b",
            link_type=LinkType.STRUCTURALLY_SIMILAR,
            state=LinkState.PROPOSED,
            confidence=0.65,
            residual_description="temporal mismatch",
            residual_dimensions={"dim": ("temporal", "spatial")},
        )
        db.insert_link(link)
        result = db.get_link("link-1")
        assert result.confidence == 0.65
        assert result.state == LinkState.PROPOSED

    def test_get_links_for_instance(self, db):
        db.insert_link(Link(id="l1", source_id="a", target_id="b", link_type=LinkType.STRUCTURALLY_SIMILAR, state=LinkState.PROPOSED, confidence=0.6))
        db.insert_link(Link(id="l2", source_id="c", target_id="a", link_type=LinkType.EXTENDS, state=LinkState.CONFIRMED, confidence=0.9))
        db.insert_link(Link(id="l3", source_id="d", target_id="e", link_type=LinkType.CONTRADICTS, state=LinkState.PROPOSED, confidence=0.5))
        links = db.get_links_for_instance("a")
        assert len(links) == 2

    def test_update_link_state(self, db):
        db.insert_link(Link(id="l1", source_id="a", target_id="b", link_type=LinkType.STRUCTURALLY_SIMILAR, state=LinkState.PROPOSED, confidence=0.6))
        db.update_link("l1", state=LinkState.CORROBORATED, confidence=0.8)
        result = db.get_link("l1")
        assert result.state == LinkState.CORROBORATED
        assert result.confidence == 0.8

    def test_get_links_by_state(self, db):
        db.insert_link(Link(id="l1", source_id="a", target_id="b", link_type=LinkType.STRUCTURALLY_SIMILAR, state=LinkState.PROPOSED, confidence=0.6))
        db.insert_link(Link(id="l2", source_id="c", target_id="d", link_type=LinkType.STRUCTURALLY_SIMILAR, state=LinkState.CORROBORATED, confidence=0.8))
        proposed = db.get_links_by_state(LinkState.PROPOSED)
        assert len(proposed) == 1
        assert proposed[0].id == "l1"


class TestVectorSearch:
    def test_find_nearest_neighbors(self, db):
        db.insert_pattern(Pattern(id="r", name="R", description="test"))
        db.insert_instance(Instance(id="a", domain="grammar", description="a", structural_signature={"x": "1"}, pattern_ids=["r"], embedding=[1.0, 0.0, 0.0]))
        db.insert_instance(Instance(id="b", domain="physics", description="b", structural_signature={"x": "2"}, pattern_ids=["r"], embedding=[0.9, 0.1, 0.0]))
        db.insert_instance(Instance(id="c", domain="cognition", description="c", structural_signature={"x": "3"}, pattern_ids=["r"], embedding=[0.0, 0.0, 1.0]))
        neighbors = db.find_nearest(query_embedding=[1.0, 0.0, 0.0], k=2, exclude_id="a")
        assert len(neighbors) == 2
        assert neighbors[0][0] == "b"  # (id, distance) tuples, closest first
