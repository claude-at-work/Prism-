import pytest
from prism.models import Pattern, Instance, Link, LinkState, LinkType


class TestPattern:
    def test_create_pattern(self):
        p = Pattern(id="recursion", name="Recursion", description="Structure that contains itself")
        assert p.id == "recursion"
        assert p.name == "Recursion"
        assert p.description == "Structure that contains itself"

    def test_pattern_requires_id(self):
        with pytest.raises(TypeError):
            Pattern(name="Recursion", description="test")


class TestInstance:
    def test_create_instance(self):
        i = Instance(
            id="grammar-irregular-verbs",
            domain="grammar",
            description="The verb 'to be' is irregular across every Indo-European language",
            structural_signature={"symmetry_type": "broken", "dimensionality": "temporal", "self_reference": "false"},
            pattern_ids=["symmetry-breaking", "asymptotic-approach"],
        )
        assert i.id == "grammar-irregular-verbs"
        assert i.domain == "grammar"
        assert len(i.pattern_ids) == 2

    def test_instance_signature_serialization(self):
        i = Instance(
            id="test",
            domain="test",
            description="test instance",
            structural_signature={"a": "1", "b": "2", "c": "3"},
            pattern_ids=["recursion"],
        )
        serialized = i.serialize_signature()
        assert "a:1" in serialized
        assert "b:2" in serialized
        assert "c:3" in serialized

    def test_instance_requires_at_least_one_pattern(self):
        with pytest.raises(ValueError, match="at least one pattern"):
            Instance(
                id="test",
                domain="test",
                description="test",
                structural_signature={"a": "1"},
                pattern_ids=[],
            )


class TestLink:
    def test_create_speculative_link(self):
        link = Link(
            id="link-1",
            source_id="instance-a",
            target_id="instance-b",
            link_type=LinkType.STRUCTURALLY_SIMILAR,
            state=LinkState.PROPOSED,
            confidence=0.65,
            residual_description="Temporal dimension does not map",
            residual_dimensions={"dimensionality": ("temporal", "spatial")},
        )
        assert link.state == LinkState.PROPOSED
        assert link.confidence == 0.65
        assert link.is_plausible()
        assert not link.is_probable()

    def test_link_probable_threshold(self):
        link = Link(
            id="link-2",
            source_id="a",
            target_id="b",
            link_type=LinkType.STRUCTURALLY_SIMILAR,
            state=LinkState.CORROBORATED,
            confidence=0.82,
        )
        assert link.is_probable()

    def test_link_state_transitions(self):
        link = Link(
            id="link-3",
            source_id="a",
            target_id="b",
            link_type=LinkType.STRUCTURALLY_SIMILAR,
            state=LinkState.PROPOSED,
            confidence=0.6,
        )
        link.elevate(LinkState.CORROBORATED, new_confidence=0.8)
        assert link.state == LinkState.CORROBORATED
        assert link.confidence == 0.8

    def test_cannot_elevate_to_confirmed_programmatically(self):
        link = Link(
            id="link-4",
            source_id="a",
            target_id="b",
            link_type=LinkType.STRUCTURALLY_SIMILAR,
            state=LinkState.PROPOSED,
            confidence=0.6,
        )
        with pytest.raises(ValueError, match="human review"):
            link.elevate(LinkState.CONFIRMED)
