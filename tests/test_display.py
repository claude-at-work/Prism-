import pytest
from prism.display import format_instance, format_link, format_pattern, format_residual, state_symbol
from prism.models import Instance, Link, LinkState, LinkType, Pattern


class TestFormatting:
    def test_format_pattern(self):
        p = Pattern(id="recursion", name="Recursion", description="Structure that contains itself")
        out = format_pattern(p)
        assert "Recursion" in out
        assert "Structure that contains itself" in out

    def test_format_instance(self):
        i = Instance(
            id="test", domain="grammar", description="A test",
            structural_signature={"symmetry": "broken"},
            pattern_ids=["symmetry-breaking"],
        )
        out = format_instance(i)
        assert "grammar" in out
        assert "A test" in out

    def test_format_link_with_state_symbol(self):
        link = Link(
            id="l1", source_id="a", target_id="b",
            link_type=LinkType.STRUCTURALLY_SIMILAR,
            state=LinkState.PROPOSED, confidence=0.65,
        )
        out = format_link(link)
        assert "0.65" in out or "65" in out

    def test_state_symbols(self):
        assert state_symbol(LinkState.CONFIRMED) != state_symbol(LinkState.PROPOSED)
        assert state_symbol(LinkState.WEAK) != state_symbol(LinkState.CORROBORATED)

    def test_format_residual(self):
        residual = {"dimensionality": ("temporal", "spatial"), "boundary": ("open", "(absent)")}
        out = format_residual(residual)
        assert "dimensionality" in out
        assert "temporal" in out
        assert "spatial" in out
