# tests/test_integration.py
import tempfile
import os
import pytest
from click.testing import CliRunner
from prism.cli import main


@pytest.fixture
def seeded_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    runner = CliRunner()
    runner.invoke(main, ["--db", path, "init"])
    runner.invoke(main, ["--db", path, "seed"])
    yield path, runner
    os.unlink(path)


class TestFullWorkflow:
    def test_seed_and_explore(self, seeded_db):
        path, runner = seeded_db

        # List patterns
        result = runner.invoke(main, ["--db", path, "patterns"])
        assert result.exit_code == 0
        assert "Recursion" in result.output

        # Wander from an instance
        result = runner.invoke(main, ["--db", path, "wander", "gram-irregular-verbs"])
        assert result.exit_code == 0
        assert "grammar" in result.output

        # Drop a description
        result = runner.invoke(main, ["--db", path, "drop", "something that resists simplification"])
        assert result.exit_code == 0
        assert "Nearest" in result.output

        # Drift
        result = runner.invoke(main, ["--db", path, "drift"])
        assert result.exit_code == 0

        # Apply pressures
        result = runner.invoke(main, ["--db", path, "pressures"])
        assert result.exit_code == 0
        assert "Pressures applied" in result.output

    def test_encode_new_instance_after_seed(self, seeded_db):
        path, runner = seeded_db

        result = runner.invoke(main, ["--db", path, "encode",
                                       "--id", "new-1",
                                       "--domain", "mathematics",
                                       "--description", "Godel's incompleteness: any sufficiently powerful system cannot prove its own consistency",
                                       "--signature", "symmetry_type=self-referential",
                                       "--signature", "dimensionality=logical",
                                       "--signature", "scope=universal",
                                       "--signature", "mechanism=self-reference",
                                       "--pattern", "recursion"])
        assert result.exit_code == 0

        # Should be findable via wander
        result = runner.invoke(main, ["--db", path, "wander", "new-1"])
        assert result.exit_code == 0
        assert "mathematics" in result.output
