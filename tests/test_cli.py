# tests/test_cli.py
import pytest
import tempfile
import os
from click.testing import CliRunner
from prism.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


class TestCLI:
    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Prism" in result.output or "prism" in result.output

    def test_init_creates_database(self, runner, db_path):
        result = runner.invoke(main, ["--db", db_path, "init"])
        assert result.exit_code == 0

    def test_add_pattern(self, runner, db_path):
        runner.invoke(main, ["--db", db_path, "init"])
        result = runner.invoke(main, ["--db", db_path, "add-pattern",
                                       "--id", "recursion",
                                       "--name", "Recursion",
                                       "--description", "Structure that contains itself"])
        assert result.exit_code == 0
        assert "recursion" in result.output.lower() or "Recursion" in result.output

    def test_add_instance(self, runner, db_path):
        runner.invoke(main, ["--db", db_path, "init"])
        runner.invoke(main, ["--db", db_path, "add-pattern",
                              "--id", "recursion", "--name", "Recursion",
                              "--description", "Structure that contains itself"])
        result = runner.invoke(main, ["--db", db_path, "encode",
                                       "--id", "test-1",
                                       "--domain", "grammar",
                                       "--description", "A test instance",
                                       "--signature", "symmetry_type=broken",
                                       "--signature", "dimensionality=temporal",
                                       "--pattern", "recursion"])
        assert result.exit_code == 0

    def test_wander(self, runner, db_path):
        runner.invoke(main, ["--db", db_path, "init"])
        runner.invoke(main, ["--db", db_path, "add-pattern",
                              "--id", "r", "--name", "R", "--description", "test"])
        runner.invoke(main, ["--db", db_path, "encode",
                              "--id", "a", "--domain", "grammar",
                              "--description", "test a",
                              "--signature", "x=1", "--pattern", "r"])
        result = runner.invoke(main, ["--db", db_path, "wander", "a"])
        assert result.exit_code == 0
        assert "grammar" in result.output

    def test_drop(self, runner, db_path):
        runner.invoke(main, ["--db", db_path, "init"])
        runner.invoke(main, ["--db", db_path, "add-pattern",
                              "--id", "r", "--name", "R", "--description", "test"])
        runner.invoke(main, ["--db", db_path, "encode",
                              "--id", "a", "--domain", "grammar",
                              "--description", "test a",
                              "--signature", "x=1", "--pattern", "r"])
        result = runner.invoke(main, ["--db", db_path, "drop", "something about grammar"])
        assert result.exit_code == 0

    def test_drift(self, runner, db_path):
        runner.invoke(main, ["--db", db_path, "init"])
        result = runner.invoke(main, ["--db", db_path, "drift"])
        assert result.exit_code == 0

    def test_patterns_list(self, runner, db_path):
        runner.invoke(main, ["--db", db_path, "init"])
        runner.invoke(main, ["--db", db_path, "add-pattern",
                              "--id", "r", "--name", "Recursion", "--description", "test"])
        result = runner.invoke(main, ["--db", db_path, "patterns"])
        assert result.exit_code == 0
        assert "Recursion" in result.output

    def test_review_link(self, runner, db_path):
        runner.invoke(main, ["--db", db_path, "init"])
        runner.invoke(main, ["--db", db_path, "add-pattern",
                              "--id", "r", "--name", "R", "--description", "test"])
        runner.invoke(main, ["--db", db_path, "encode",
                              "--id", "a", "--domain", "grammar", "--description", "a",
                              "--signature", "symmetry=broken", "--signature", "dim=temporal",
                              "--pattern", "r"])
        runner.invoke(main, ["--db", db_path, "encode",
                              "--id", "b", "--domain", "physics", "--description", "b",
                              "--signature", "symmetry=broken", "--signature", "dim=temporal",
                              "--pattern", "r"])
        result = runner.invoke(main, ["--db", db_path, "wander", "a"])
        assert result.exit_code == 0
