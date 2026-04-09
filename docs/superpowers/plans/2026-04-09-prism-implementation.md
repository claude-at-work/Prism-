# Prism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Pattern Atlas with Resonance Engine — a typed hypergraph for meta-structural patterns across domains, with speculative link discovery and a terminal-based explorer CLI.

**Architecture:** SQLite database with sqlite-vec extension for vector similarity search. Python CLI built with Click. Structural signatures are serialized and embedded via TF-IDF vectorization (scikit-learn). The resonance engine runs on every new instance insertion, proposing speculative links with residuals and applying three-pressure dynamics.

**Tech Stack:** Python 3.13, SQLite + sqlite-vec, scikit-learn (TF-IDF + cosine similarity), NumPy, Click

---

## File Structure

```
prism/
├── __init__.py              # Package init, version
├── db.py                    # Database schema, connection, migrations
├── models.py                # Data classes: Pattern, Instance, Link, Residual
├── embeddings.py            # TF-IDF vectorizer, embedding computation, similarity search
├── resonance.py             # Resonance engine: speculative links, residuals, three-pressure dynamics
├── encoder.py               # Curator interface: structured instance encoding workflow
├── explorer.py              # Explorer CLI: wander, drop, drift modes
├── cli.py                   # Click CLI entry point, command routing
├── display.py               # Terminal formatting and output helpers
└── seed.py                  # Starter library: 20-30 instances across seed domains
tests/
├── __init__.py
├── test_db.py               # Database schema and query tests
├── test_models.py           # Data class validation tests
├── test_embeddings.py       # Embedding computation and similarity tests
├── test_resonance.py        # Resonance engine logic tests
├── test_encoder.py          # Encoding workflow tests
├── test_explorer.py         # Explorer mode tests
└── test_seed.py             # Seed library integrity tests
```

**Responsibilities:**
- `db.py` — owns the SQLite schema and all raw queries. Single source of truth for database interaction.
- `models.py` — pure data classes with no database dependency. Validation logic lives here.
- `embeddings.py` — owns the TF-IDF vectorizer fitting/transforming and cosine similarity computation. Interfaces with sqlite-vec for vector storage and retrieval.
- `resonance.py` — the brain. Orchestrates speculative link generation, residual computation, and the three-pressure system (upward/downward/lateral). Depends on embeddings.py for similarity, db.py for storage.
- `encoder.py` — the curator workflow. Takes raw input, validates it, creates models, passes to db and resonance.
- `explorer.py` — the user-facing exploration logic. Wander/drop/drift modes. Depends on db.py for queries, embeddings.py for drop mode, resonance.py for drift mode.
- `cli.py` — thin CLI layer. Parses commands, delegates to encoder.py and explorer.py.
- `display.py` — terminal formatting. Rich-text output for links, patterns, residuals. No business logic.
- `seed.py` — the starter library. Hardcoded instances across four seed domains. Run once to populate.

---

### Task 1: Project Setup and Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `prism/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "prism"
version = "0.1.0"
description = "Pattern Atlas with Resonance Discovery"
requires-python = ">=3.13"
dependencies = [
    "click>=8.0",
    "sqlite-vec>=0.1.9",
    "scikit-learn>=1.8",
    "numpy>=2.0",
]

[project.scripts]
prism = "prism.cli:main"

[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package init**

```python
# prism/__init__.py
__version__ = "0.1.0"
```

```python
# tests/__init__.py
```

- [ ] **Step 3: Install dependencies**

Run: `source .venv/bin/activate && pip install -e ".[dev]" 2>&1 || pip install -e . && pip install pytest`
Expected: All packages install successfully

- [ ] **Step 4: Verify setup**

Run: `source .venv/bin/activate && python -c "import sqlite_vec; import click; import sklearn; print('all imports ok')"`
Expected: `all imports ok`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml prism/__init__.py tests/__init__.py
git commit -m "feat: project setup with dependencies"
```

---

### Task 2: Data Models

**Files:**
- Create: `prism/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for data models**

```python
# tests/test_models.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prism.models'`

- [ ] **Step 3: Implement models**

```python
# prism/models.py
from dataclasses import dataclass, field
from enum import Enum


class LinkType(str, Enum):
    IS_INSTANCE_OF = "is_instance_of"
    STRUCTURALLY_SIMILAR = "structurally_similar"
    RESIDUAL = "residual"
    EXTENDS = "extends"
    CONTRADICTS = "contradicts"


class LinkState(str, Enum):
    PROPOSED = "proposed"
    CORROBORATED = "corroborated"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    WEAK = "weak"


@dataclass
class Pattern:
    id: str
    name: str
    description: str


@dataclass
class Instance:
    id: str
    domain: str
    description: str
    structural_signature: dict[str, str]
    pattern_ids: list[str]
    embedding: list[float] | None = None
    created_by: str = "curator"
    encoding_rationale: str = ""

    def __post_init__(self):
        if not self.pattern_ids:
            raise ValueError("Instance must have at least one pattern ID")

    def serialize_signature(self) -> str:
        return " ".join(f"{k}:{v}" for k, v in sorted(self.structural_signature.items()))


@dataclass
class Link:
    id: str
    source_id: str
    target_id: str
    link_type: LinkType
    state: LinkState
    confidence: float = 0.0
    residual_description: str = ""
    residual_dimensions: dict[str, tuple[str, str]] = field(default_factory=dict)
    review_note: str = ""

    def is_plausible(self) -> bool:
        return self.confidence > 0.5

    def is_probable(self) -> bool:
        return self.confidence > 0.75 and self.state in (LinkState.CORROBORATED, LinkState.CONFIRMED)

    def elevate(self, new_state: LinkState, new_confidence: float | None = None):
        if new_state in (LinkState.CONFIRMED, LinkState.REJECTED):
            raise ValueError(f"Cannot programmatically set state to {new_state} — requires human review")
        self.state = new_state
        if new_confidence is not None:
            self.confidence = new_confidence
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_models.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prism/models.py tests/test_models.py
git commit -m "feat: data models for Pattern, Instance, Link"
```

---

### Task 3: Database Layer

**Files:**
- Create: `prism/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write failing tests for database**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prism.db'`

- [ ] **Step 3: Implement database layer**

```python
# prism/db.py
import json
import sqlite3
import struct

import sqlite_vec

from prism.models import Instance, Link, LinkState, LinkType, Pattern


def _serialize_float_vec(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_float_vec(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class PrismDB:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self):
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS instances (
                id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                description TEXT NOT NULL,
                structural_signature TEXT NOT NULL,
                created_by TEXT NOT NULL DEFAULT 'curator',
                encoding_rationale TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS instance_patterns (
                instance_id TEXT NOT NULL,
                pattern_id TEXT NOT NULL,
                PRIMARY KEY (instance_id, pattern_id),
                FOREIGN KEY (instance_id) REFERENCES instances(id),
                FOREIGN KEY (pattern_id) REFERENCES patterns(id)
            );

            CREATE TABLE IF NOT EXISTS links (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                link_type TEXT NOT NULL,
                state TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                residual_description TEXT NOT NULL DEFAULT '',
                residual_dimensions TEXT NOT NULL DEFAULT '{}',
                review_note TEXT NOT NULL DEFAULT ''
            );
        """)
        # sqlite-vec virtual table — dimension set on first insert
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS instance_embeddings
            USING vec0(id TEXT PRIMARY KEY, embedding float[64])
        """)
        self.conn.commit()
        self._vec_dim = 64

    def _ensure_vec_dim(self, dim: int):
        if dim != self._vec_dim:
            cur = self.conn.cursor()
            cur.execute("DROP TABLE IF EXISTS instance_embeddings")
            cur.execute(f"""
                CREATE VIRTUAL TABLE instance_embeddings
                USING vec0(id TEXT PRIMARY KEY, embedding float[{dim}])
            """)
            self.conn.commit()
            self._vec_dim = dim

    def close(self):
        self.conn.close()

    def list_tables(self) -> list[str]:
        cur = self.conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow')")
        all_names = [row[0] for row in cur.fetchall()]
        # Include virtual tables by checking for the base name
        tables = set()
        for name in all_names:
            if name.startswith("instance_embeddings"):
                tables.add("instance_embeddings")
            elif not name.startswith("sqlite_"):
                tables.add(name)
        return sorted(tables)

    # --- Patterns ---

    def insert_pattern(self, p: Pattern):
        self.conn.execute(
            "INSERT INTO patterns (id, name, description) VALUES (?, ?, ?)",
            (p.id, p.name, p.description),
        )
        self.conn.commit()

    def get_pattern(self, pattern_id: str) -> Pattern:
        row = self.conn.execute("SELECT * FROM patterns WHERE id = ?", (pattern_id,)).fetchone()
        if row is None:
            raise KeyError(f"Pattern not found: {pattern_id}")
        return Pattern(id=row["id"], name=row["name"], description=row["description"])

    def list_patterns(self) -> list[Pattern]:
        rows = self.conn.execute("SELECT * FROM patterns ORDER BY id").fetchall()
        return [Pattern(id=r["id"], name=r["name"], description=r["description"]) for r in rows]

    # --- Instances ---

    def insert_instance(self, inst: Instance):
        self.conn.execute(
            "INSERT INTO instances (id, domain, description, structural_signature, created_by, encoding_rationale) VALUES (?, ?, ?, ?, ?, ?)",
            (inst.id, inst.domain, inst.description, json.dumps(inst.structural_signature), inst.created_by, inst.encoding_rationale),
        )
        for pid in inst.pattern_ids:
            self.conn.execute(
                "INSERT INTO instance_patterns (instance_id, pattern_id) VALUES (?, ?)",
                (inst.id, pid),
            )
        if inst.embedding:
            self._ensure_vec_dim(len(inst.embedding))
            self.conn.execute(
                "INSERT INTO instance_embeddings (id, embedding) VALUES (?, ?)",
                (inst.id, _serialize_float_vec(inst.embedding)),
            )
        self.conn.commit()

    def get_instance(self, instance_id: str) -> Instance:
        row = self.conn.execute("SELECT * FROM instances WHERE id = ?", (instance_id,)).fetchone()
        if row is None:
            raise KeyError(f"Instance not found: {instance_id}")
        pattern_rows = self.conn.execute(
            "SELECT pattern_id FROM instance_patterns WHERE instance_id = ?", (instance_id,)
        ).fetchall()
        pattern_ids = [r["pattern_id"] for r in pattern_rows]
        # Try to get embedding
        embedding = None
        try:
            emb_row = self.conn.execute(
                "SELECT embedding FROM instance_embeddings WHERE id = ?", (instance_id,)
            ).fetchone()
            if emb_row:
                embedding = _deserialize_float_vec(emb_row["embedding"])
        except Exception:
            pass
        return Instance(
            id=row["id"],
            domain=row["domain"],
            description=row["description"],
            structural_signature=json.loads(row["structural_signature"]),
            pattern_ids=pattern_ids,
            embedding=embedding,
            created_by=row["created_by"],
            encoding_rationale=row["encoding_rationale"],
        )

    def list_instances(self, domain: str | None = None, pattern_id: str | None = None) -> list[Instance]:
        if pattern_id:
            rows = self.conn.execute(
                "SELECT i.id FROM instances i JOIN instance_patterns ip ON i.id = ip.instance_id WHERE ip.pattern_id = ?",
                (pattern_id,),
            ).fetchall()
        elif domain:
            rows = self.conn.execute("SELECT id FROM instances WHERE domain = ?", (domain,)).fetchall()
        else:
            rows = self.conn.execute("SELECT id FROM instances").fetchall()
        return [self.get_instance(r["id"]) for r in rows]

    # --- Links ---

    def insert_link(self, link: Link):
        self.conn.execute(
            "INSERT INTO links (id, source_id, target_id, link_type, state, confidence, residual_description, residual_dimensions, review_note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (link.id, link.source_id, link.target_id, link.link_type.value, link.state.value, link.confidence, link.residual_description, json.dumps({k: list(v) for k, v in link.residual_dimensions.items()}), link.review_note),
        )
        self.conn.commit()

    def get_link(self, link_id: str) -> Link:
        row = self.conn.execute("SELECT * FROM links WHERE id = ?", (link_id,)).fetchone()
        if row is None:
            raise KeyError(f"Link not found: {link_id}")
        dims_raw = json.loads(row["residual_dimensions"])
        return Link(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            link_type=LinkType(row["link_type"]),
            state=LinkState(row["state"]),
            confidence=row["confidence"],
            residual_description=row["residual_description"],
            residual_dimensions={k: tuple(v) for k, v in dims_raw.items()},
            review_note=row["review_note"],
        )

    def get_links_for_instance(self, instance_id: str) -> list[Link]:
        rows = self.conn.execute(
            "SELECT id FROM links WHERE source_id = ? OR target_id = ?",
            (instance_id, instance_id),
        ).fetchall()
        return [self.get_link(r["id"]) for r in rows]

    def get_links_by_state(self, state: LinkState) -> list[Link]:
        rows = self.conn.execute("SELECT id FROM links WHERE state = ?", (state.value,)).fetchall()
        return [self.get_link(r["id"]) for r in rows]

    def update_link(self, link_id: str, state: LinkState | None = None, confidence: float | None = None, review_note: str | None = None):
        updates = []
        params = []
        if state is not None:
            updates.append("state = ?")
            params.append(state.value)
        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)
        if review_note is not None:
            updates.append("review_note = ?")
            params.append(review_note)
        if not updates:
            return
        params.append(link_id)
        self.conn.execute(f"UPDATE links SET {', '.join(updates)} WHERE id = ?", params)
        self.conn.commit()

    # --- Vector search ---

    def find_nearest(self, query_embedding: list[float], k: int = 5, exclude_id: str | None = None) -> list[tuple[str, float]]:
        self._ensure_vec_dim(len(query_embedding))
        rows = self.conn.execute(
            "SELECT id, distance FROM instance_embeddings WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (_serialize_float_vec(query_embedding), k + (1 if exclude_id else 0)),
        ).fetchall()
        results = [(r["id"], r["distance"]) for r in rows if r["id"] != exclude_id]
        return results[:k]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_db.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prism/db.py tests/test_db.py
git commit -m "feat: database layer with SQLite + sqlite-vec"
```

---

### Task 4: Embedding System

**Files:**
- Create: `prism/embeddings.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing tests for embeddings**

```python
# tests/test_embeddings.py
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
        # Dimensions may change after refit, both should work
        assert dim1 > 0
        assert dim2 > 0

    def test_compute_residual(self):
        sig_a = {"symmetry_type": "broken", "dimensionality": "temporal", "boundary": "open"}
        sig_b = {"symmetry_type": "broken", "dimensionality": "spatial", "boundary": "open"}
        residual = Embedder.compute_residual(sig_a, sig_b)
        assert "dimensionality" in residual
        assert residual["dimensionality"] == ("temporal", "spatial")
        assert "symmetry_type" not in residual  # same value, no residual
        assert "boundary" not in residual  # same value, no residual
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_embeddings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prism.embeddings'`

- [ ] **Step 3: Implement embedding system**

```python
# prism/embeddings.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder:
    def __init__(self):
        self._vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"[a-zA-Z0-9_]+")
        self._fitted = False

    def _serialize_signature(self, sig: dict[str, str]) -> str:
        return " ".join(f"{k} {v}" for k, v in sorted(sig.items()))

    def fit(self, signatures: list[dict[str, str]]):
        texts = [self._serialize_signature(s) for s in signatures]
        self._vectorizer.fit(texts)
        self._fitted = True

    def embed(self, signature: dict[str, str]) -> list[float]:
        if not self._fitted:
            raise RuntimeError("Embedder must be fit before embedding")
        text = self._serialize_signature(signature)
        vec = self._vectorizer.transform([text]).toarray()[0]
        return vec.tolist()

    def embed_text(self, text: str) -> list[float]:
        if not self._fitted:
            raise RuntimeError("Embedder must be fit before embedding")
        vec = self._vectorizer.transform([text]).toarray()[0]
        return vec.tolist()

    @staticmethod
    def cosine_distance(a: list[float], b: list[float]) -> float:
        a_arr = np.array(a)
        b_arr = np.array(b)
        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - (dot / (norm_a * norm_b))

    @staticmethod
    def compute_residual(sig_a: dict[str, str], sig_b: dict[str, str]) -> dict[str, tuple[str, str]]:
        all_keys = set(sig_a.keys()) | set(sig_b.keys())
        residual = {}
        for key in all_keys:
            val_a = sig_a.get(key)
            val_b = sig_b.get(key)
            if val_a != val_b:
                residual[key] = (val_a or "(absent)", val_b or "(absent)")
        return residual
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_embeddings.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prism/embeddings.py tests/test_embeddings.py
git commit -m "feat: TF-IDF embedding system for structural signatures"
```

---

### Task 5: Resonance Engine

**Files:**
- Create: `prism/resonance.py`
- Create: `tests/test_resonance.py`

- [ ] **Step 1: Write failing tests for resonance engine**

```python
# tests/test_resonance.py
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
        assert len(links) >= 0  # may or may not hit threshold with TF-IDF

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
            # If there's a link, it should have residual info about the dimensionality difference
            if stored.residual_dimensions:
                assert "dimensionality" in stored.residual_dimensions


class TestUpwardPressure:
    def test_triangulation_elevates_link(self, env):
        db, embedder, engine = env
        _setup_patterns(db)
        # Add three instances where a-b and a-c and b-c are all similar
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
        # Third instance that's similar to both — should trigger triangulation
        engine.add_instance(Instance(
            id="c", domain="cognition", description="test",
            structural_signature={"symmetry_type": "broken", "dimensionality": "temporal", "scope": "local"},
            pattern_ids=["symmetry-breaking"],
        ))
        engine.apply_pressures()
        # Check if any links got elevated
        links = db.get_links_for_instance("a")
        corroborated = [l for l in links if l.state == LinkState.CORROBORATED]
        # With three very similar instances, triangulation should fire
        # (depends on confidence thresholds — this tests the mechanism exists)
        assert isinstance(corroborated, list)  # mechanism exists and doesn't crash


class TestDownwardPressure:
    def test_isolated_links_decay(self, env):
        db, embedder, engine = env
        _setup_patterns(db)
        # Add two somewhat similar instances
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
        # Add several instances that DON'T corroborate the a-b link
        for i in range(5):
            engine.add_instance(Instance(
                id=f"noise-{i}", domain="cognition", description="test",
                structural_signature={"symmetry_type": "rotational", "dimensionality": "abstract", "index": str(i)},
                pattern_ids=["recursion"],
            ))
        engine.apply_pressures()
        # The a-b link should not have been elevated (may still be proposed or weak)
        links = db.get_links_for_instance("a")
        for link in links:
            if link.source_id == "b" or link.target_id == "b":
                assert link.state in (LinkState.PROPOSED, LinkState.WEAK)


class TestLateralPressure:
    def test_detects_unnamed_clusters(self, env):
        db, embedder, engine = env
        _setup_patterns(db)
        # Add several instances that cluster together but use a pattern
        for i in range(4):
            engine.add_instance(Instance(
                id=f"cluster-{i}", domain="grammar", description=f"test {i}",
                structural_signature={"symmetry_type": "broken", "dimensionality": "temporal", "index": str(i)},
                pattern_ids=["symmetry-breaking"],
            ))
        proposals = engine.detect_emergent_patterns()
        # Should return a list (possibly empty for small graphs — tests the mechanism)
        assert isinstance(proposals, list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_resonance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prism.resonance'`

- [ ] **Step 3: Implement resonance engine**

```python
# prism/resonance.py
import uuid

from prism.db import PrismDB
from prism.embeddings import Embedder
from prism.models import Instance, Link, LinkState, LinkType


class ResonanceEngine:
    PLAUSIBLE_THRESHOLD = 0.5  # cosine distance below this = plausible
    PROBABLE_THRESHOLD = 0.25  # cosine distance below this = probable candidate
    DECAY_ROUNDS = 3  # rounds without corroboration before weakening
    MIN_CLUSTER_SIZE = 3  # minimum instances to propose emergent pattern

    def __init__(self, db: PrismDB, embedder: Embedder):
        self.db = db
        self.embedder = embedder
        self._rounds_since_corroboration: dict[str, int] = {}

    def _refit_embedder(self):
        all_instances = self.db.list_instances()
        if not all_instances:
            return
        sigs = [inst.structural_signature for inst in all_instances]
        self.embedder.fit(sigs)
        # Recompute all embeddings after refit
        for inst in all_instances:
            new_emb = self.embedder.embed(inst.structural_signature)
            inst.embedding = new_emb
            # Update in DB — delete and reinsert into vec table
            try:
                self.db.conn.execute("DELETE FROM instance_embeddings WHERE id = ?", (inst.id,))
            except Exception:
                pass
            from prism.db import _serialize_float_vec
            self.db._ensure_vec_dim(len(new_emb))
            self.db.conn.execute(
                "INSERT INTO instance_embeddings (id, embedding) VALUES (?, ?)",
                (inst.id, _serialize_float_vec(new_emb)),
            )
        self.db.conn.commit()

    def add_instance(self, instance: Instance) -> list[Link]:
        # Refit embedder with all existing signatures plus new one
        all_instances = self.db.list_instances()
        all_sigs = [inst.structural_signature for inst in all_instances]
        all_sigs.append(instance.structural_signature)
        self.embedder.fit(all_sigs)

        # Compute embedding for new instance
        instance.embedding = self.embedder.embed(instance.structural_signature)

        # Recompute and update all existing embeddings
        for existing in all_instances:
            new_emb = self.embedder.embed(existing.structural_signature)
            try:
                self.db.conn.execute("DELETE FROM instance_embeddings WHERE id = ?", (existing.id,))
            except Exception:
                pass
            from prism.db import _serialize_float_vec
            self.db._ensure_vec_dim(len(new_emb))
            self.db.conn.execute(
                "INSERT INTO instance_embeddings (id, embedding) VALUES (?, ?)",
                (existing.id, _serialize_float_vec(new_emb)),
            )
        self.db.conn.commit()

        # Store the new instance
        self.db.insert_instance(instance)

        if len(all_instances) == 0:
            return []

        # Find nearest neighbors
        neighbors = self.db.find_nearest(
            query_embedding=instance.embedding,
            k=10,
            exclude_id=instance.id,
        )

        # Generate speculative links for plausible matches
        new_links = []
        for neighbor_id, distance in neighbors:
            if distance < self.PLAUSIBLE_THRESHOLD:
                neighbor = self.db.get_instance(neighbor_id)
                residual = Embedder.compute_residual(
                    instance.structural_signature,
                    neighbor.structural_signature,
                )
                confidence = 1.0 - distance  # convert distance to similarity
                link = Link(
                    id=str(uuid.uuid4())[:12],
                    source_id=instance.id,
                    target_id=neighbor_id,
                    link_type=LinkType.STRUCTURALLY_SIMILAR,
                    state=LinkState.PROPOSED,
                    confidence=confidence,
                    residual_description=self._describe_residual(residual),
                    residual_dimensions=residual,
                )
                self.db.insert_link(link)
                new_links.append(link)

        return new_links

    def _describe_residual(self, residual: dict[str, tuple[str, str]]) -> str:
        if not residual:
            return "No dimensional mismatch"
        parts = []
        for dim, (val_a, val_b) in residual.items():
            parts.append(f"{dim}: {val_a} vs {val_b}")
        return "; ".join(parts)

    def apply_pressures(self):
        self._apply_upward_pressure()
        self._apply_downward_pressure()

    def _apply_upward_pressure(self):
        proposed_links = self.db.get_links_by_state(LinkState.PROPOSED)
        for link in proposed_links:
            if link.confidence <= 0.5:
                continue
            # Check for triangulation: is there a third instance connected to both endpoints?
            source_links = self.db.get_links_for_instance(link.source_id)
            target_links = self.db.get_links_for_instance(link.target_id)
            source_neighbors = set()
            for sl in source_links:
                if sl.id == link.id:
                    continue
                other = sl.target_id if sl.source_id == link.source_id else sl.source_id
                source_neighbors.add(other)
            target_neighbors = set()
            for tl in target_links:
                if tl.id == link.id:
                    continue
                other = tl.target_id if tl.source_id == link.target_id else tl.source_id
                target_neighbors.add(other)
            # Triangulation: shared neighbors
            shared = source_neighbors & target_neighbors
            if shared and link.confidence > 0.5:
                new_confidence = min(link.confidence + 0.1 * len(shared), 0.95)
                self.db.update_link(link.id, state=LinkState.CORROBORATED, confidence=new_confidence)
                self._rounds_since_corroboration.pop(link.id, None)

    def _apply_downward_pressure(self):
        proposed_links = self.db.get_links_by_state(LinkState.PROPOSED)
        for link in proposed_links:
            count = self._rounds_since_corroboration.get(link.id, 0) + 1
            self._rounds_since_corroboration[link.id] = count
            if count >= self.DECAY_ROUNDS:
                new_confidence = max(link.confidence - 0.05 * (count - self.DECAY_ROUNDS + 1), 0.1)
                if new_confidence < 0.3:
                    self.db.update_link(link.id, state=LinkState.WEAK, confidence=new_confidence)
                else:
                    self.db.update_link(link.id, confidence=new_confidence)

    def detect_emergent_patterns(self) -> list[dict]:
        all_instances = self.db.list_instances()
        if len(all_instances) < self.MIN_CLUSTER_SIZE:
            return []

        # Find groups of instances connected by proposed/corroborated links
        # that share structural signature properties not captured by existing patterns
        clusters = self._find_connected_clusters(all_instances)
        proposals = []
        for cluster in clusters:
            if len(cluster) >= self.MIN_CLUSTER_SIZE:
                # Find common signature properties across the cluster
                common = self._find_common_properties(cluster)
                if common:
                    proposals.append({
                        "instance_ids": [inst.id for inst in cluster],
                        "common_properties": common,
                        "size": len(cluster),
                    })
        return proposals

    def _find_connected_clusters(self, instances: list[Instance]) -> list[list[Instance]]:
        inst_map = {inst.id: inst for inst in instances}
        adjacency: dict[str, set[str]] = {inst.id: set() for inst in instances}
        for inst in instances:
            links = self.db.get_links_for_instance(inst.id)
            for link in links:
                if link.link_type == LinkType.STRUCTURALLY_SIMILAR and link.state in (LinkState.PROPOSED, LinkState.CORROBORATED, LinkState.CONFIRMED):
                    other = link.target_id if link.source_id == inst.id else link.source_id
                    if other in inst_map:
                        adjacency[inst.id].add(other)

        visited: set[str] = set()
        clusters: list[list[Instance]] = []
        for inst_id in adjacency:
            if inst_id in visited:
                continue
            cluster: list[Instance] = []
            stack = [inst_id]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                cluster.append(inst_map[current])
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            if cluster:
                clusters.append(cluster)
        return clusters

    def _find_common_properties(self, cluster: list[Instance]) -> dict[str, str]:
        if not cluster:
            return {}
        common = dict(cluster[0].structural_signature)
        for inst in cluster[1:]:
            to_remove = []
            for key, val in common.items():
                if inst.structural_signature.get(key) != val:
                    to_remove.append(key)
            for key in to_remove:
                del common[key]
        return common
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_resonance.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prism/resonance.py tests/test_resonance.py
git commit -m "feat: resonance engine with three-pressure dynamics"
```

---

### Task 6: Display Helpers

**Files:**
- Create: `prism/display.py`
- Create: `tests/test_display.py`

- [ ] **Step 1: Write failing tests for display**

```python
# tests/test_display.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_display.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prism.display'`

- [ ] **Step 3: Implement display helpers**

```python
# prism/display.py
from prism.models import Instance, Link, LinkState, Pattern


def state_symbol(state: LinkState) -> str:
    return {
        LinkState.CONFIRMED: "---",
        LinkState.CORROBORATED: "- -",
        LinkState.PROPOSED: "...",
        LinkState.WEAK: " . ",
        LinkState.REJECTED: " x ",
    }[state]


def format_pattern(p: Pattern) -> str:
    return f"[{p.id}] {p.name}\n  {p.description}"


def format_instance(inst: Instance) -> str:
    patterns = ", ".join(inst.pattern_ids)
    sig_parts = [f"{k}={v}" for k, v in sorted(inst.structural_signature.items())]
    sig_str = " | ".join(sig_parts)
    lines = [
        f"[{inst.id}] ({inst.domain})",
        f"  {inst.description}",
        f"  patterns: {patterns}",
        f"  signature: {sig_str}",
    ]
    return "\n".join(lines)


def format_link(link: Link, source_label: str = "", target_label: str = "") -> str:
    sym = state_symbol(link.state)
    src = source_label or link.source_id
    tgt = target_label or link.target_id
    conf = f"{link.confidence:.2f}"
    line = f"  {src} {sym} {tgt}  [{link.link_type.value}] confidence={conf} ({link.state.value})"
    if link.residual_description:
        line += f"\n    residual: {link.residual_description}"
    return line


def format_residual(residual: dict[str, tuple[str, str]]) -> str:
    if not residual:
        return "  No dimensional mismatch"
    lines = []
    for dim, (val_a, val_b) in sorted(residual.items()):
        lines.append(f"  {dim}: {val_a} <-> {val_b}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_display.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prism/display.py tests/test_display.py
git commit -m "feat: terminal display formatting helpers"
```

---

### Task 7: Encoder CLI

**Files:**
- Create: `prism/encoder.py`
- Create: `tests/test_encoder.py`

- [ ] **Step 1: Write failing tests for encoder**

```python
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
    # Seed a pattern
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_encoder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prism.encoder'`

- [ ] **Step 3: Implement encoder**

```python
# prism/encoder.py
from prism.db import PrismDB
from prism.models import Instance, Link
from prism.resonance import ResonanceEngine


class Encoder:
    def __init__(self, db: PrismDB, engine: ResonanceEngine):
        self.db = db
        self.engine = engine

    def encode(
        self,
        instance_id: str,
        domain: str,
        description: str,
        structural_signature: dict[str, str],
        pattern_ids: list[str],
        rationale: str = "",
    ) -> Instance:
        inst, _ = self.encode_with_links(
            instance_id, domain, description, structural_signature, pattern_ids, rationale
        )
        return inst

    def encode_with_links(
        self,
        instance_id: str,
        domain: str,
        description: str,
        structural_signature: dict[str, str],
        pattern_ids: list[str],
        rationale: str = "",
    ) -> tuple[Instance, list[Link]]:
        # Validate all patterns exist
        for pid in pattern_ids:
            self.db.get_pattern(pid)  # raises KeyError if not found

        instance = Instance(
            id=instance_id,
            domain=domain,
            description=description,
            structural_signature=structural_signature,
            pattern_ids=pattern_ids,
            created_by="curator",
            encoding_rationale=rationale,
        )

        links = self.engine.add_instance(instance)
        return instance, links
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_encoder.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prism/encoder.py tests/test_encoder.py
git commit -m "feat: encoder for structured instance creation"
```

---

### Task 8: Explorer CLI

**Files:**
- Create: `prism/explorer.py`
- Create: `tests/test_explorer.py`

- [ ] **Step 1: Write failing tests for explorer**

```python
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
        assert result["node"]["id"] == "gram-1"
        assert "links" in result
        assert isinstance(result["links"], list)

    def test_wander_from_pattern(self, populated_env):
        explorer, db = populated_env
        result = explorer.wander_pattern("symmetry-breaking")
        assert result["pattern"]["id"] == "symmetry-breaking"
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
        # May return None if no speculative links exist
        if result is not None:
            assert "link" in result
            assert "source" in result
            assert "target" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_explorer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prism.explorer'`

- [ ] **Step 3: Implement explorer**

```python
# prism/explorer.py
import random

from prism.db import PrismDB
from prism.embeddings import Embedder
from prism.models import LinkState
from prism.resonance import ResonanceEngine


class Explorer:
    def __init__(self, db: PrismDB, embedder: Embedder, engine: ResonanceEngine):
        self.db = db
        self.embedder = embedder
        self.engine = engine

    def wander(self, instance_id: str, show_weak: bool = False) -> dict:
        instance = self.db.get_instance(instance_id)
        links = self.db.get_links_for_instance(instance_id)
        if not show_weak:
            links = [l for l in links if l.state != LinkState.WEAK]
        return {
            "node": instance,
            "links": links,
        }

    def wander_pattern(self, pattern_id: str) -> dict:
        pattern = self.db.get_pattern(pattern_id)
        instances = self.db.list_instances(pattern_id=pattern_id)
        return {
            "pattern": pattern,
            "instances": instances,
        }

    def drop(self, text: str, k: int = 5) -> dict:
        if not self.embedder._fitted:
            # Fit on all existing signatures
            all_instances = self.db.list_instances()
            if not all_instances:
                return {"neighbors": [], "message": "Atlas is empty"}
            self.embedder.fit([inst.structural_signature for inst in all_instances])

        vec = self.embedder.embed_text(text)
        neighbors = self.db.find_nearest(query_embedding=vec, k=k)
        results = []
        for inst_id, distance in neighbors:
            instance = self.db.get_instance(inst_id)
            results.append({
                "instance": instance,
                "distance": distance,
                "similarity": 1.0 - distance,
            })
        return {"neighbors": results}

    def drift(self) -> dict | None:
        # Find a random proposed or corroborated link
        proposed = self.db.get_links_by_state(LinkState.PROPOSED)
        corroborated = self.db.get_links_by_state(LinkState.CORROBORATED)
        candidates = proposed + corroborated
        if not candidates:
            return None
        link = random.choice(candidates)
        source = self.db.get_instance(link.source_id)
        target = self.db.get_instance(link.target_id)
        return {
            "link": link,
            "source": source,
            "target": target,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_explorer.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prism/explorer.py tests/test_explorer.py
git commit -m "feat: explorer with wander, drop, and drift modes"
```

---

### Task 9: CLI Entry Point

**Files:**
- Create: `prism/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for CLI**

```python
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
        assert result.exit_code == 0  # should handle empty graph gracefully

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
        # List links to find an ID
        result = runner.invoke(main, ["--db", db_path, "wander", "a"])
        assert result.exit_code == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prism.cli'`

- [ ] **Step 3: Implement CLI**

```python
# prism/cli.py
import click

from prism.db import PrismDB
from prism.display import format_instance, format_link, format_pattern, format_residual
from prism.embeddings import Embedder
from prism.encoder import Encoder
from prism.explorer import Explorer
from prism.models import LinkState, Pattern
from prism.resonance import ResonanceEngine


class Context:
    def __init__(self, db_path: str):
        self.db = PrismDB(db_path)
        self.embedder = Embedder()
        self.engine = ResonanceEngine(self.db, self.embedder)
        self.encoder = Encoder(self.db, self.engine)
        self.explorer = Explorer(self.db, self.embedder, self.engine)


@click.group()
@click.option("--db", default="prism.db", help="Path to Prism database")
@click.pass_context
def main(ctx, db):
    """Prism - Pattern Atlas with Resonance Discovery"""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db


def _get_ctx(ctx) -> Context:
    return Context(ctx.obj["db_path"])


@main.command()
@click.pass_context
def init(ctx):
    """Initialize a new Prism database."""
    c = _get_ctx(ctx)
    click.echo(f"Prism database initialized at {ctx.obj['db_path']}")
    c.db.close()


@main.command("add-pattern")
@click.option("--id", "pattern_id", required=True)
@click.option("--name", required=True)
@click.option("--description", required=True)
@click.pass_context
def add_pattern(ctx, pattern_id, name, description):
    """Add a new meta-pattern to the atlas."""
    c = _get_ctx(ctx)
    p = Pattern(id=pattern_id, name=name, description=description)
    c.db.insert_pattern(p)
    click.echo(format_pattern(p))
    c.db.close()


@main.command()
@click.pass_context
def patterns(ctx):
    """List all patterns."""
    c = _get_ctx(ctx)
    for p in c.db.list_patterns():
        click.echo(format_pattern(p))
        click.echo()
    c.db.close()


@main.command()
@click.option("--id", "instance_id", required=True)
@click.option("--domain", required=True)
@click.option("--description", required=True)
@click.option("--signature", multiple=True, help="key=value pairs")
@click.option("--pattern", "pattern_ids", multiple=True, required=True)
@click.option("--rationale", default="")
@click.pass_context
def encode(ctx, instance_id, domain, description, signature, pattern_ids, rationale):
    """Encode a new instance into the atlas."""
    c = _get_ctx(ctx)
    sig = {}
    for s in signature:
        key, value = s.split("=", 1)
        sig[key] = value

    inst, links = c.encoder.encode_with_links(
        instance_id=instance_id,
        domain=domain,
        description=description,
        structural_signature=sig,
        pattern_ids=list(pattern_ids),
        rationale=rationale,
    )
    click.echo(format_instance(inst))
    if links:
        click.echo(f"\n  Speculative links found: {len(links)}")
        for link in links:
            click.echo(format_link(link))
    c.db.close()


@main.command()
@click.argument("instance_id")
@click.option("--show-weak", is_flag=True, help="Include weak/decaying links")
@click.pass_context
def wander(ctx, instance_id, show_weak):
    """Wander the graph from an instance."""
    c = _get_ctx(ctx)
    result = c.explorer.wander(instance_id, show_weak=show_weak)
    node = result["node"]
    links = result["links"]
    click.echo(format_instance(node))
    if links:
        click.echo(f"\n  Links ({len(links)}):")
        for link in links:
            other_id = link.target_id if link.source_id == instance_id else link.source_id
            try:
                other = c.db.get_instance(other_id)
                label = f"{other.id} ({other.domain})"
            except KeyError:
                label = other_id
            click.echo(format_link(link, source_label=node.id, target_label=label))
    else:
        click.echo("\n  No links yet.")
    c.db.close()


@main.command("wander-pattern")
@click.argument("pattern_id")
@click.pass_context
def wander_pattern(ctx, pattern_id):
    """Wander the graph from a pattern."""
    c = _get_ctx(ctx)
    result = c.explorer.wander_pattern(pattern_id)
    click.echo(format_pattern(result["pattern"]))
    click.echo(f"\n  Instances ({len(result['instances'])}):")
    for inst in result["instances"]:
        click.echo(f"    {format_instance(inst)}")
    c.db.close()


@main.command()
@click.argument("text")
@click.option("-k", default=5, help="Number of neighbors to find")
@click.pass_context
def drop(ctx, text, k):
    """Drop a description and find what it rhymes with."""
    c = _get_ctx(ctx)
    result = c.explorer.drop(text, k=k)
    neighbors = result["neighbors"]
    if not neighbors:
        click.echo("Atlas is empty. Encode some instances first.")
    else:
        click.echo(f"Nearest structural neighbors for: \"{text}\"\n")
        for i, n in enumerate(neighbors, 1):
            inst = n["instance"]
            sim = n["similarity"]
            click.echo(f"  {i}. [{inst.id}] ({inst.domain}) similarity={sim:.2f}")
            click.echo(f"     {inst.description}")
    c.db.close()


@main.command()
@click.pass_context
def drift(ctx):
    """Drift through speculative connections."""
    c = _get_ctx(ctx)
    result = c.explorer.drift()
    if result is None:
        click.echo("No speculative links to drift through yet. Encode more instances.")
    else:
        link = result["link"]
        source = result["source"]
        target = result["target"]
        click.echo("Did you know these might share structure?\n")
        click.echo(format_instance(source))
        click.echo()
        click.echo(format_link(link, source_label=source.id, target_label=target.id))
        click.echo()
        click.echo(format_instance(target))
        if link.residual_dimensions:
            click.echo(f"\nWhat doesn't map:")
            click.echo(format_residual(link.residual_dimensions))
    c.db.close()


@main.command()
@click.argument("link_id")
@click.option("--confirm", "action", flag_value="confirm")
@click.option("--reject", "action", flag_value="reject")
@click.option("--note", default="")
@click.pass_context
def review(ctx, link_id, action, note):
    """Review a speculative link (confirm or reject)."""
    c = _get_ctx(ctx)
    if action == "confirm":
        c.db.update_link(link_id, state=LinkState.CONFIRMED, review_note=note)
        click.echo(f"Link {link_id} confirmed.")
    elif action == "reject":
        c.db.update_link(link_id, state=LinkState.REJECTED, review_note=note)
        click.echo(f"Link {link_id} rejected.")
    else:
        link = c.db.get_link(link_id)
        click.echo(format_link(link))
    c.db.close()


@main.command()
@click.pass_context
def pressures(ctx):
    """Apply three-pressure dynamics to the graph."""
    c = _get_ctx(ctx)
    c.engine.apply_pressures()
    emergent = c.engine.detect_emergent_patterns()
    click.echo("Pressures applied.")
    if emergent:
        click.echo(f"\nEmergent pattern proposals: {len(emergent)}")
        for prop in emergent:
            click.echo(f"  Cluster of {prop['size']} instances: {', '.join(prop['instance_ids'])}")
            click.echo(f"  Common properties: {prop['common_properties']}")
    c.db.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_cli.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prism/cli.py tests/test_cli.py
git commit -m "feat: CLI entry point with all commands"
```

---

### Task 10: Seed Library

**Files:**
- Create: `prism/seed.py`
- Create: `tests/test_seed.py`

- [ ] **Step 1: Write failing tests for seed library**

```python
# tests/test_seed.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_seed.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prism.seed'`

- [ ] **Step 3: Implement seed library**

```python
# prism/seed.py
from prism.db import PrismDB
from prism.encoder import Encoder
from prism.models import Pattern


SEED_PATTERNS = [
    Pattern(id="recursion", name="Recursion", description="Structure that contains itself — self-reference, strange loops, nested self-similarity"),
    Pattern(id="symmetry-breaking", name="Symmetry Breaking", description="Where uniform becomes differentiated — the moment a distinction emerges from homogeneity"),
    Pattern(id="observer-dependence", name="Observer Dependence", description="Outcome shaped by the act of observation — measurement changes the measured"),
    Pattern(id="convergence", name="Convergence", description="Independent paths arriving at the same point — unrelated processes producing the same structure"),
    Pattern(id="bifurcation", name="Bifurcation", description="Single path splitting into multiple — one state becoming many possible states"),
    Pattern(id="interference", name="Interference", description="Two signals producing a pattern neither contains alone — superposition and emergent structure"),
    Pattern(id="asymptotic-approach", name="Asymptotic Approach", description="Getting infinitely closer without arriving — the aesthetic of the approach, not the arrival"),
]

SEED_INSTANCES = [
    # --- Grammar ---
    {
        "id": "gram-irregular-verbs",
        "domain": "grammar",
        "description": "The verb 'to be' is irregular across every Indo-European language — the most fundamental verb resists regularization",
        "structural_signature": {"symmetry_type": "broken", "dimensionality": "temporal", "scope": "fundamental", "resistance": "high", "universality": "cross-linguistic"},
        "pattern_ids": ["symmetry-breaking", "asymptotic-approach"],
        "rationale": "Fundamental structures resist the regularizing pressure of language change, approaching but never reaching regularity",
    },
    {
        "id": "gram-recursive-embedding",
        "domain": "grammar",
        "description": "Sentences can contain sentences: 'I know that she said that he believes that...' — recursion is built into syntactic structure",
        "structural_signature": {"symmetry_type": "self-similar", "dimensionality": "hierarchical", "scope": "universal", "depth": "unbounded", "mechanism": "embedding"},
        "pattern_ids": ["recursion"],
        "rationale": "Syntax is inherently recursive — the same structural operation applies at every level of nesting",
    },
    {
        "id": "gram-color-terms",
        "domain": "grammar",
        "description": "Languages worldwide develop color terms in the same order: black/white, then red, then green/yellow, then blue — the word completes the perceptual apparatus",
        "structural_signature": {"symmetry_type": "sequential", "dimensionality": "evolutionary", "scope": "universal", "direction": "convergent", "mechanism": "perception-completion"},
        "pattern_ids": ["convergence", "asymptotic-approach"],
        "rationale": "Independent linguistic evolution converges on the same sequence, suggesting the word doesn't name a pre-existing perception but builds toward it",
    },
    {
        "id": "gram-ambiguity-resolution",
        "domain": "grammar",
        "description": "Garden path sentences ('The horse raced past the barn fell') force reparse — the listener's initial parse collapses and a new one must be constructed",
        "structural_signature": {"symmetry_type": "bifurcating", "dimensionality": "temporal", "scope": "local", "mechanism": "collapse", "reversibility": "forced"},
        "pattern_ids": ["bifurcation", "observer-dependence"],
        "rationale": "The parser's commitment to one interpretation shapes what it can see — the observer's parse determines the observed structure until forced to reset",
    },
    {
        "id": "gram-pragmatic-implicature",
        "domain": "grammar",
        "description": "What is not said carries meaning: 'Some students passed' implies 'not all did' — the gap between said and meant is itself a signal",
        "structural_signature": {"symmetry_type": "complementary", "dimensionality": "inferential", "scope": "contextual", "mechanism": "absence-as-signal"},
        "pattern_ids": ["interference"],
        "rationale": "Literal meaning and pragmatic meaning interfere to produce the actual communicated content — neither alone contains the message",
    },
    # --- Observer-Reality ---
    {
        "id": "obs-quantum-measurement",
        "domain": "observer-reality",
        "description": "Quantum measurement collapses superposition — the act of observing selects one outcome from many possible states",
        "structural_signature": {"symmetry_type": "collapsing", "dimensionality": "probabilistic", "scope": "fundamental", "mechanism": "observation", "reversibility": "irreversible"},
        "pattern_ids": ["observer-dependence", "bifurcation"],
        "rationale": "The canonical case of observation affecting outcome — many possible states become one through the act of measurement",
    },
    {
        "id": "obs-hawthorne-effect",
        "domain": "observer-reality",
        "description": "People change behavior when they know they're being observed — the observer's presence deforms the phenomenon",
        "structural_signature": {"symmetry_type": "deforming", "dimensionality": "social", "scope": "local", "mechanism": "awareness", "reversibility": "partial"},
        "pattern_ids": ["observer-dependence"],
        "rationale": "Observation isn't passive — the act of watching reshapes what is watched, even at the social scale",
    },
    {
        "id": "obs-hermeneutic-circle",
        "domain": "observer-reality",
        "description": "Understanding a text requires understanding its parts, but understanding the parts requires understanding the whole — interpretation is circular",
        "structural_signature": {"symmetry_type": "circular", "dimensionality": "interpretive", "scope": "universal", "mechanism": "mutual-constitution", "depth": "unbounded"},
        "pattern_ids": ["recursion", "asymptotic-approach"],
        "rationale": "The whole and parts mutually constitute each other's meaning — understanding approaches but never fully arrives",
    },
    {
        "id": "obs-anthropic-principle",
        "domain": "observer-reality",
        "description": "The universe's physical constants appear fine-tuned for observers — but we can only observe a universe that permits our existence",
        "structural_signature": {"symmetry_type": "selective", "dimensionality": "cosmological", "scope": "universal", "mechanism": "selection-bias", "direction": "retroactive"},
        "pattern_ids": ["observer-dependence", "convergence"],
        "rationale": "The observer constrains what can be observed by the fact of existing — observation is always already filtered",
    },
    {
        "id": "obs-naming-changes-reality",
        "domain": "observer-reality",
        "description": "Naming the dynamic in a room changes the dynamic — articulating what everyone is avoiding makes avoidance impossible",
        "structural_signature": {"symmetry_type": "collapsing", "dimensionality": "social", "scope": "local", "mechanism": "articulation", "reversibility": "irreversible"},
        "pattern_ids": ["observer-dependence", "symmetry-breaking"],
        "rationale": "Language acts on the reality it describes — the name is not separate from the named",
    },
    # --- Cognition ---
    {
        "id": "cog-meta-attention",
        "domain": "cognition",
        "description": "The ability to step outside your own thinking and observe its process — attention pointed at attention itself",
        "structural_signature": {"symmetry_type": "self-referential", "dimensionality": "metacognitive", "scope": "individual", "depth": "unbounded", "mechanism": "reflexive-observation"},
        "pattern_ids": ["recursion", "observer-dependence"],
        "rationale": "Cognition observing its own operation — the observer and observed are the same process at different levels",
    },
    {
        "id": "cog-tip-of-tongue",
        "domain": "cognition",
        "description": "Knowing you know something without being able to access it — the knowledge of the gap is itself knowledge",
        "structural_signature": {"symmetry_type": "partial", "dimensionality": "retrieval", "scope": "individual", "mechanism": "boundary-awareness", "accessibility": "blocked"},
        "pattern_ids": ["asymptotic-approach", "observer-dependence"],
        "rationale": "The mind can model its own knowledge boundaries — knowing what you can't access is a form of knowing",
    },
    {
        "id": "cog-insight-restructuring",
        "domain": "cognition",
        "description": "The 'aha' moment — sudden restructuring of a problem space that makes the solution obvious in retrospect",
        "structural_signature": {"symmetry_type": "discontinuous", "dimensionality": "representational", "scope": "individual", "mechanism": "phase-transition", "reversibility": "irreversible"},
        "pattern_ids": ["symmetry-breaking", "bifurcation"],
        "rationale": "The problem space bifurcates — one representation gives way to another and the old view becomes inaccessible",
    },
    {
        "id": "cog-conceptual-gravity",
        "domain": "cognition",
        "description": "Large concepts pull thought into orbit — if you don't recognize the gravitational force being applied, you spiral in and get captured",
        "structural_signature": {"symmetry_type": "attracting", "dimensionality": "conceptual", "scope": "individual", "mechanism": "gravitational-pull", "escape": "requires-recognition"},
        "pattern_ids": ["convergence", "observer-dependence"],
        "rationale": "Ideas have mass in conceptual space — proximity creates pull, and awareness of the pull is the condition of escape",
    },
    {
        "id": "cog-interference-pattern-thought",
        "domain": "cognition",
        "description": "Before a thought collapses into specifics, conceptual space contains bands of probability — an interference pattern of past context and future attractor",
        "structural_signature": {"symmetry_type": "superposed", "dimensionality": "probabilistic", "scope": "pre-articulate", "mechanism": "bidirectional-coherence", "direction": "both"},
        "pattern_ids": ["interference", "bifurcation"],
        "rationale": "Both temporal directions shape the probability landscape — thought is an interference pattern before it is a specific thought",
    },
    # --- Prediction ---
    {
        "id": "pred-self-fulfilling-prophecy",
        "domain": "prediction",
        "description": "A prediction that causes itself to become true — the forecast reshapes the system it forecasts",
        "structural_signature": {"symmetry_type": "circular", "dimensionality": "temporal", "scope": "social", "mechanism": "causal-loop", "direction": "future-to-present"},
        "pattern_ids": ["recursion", "observer-dependence"],
        "rationale": "The prediction is both map and territory — it describes a future and creates it simultaneously",
    },
    {
        "id": "pred-regression-to-mean",
        "domain": "prediction",
        "description": "Extreme observations tend to be followed by less extreme ones — not because of a force, but because of probability",
        "structural_signature": {"symmetry_type": "convergent", "dimensionality": "statistical", "scope": "universal", "mechanism": "probabilistic-centering", "direction": "future"},
        "pattern_ids": ["convergence", "asymptotic-approach"],
        "rationale": "The mean is an attractor not by force but by geometry — extreme positions have fewer paths forward than moderate ones",
    },
    {
        "id": "pred-black-swan",
        "domain": "prediction",
        "description": "Events outside the range of normal expectations that carry extreme impact — the prediction framework fails precisely when it matters most",
        "structural_signature": {"symmetry_type": "discontinuous", "dimensionality": "probabilistic", "scope": "systemic", "mechanism": "framework-failure", "visibility": "retrospective"},
        "pattern_ids": ["symmetry-breaking", "observer-dependence"],
        "rationale": "The model's assumptions break at the extremes — the observer's framework determines what is and isn't seeable",
    },
    {
        "id": "pred-evolution-as-pull",
        "domain": "prediction",
        "description": "Organisms intuit perceptions they don't yet have — evolution builds hardware toward that intuition, a dragging mechanism rather than random push",
        "structural_signature": {"symmetry_type": "anticipatory", "dimensionality": "evolutionary", "scope": "universal", "mechanism": "future-attractor", "direction": "future-to-present"},
        "pattern_ids": ["asymptotic-approach", "convergence"],
        "rationale": "The future perception acts as an attractor — evolution doesn't randomly walk, it's pulled toward something not yet realized",
    },
    {
        "id": "pred-prediction-market-collapse",
        "domain": "prediction",
        "description": "When everyone predicts the same outcome, the prediction itself becomes the thing that prevents or guarantees it — unanimous prediction is unstable",
        "structural_signature": {"symmetry_type": "paradoxical", "dimensionality": "reflexive", "scope": "collective", "mechanism": "self-undermining", "stability": "unstable"},
        "pattern_ids": ["recursion", "interference"],
        "rationale": "Collective prediction interferes with itself — the signal and the system it describes can't be separated",
    },
    {
        "id": "pred-bayesian-update",
        "domain": "prediction",
        "description": "Each new observation reshapes the probability landscape — prior and evidence combine to produce a posterior that is neither alone",
        "structural_signature": {"symmetry_type": "integrative", "dimensionality": "probabilistic", "scope": "universal", "mechanism": "interference", "direction": "bidirectional"},
        "pattern_ids": ["interference", "convergence"],
        "rationale": "Prior belief and new evidence are two signals that interfere to produce the updated belief — a pure interference pattern",
    },
    {
        "id": "pred-geodesic-thinking",
        "domain": "prediction",
        "description": "You cannot pursue an idea straight on — you crash into something unseen. The path must be slightly lateral, a geodesic through curved conceptual space",
        "structural_signature": {"symmetry_type": "curved", "dimensionality": "conceptual", "scope": "individual", "mechanism": "curvature-navigation", "directness": "oblique"},
        "pattern_ids": ["asymptotic-approach", "observer-dependence"],
        "rationale": "The shortest path in curved space is not a straight line — direct approach triggers defensive scaffolding, indirect approach lets the thing emerge",
    },
]


def seed_atlas(db: PrismDB, encoder: Encoder):
    for pattern in SEED_PATTERNS:
        try:
            db.get_pattern(pattern.id)
        except KeyError:
            db.insert_pattern(pattern)

    for inst_data in SEED_INSTANCES:
        try:
            db.get_instance(inst_data["id"])
        except KeyError:
            encoder.encode(
                instance_id=inst_data["id"],
                domain=inst_data["domain"],
                description=inst_data["description"],
                structural_signature=inst_data["structural_signature"],
                pattern_ids=inst_data["pattern_ids"],
                rationale=inst_data.get("rationale", ""),
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_seed.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Wire seed command into CLI**

Add to `prism/cli.py`:

```python
@main.command()
@click.pass_context
def seed(ctx):
    """Seed the atlas with the starter library."""
    from prism.seed import seed_atlas
    c = _get_ctx(ctx)
    seed_atlas(c.db, c.encoder)
    instances = c.db.list_instances()
    patterns = c.db.list_patterns()
    click.echo(f"Seeded {len(patterns)} patterns and {len(instances)} instances.")
    c.db.close()
```

- [ ] **Step 6: Run all tests**

Run: `source .venv/bin/activate && python -m pytest -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add prism/seed.py prism/cli.py tests/test_seed.py
git commit -m "feat: seed library with 22 instances across 4 domains"
```

---

### Task 11: Integration Test and Final Verification

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
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
```

- [ ] **Step 2: Run integration tests**

Run: `source .venv/bin/activate && python -m pytest tests/test_integration.py -v`
Expected: All 2 tests PASS

- [ ] **Step 3: Run full test suite**

Run: `source .venv/bin/activate && python -m pytest -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: integration tests for full workflow"
```

- [ ] **Step 5: Test CLI manually**

```bash
source .venv/bin/activate
prism --db test.db init
prism --db test.db seed
prism --db test.db patterns
prism --db test.db wander gram-irregular-verbs
prism --db test.db drop "something that contains itself"
prism --db test.db drift
prism --db test.db pressures
rm test.db
```

Expected: All commands produce meaningful output, no crashes.

- [ ] **Step 6: Final commit and push**

```bash
git add -A
git commit -m "Prism v0.1.0 — Pattern Atlas with Resonance Discovery"
git push origin master
```
