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
            (
                link.id,
                link.source_id,
                link.target_id,
                link.link_type.value,
                link.state.value,
                link.confidence,
                link.residual_description,
                json.dumps({k: list(v) for k, v in link.residual_dimensions.items()}),
                link.review_note,
            ),
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
