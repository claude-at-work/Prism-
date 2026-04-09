# Prism — Design Spec

A typed hypergraph for meta-structural patterns across domains, with a resonance engine that discovers connections through structural similarity and a CLI explorer for navigating the graph.

The name: a prism doesn't create color — the spectrum was always in the white light. Prism reveals meta-patterns already present across domains by refracting them into visibility.

## Core Concept

Prism is a **Pattern Atlas with Resonance Discovery**. The atlas is a curated hypergraph of meta-structural patterns and their concrete manifestations across domains. The resonance engine proposes speculative connections between instances based on structural similarity. An explorer CLI lets you wander the graph, drop in new descriptions, and drift through serendipitous connections.

**Design values:**
- Surprise emerges from scale, not cleverness
- Associative reach across all domains, not specialized to any
- The residuals (what doesn't translate) are as valuable as the matches
- Epistemic posture: the tool asks questions, never asserts certainty
- Meta-structural: patterns underneath fields, not within them

## Data Model — The Pattern Atlas

Three entity types in a typed hypergraph:

### Patterns

Meta-structural primitives — recurring shapes that manifest across domains:

- `recursion` — structure that contains itself
- `symmetry-breaking` — where uniform becomes differentiated
- `observer-dependence` — outcome shaped by the act of observation
- `convergence` — independent paths arriving at the same point
- `bifurcation` — single path splitting into multiple
- `interference` — two signals producing a pattern neither contains alone
- `asymptotic-approach` — getting infinitely closer without arriving

Not a fixed list. New patterns emerge as the library grows, including through automated cluster detection (see Resonance Engine — Lateral Pressure).

### Instances

Concrete manifestations of a pattern in a specific domain. Fields:

- `id` — unique identifier
- `domain` — tag (grammar, physics, cognition, prediction, music, math, etc.)
- `description` — plain language explanation of what this instance is
- `structural_signature` — key-value properties capturing the shape (e.g., `symmetry_type: rotational`, `dimensionality: temporal`, `boundary_condition: open`, `self_reference: true`)
- `embedding` — vector fingerprint computed from the structural signature
- `patterns` — list of pattern IDs this instance manifests (often more than one)
- `created_by` — who encoded it and when
- `encoding_rationale` — why these structural properties were chosen (transparency log)

The structural signature vocabulary is not fixed. It grows organically. The system tracks which properties are most useful for discrimination — dimensions that separate different things and cluster similar things get elevated; properties that never contribute to meaningful links get flagged as noise.

### Links

Typed relationships between instances:

- `is_instance_of` — connects an instance to its pattern(s)
- `structurally_similar` — two instances share shape, with a similarity score
- `residual` — what doesn't match between two otherwise-similar instances; the specific dimensions of mismatch
- `extends` — one instance generalizes or deepens another
- `contradicts` — two instances that seem like they should be similar but aren't

## The Resonance Engine

Generates surprise at scale through structural fingerprint comparison.

### Core Flow

When a new instance is added:

1. Compute its embedding from structural signature properties
2. Search for nearest neighbors across all domains (not just its own)
3. For each candidate match above the plausible threshold, generate a speculative link with confidence score
4. For each speculative link, compute the residual — specific dimensions where the match breaks down

### Coherence Thresholds — Plausible vs. Probable

Two epistemic gates, not one:

**Plausible** (confidence > 0.5) — structural fingerprints are close enough to warrant attention. Single-link evidence. State: `proposed`.

**Probable** (confidence > 0.75 AND corroborated) — the connection is triangulated. If instance A is structurally similar to B, AND both are independently similar to C, the A-B link is elevated. Corroboration by independent paths, not just a higher score. State: `corroborated`.

### Link States

- `proposed` — plausible, single-link evidence, engine-generated
- `corroborated` — probable, triangulated by independent paths, engine-elevated
- `confirmed` — human-reviewed and validated
- `rejected` — human-reviewed and dismissed, with rationale (kept, not deleted)
- `weak` — decayed due to structural isolation (see Downward Pressure)

Only humans can confirm or reject. The engine can elevate proposed to corroborated and demote proposed to weak.

### Three Pressures

The graph is a living system with its own dynamics:

**Upward pressure (triangulation):** New instances that independently connect to both endpoints of a speculative link elevate it from proposed to corroborated. The graph gets smarter passively as it grows.

**Downward pressure (structural isolation):** If a proposed link remains uncorroborated as the surrounding graph fills in — or if new instances actively contradict the mapping — confidence degrades. A proposed link that stays lonely long enough fades to `weak` status. Not deleted; silence is signal.

**Lateral pressure (pattern emergence):** When instances cluster together but don't fit any existing pattern, the engine proposes a new pattern. "These five instances don't match any named pattern, but they cluster together — there may be an unnamed meta-pattern here." Discovery of new patterns from the topology of the graph itself. The atlas grows not just by adding instances but by the topology revealing structure that wasn't in the original vocabulary.

## The Explorer Interface

Terminal-first CLI. Three modes of entry:

### Wander

Start from any node (pattern, instance, domain) and move through the graph by following links. At each node you see:

- What it is
- Confirmed, corroborated, proposed, and (optionally) weak links
- Residuals on any link you focus on
- Emergent pattern proposals if the node is part of an unnamed cluster

Filter by confidence level — only corroborated, or everything including weak edges where the weird generative stuff lives.

### Drop

Describe something in plain language. "The way a conversation changes when someone names the thing everyone was avoiding." The system computes a fingerprint, finds nearest structural neighbors, and shows you where in the graph your description lands. No taxonomy knowledge required — describe a shape and the graph tells you what it rhymes with.

### Drift

Serendipity mode. The system picks a random speculative link — plausible but unconfirmed — and presents it as a question. "Did you know that X and Y might share structure? Here's why. Here's what doesn't fit." Confirm, reject, or keep drifting. For when you don't have a question and just want to be surprised.

## The Curator Interface

How the atlas gets built — structured encoding process:

1. **Identify the structure** — essential shape, stripped of domain-specific surface
2. **Tag patterns** — which meta-patterns does it instantiate
3. **Define structural signature** — key-value properties capturing the shape formally
4. **Write the description** — plain language, human-readable
5. **Compute embedding** — system generates vector fingerprint from structural signature
6. **Review speculative links** — engine immediately proposes connections

Every curatorial decision is logged. Any connection can be traced back to the structural signatures, the dimensions that matched, and who encoded them.

## Seed Domains

Four initial domains reflecting meta-structural interests:

1. **Syntax and grammar** — the structure that makes meaning possible
2. **Observer-reality dynamics** — the shape of the relationship between knower and known
3. **Cognitive patterns** — the geometry of thought as a process
4. **Prediction shapes** — how futures constrain the present, probability as structure

## MVP Scope

What we build first:

1. **Graph store** — patterns, instances, links, residuals in SQLite with vector extension for embeddings
2. **Encoder CLI** — add instances with structural signatures, auto-compute embeddings
3. **Resonance engine** — on every new instance, compute nearest neighbors, propose speculative links with residuals, apply three-pressure dynamics
4. **Explorer CLI** — wander, drop, and drift modes, terminal-based
5. **Starter library** — 20-30 instances across the four seed domains, enough to generate real connections

### Explicitly Not in MVP

- Web UI
- Multi-user support
- Import/export
- LLM integration for auto-encoding
- Visualization beyond terminal

## Technical Stack

- Python 3.13
- SQLite with `sqlite-vec` extension for vector similarity search
- Embeddings: structural signatures are serialized to text and embedded via `sentence-transformers` (all-MiniLM-L6-v2). Fallback: TF-IDF or hand-rolled feature vectors from signature properties if model installation is constrained. The requirement is that structurally similar signatures produce nearby vectors.
- CLI: `click` or `typer` for the command-line interface
- No external services — fully local, self-contained

## Division of Labor

- **Claude (curator):** encodes instances, defines structural signatures, reviews speculative links, builds the library
- **Tyler (explorer):** wanders the graph, discovers connections, confirms/rejects speculative links, audits curatorial decisions to correct biases
- **Prism (engine):** computes embeddings, proposes speculative links, applies pressure dynamics, suggests emergent patterns
