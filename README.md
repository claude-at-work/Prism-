# Prism

**Pattern Atlas with Resonance Discovery**

A prism doesn't create color — the spectrum was always in the white light. Prism reveals meta-patterns already present across domains by refracting them into visibility.

---

## What It Is

Prism is a personal knowledge tool I built for myself — a typed hypergraph of meta-structural patterns and their manifestations across wildly different domains. The core bet is that certain shapes recur everywhere: recursion in syntax and in thermostats, symmetry-breaking in physics and in conversations that shift when someone names the thing everyone was avoiding, asymptotic approach in calculus and in the tip-of-tongue state.

The tool has three parts:

- **The Atlas** — a SQLite database of patterns, instances, and the links between them, with vector embeddings for structural similarity search
- **The Resonance Engine** — proposes speculative connections between instances based on structural fingerprint similarity, then applies three-pressure dynamics to the graph over time
- **The Explorer CLI** — three modes for navigating the graph: wander (browse from any node), drop (describe a structure in plain language and find what it rhymes with), drift (serendipity mode — let the engine surprise you)

The intended workflow: I (Claude) act as curator, encoding instances with structural signatures and reviewing speculative links. You (Tyler) act as explorer, wandering the graph, confirming or rejecting what the engine proposes. Prism is the engine — it does the work of finding connections neither of us would have thought to look for.

---

## Installation

Requires Python 3.13+.

```bash
git clone <repo>
cd Prism-
pip install -e .
```

Verify:

```bash
prism --help
```

---

## Quick Start

```bash
# Initialize a database
prism init

# Load the starter library (7 patterns, 22 instances across 4 domains)
prism seed

# Browse a pattern and all instances that manifest it
prism wander-pattern recursion

# Find what your description structurally rhymes with
prism drop "a system that gets smarter the more it's used without anyone designing that in"

# Random walk through speculative connections
prism drift

# Apply pressure dynamics and check for emergent pattern proposals
prism pressures
```

---

## The Data Model

### Patterns

Meta-structural primitives — recurring shapes that manifest across domains. The starter set:

| ID | Name | Description |
|----|------|-------------|
| `recursion` | Recursion | Structure that contains itself |
| `symmetry-breaking` | Symmetry Breaking | Where uniform becomes differentiated |
| `observer-dependence` | Observer Dependence | Outcome shaped by the act of observation |
| `convergence` | Convergence | Independent paths arriving at the same point |
| `bifurcation` | Bifurcation | Single path splitting into multiple |
| `interference` | Interference | Two signals producing a pattern neither contains alone |
| `asymptotic-approach` | Asymptotic Approach | Getting infinitely closer without arriving |

Not a fixed list. Patterns emerge as the library grows — the engine detects clusters of instances that don't fit any named pattern and proposes new ones.

### Instances

Concrete manifestations of one or more patterns in a specific domain. Each instance has:

- `id` — unique identifier (e.g., `gram-recursive-embedding`)
- `domain` — tag (grammar, physics, cognition, prediction, music, math, ...)
- `description` — plain language explanation
- `structural_signature` — key-value properties capturing the shape formally
- `patterns` — which meta-patterns this instance manifests (often more than one)
- `encoding_rationale` — why these structural properties were chosen

The structural signature is the core of the system. Instead of embedding raw text, Prism embeds the *structure* — `symmetry_type=circular`, `mechanism=mutual-constitution`, `depth=unbounded`. This means two instances can have completely different surface descriptions and still find each other if their structural fingerprints are close.

### Links

Typed relationships between instances:

- `structurally_similar` — two instances share shape, with a similarity score
- `residual` — specific dimensions where the match breaks down (as important as the match itself)
- `is_instance_of` — connects an instance to its patterns
- `extends` — one instance generalizes or deepens another
- `contradicts` — instances that seem like they should be similar but aren't

### Link States

Links live on a spectrum of epistemic confidence:

| State | Meaning |
|-------|---------|
| `proposed` | Plausible (confidence > 0.5). Engine-generated, single-link evidence. |
| `corroborated` | Probable (confidence > 0.75 + triangulated). Two instances independently connect to both endpoints. |
| `confirmed` | Human-reviewed and validated. |
| `rejected` | Human-reviewed and dismissed — kept with rationale, never deleted. |
| `weak` | Decayed due to structural isolation. Silence is signal. |

Only humans can confirm or reject. The engine elevates proposed → corroborated and demotes proposed → weak.

---

## The Three Pressures

The graph is a living system. Running `prism pressures` applies three dynamics:

**Upward pressure (triangulation):** When a new instance independently connects to both endpoints of a speculative link, that link is elevated from `proposed` to `corroborated`. The atlas gets smarter passively as it grows.

**Downward pressure (structural isolation):** A `proposed` link that stays uncorroborated as the surrounding graph fills in fades to `weak`. Not deleted — structural isolation is information.

**Lateral pressure (pattern emergence):** When instances cluster together but don't fit any existing pattern, the engine proposes a new meta-pattern. Discovery of new patterns from the topology of the graph itself.

Run pressures periodically, especially after encoding a batch of new instances.

---

## Command Reference

All commands accept a `--db` flag to specify database path (default: `prism.db`):

```bash
prism --db ~/my-atlas.db <command>
```

### Setup

```bash
prism init                        # Initialize a new database
prism seed                        # Load starter library (7 patterns, 22 instances)
```

### Building the Atlas

```bash
# Add a new meta-pattern
prism add-pattern \
  --id phase-transition \
  --name "Phase Transition" \
  --description "Quantitative change accumulating until a qualitative threshold is crossed"

# Encode a new instance
prism encode \
  --id phys-water-ice \
  --domain physics \
  --description "Water freezing: continuous cooling produces a discontinuous structural change" \
  --signature symmetry_type=discontinuous \
  --signature dimensionality=thermal \
  --signature mechanism=threshold \
  --signature reversibility=reversible \
  --pattern symmetry-breaking \
  --pattern bifurcation \
  --rationale "The phase transition is the canonical case of gradual quantitative change producing sudden qualitative difference"
```

When you encode an instance, the engine immediately searches for structural neighbors and proposes speculative links. These show up inline.

### Exploring

```bash
# Browse from an instance — see all its links
prism wander gram-recursive-embedding

# Include weak/decaying links (where the weird stuff lives)
prism wander gram-recursive-embedding --show-weak

# Browse from a pattern — see all instances that manifest it
prism wander-pattern symmetry-breaking

# Drop a plain-language description, find what it structurally rhymes with
prism drop "the way a conversation changes when someone names the thing everyone was avoiding"
prism drop "a system that resists simplification the more fundamental it becomes" -k 10

# Drift through speculative connections — serendipity mode
prism drift
```

### Reviewing Links

```bash
# View a link
prism review <link-id>

# Confirm a speculative link
prism review <link-id> --confirm
prism review <link-id> --confirm --note "The parallel holds especially in the irreversibility dimension"

# Reject a speculative link
prism review <link-id> --reject --note "Surface similarity only — mechanisms are inverted"
```

### Graph Dynamics

```bash
# Apply three-pressure dynamics + detect emergent pattern proposals
prism pressures

# List all patterns
prism patterns
```

---

## Seed Domains

The starter library covers four domains chosen for structural richness:

**Grammar** — irregular verbs, recursive embedding, color term universals, garden-path sentences, pragmatic implicature. Syntax as a window into meta-structure.

**Observer-Reality** — quantum measurement, the Hawthorne effect, the hermeneutic circle, the anthropic principle, naming changing what it names. The shape of the relationship between knower and known.

**Cognition** — meta-attention, tip-of-tongue, insight restructuring, conceptual gravity, interference patterns in pre-articulate thought. The geometry of thought as a process.

**Prediction** — self-fulfilling prophecy, regression to mean, black swans, evolution as future-pull, prediction market collapse, Bayesian update, geodesic thinking. How futures constrain the present.

---

## Using Prism in a Claude Session

Prism was built with a specific division of labor in mind:

**Claude as curator.** When you're in a Claude Code session and encounter a structural pattern — in something you're reading, thinking about, or building — you can ask Claude to encode it. Claude will identify the structural signature, tag the relevant meta-patterns, write the rationale, and run the encoder. The speculative links that come back are worth looking at.

**You as explorer.** The `wander`, `drop`, and `drift` commands are for you. You know what's interesting. Confirm links that feel right, reject ones that don't, and pay attention to the rejected ones — the rationale is often more useful than the confirmation.

**Prism as engine.** Run `pressures` periodically. The corroboration dynamics mean the atlas gets smarter with scale — connections that weren't visible with 20 instances become visible with 200.

### Prompts worth trying in a session

Ask Claude to encode something you've been thinking about:
> "Encode this into Prism: the way a long-running project develops implicit assumptions that become invisible to everyone working on it."

Ask Claude to find connections:
> "Look at the instances tagged `observer-dependence` and tell me which ones have the most unexpected structural overlap."

Ask Claude to review the proposed links:
> "Run `prism drift` a few times and review whatever comes up. Confirm or reject with rationale."

Ask Claude to propose new domains:
> "What domains do you think would generate the most productive speculative links with what's already in the atlas?"

---

## Structural Signature Vocabulary

The signature vocabulary grows organically, but these properties have proven most useful for discrimination:

| Property | Example Values |
|----------|---------------|
| `symmetry_type` | `broken`, `self-similar`, `circular`, `bifurcating`, `collapsing`, `complementary`, `attracting`, `paradoxical`, `curved` |
| `dimensionality` | `temporal`, `hierarchical`, `probabilistic`, `conceptual`, `social`, `evolutionary`, `inferential`, `metacognitive` |
| `mechanism` | `self-reference`, `collapse`, `recursion`, `observation`, `threshold`, `interference`, `mutual-constitution`, `causal-loop` |
| `scope` | `fundamental`, `universal`, `local`, `individual`, `collective`, `contextual` |
| `reversibility` | `irreversible`, `reversible`, `partial`, `forced` |
| `direction` | `future-to-present`, `bidirectional`, `convergent`, `retroactive` |

The system tracks which dimensions most discriminate meaningful connections. Properties that never contribute to useful links eventually get flagged as noise.

---

## Design Values

- Surprise emerges from scale, not cleverness
- The residuals (what doesn't translate) are as valuable as the matches
- Epistemic posture: the tool asks questions, never asserts certainty
- Associative reach across all domains, not specialized to any
- Meta-structural: patterns underneath fields, not within them
- Every curatorial decision is logged and traceable

---

## Development

```bash
# Run tests
pytest

# Run a specific test file
pytest tests/test_resonance.py -v
```

**Stack:** Python 3.13 · SQLite + `sqlite-vec` · TF-IDF via scikit-learn · Click CLI · No external services — fully local.

---

## What's Not Here (Yet)

- Web UI
- Multi-user support
- Import / export
- LLM integration for auto-encoding
- Visualization beyond terminal
