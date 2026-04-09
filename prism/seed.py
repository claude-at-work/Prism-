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
