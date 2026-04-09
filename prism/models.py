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
