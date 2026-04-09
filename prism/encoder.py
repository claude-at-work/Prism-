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
