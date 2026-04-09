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
