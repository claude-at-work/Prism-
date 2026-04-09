import uuid

from prism.db import PrismDB, _serialize_float_vec
from prism.embeddings import Embedder
from prism.models import Instance, Link, LinkState, LinkType


class ResonanceEngine:
    PLAUSIBLE_THRESHOLD = 1.2  # L2 distance on unit vectors: 0=identical, sqrt(2)≈1.41=orthogonal, 2=opposite
    PROBABLE_THRESHOLD = 0.8  # tighter threshold for probable
    DECAY_ROUNDS = 3  # rounds without corroboration before weakening
    MIN_CLUSTER_SIZE = 3  # minimum instances to propose emergent pattern

    def __init__(self, db: PrismDB, embedder: Embedder):
        self.db = db
        self.embedder = embedder
        self._rounds_since_corroboration: dict[str, int] = {}

    def _refit_and_update_all(self, instances: list[Instance]):
        """Refit embedder on all instance signatures and update stored embeddings."""
        sigs = [inst.structural_signature for inst in instances]
        self.embedder.fit(sigs)
        for inst in instances:
            new_emb = self.embedder.embed(inst.structural_signature)
            inst.embedding = new_emb
            try:
                self.db.conn.execute("DELETE FROM instance_embeddings WHERE id = ?", (inst.id,))
            except Exception:
                pass
            self.db._ensure_vec_dim(len(new_emb))
            self.db.conn.execute(
                "INSERT INTO instance_embeddings (id, embedding) VALUES (?, ?)",
                (inst.id, _serialize_float_vec(new_emb)),
            )
        self.db.conn.commit()

    def add_instance(self, instance: Instance) -> list[Link]:
        """Insert an instance, refit embedder, find nearest neighbors, create speculative links."""
        existing = self.db.list_instances()

        # Refit on all signatures including the new one
        all_instances = existing + [instance]
        all_sigs = [inst.structural_signature for inst in all_instances]
        self.embedder.fit(all_sigs)

        # Recompute embeddings for existing instances with new vocabulary
        for inst in existing:
            new_emb = self.embedder.embed(inst.structural_signature)
            inst.embedding = new_emb
            try:
                self.db.conn.execute("DELETE FROM instance_embeddings WHERE id = ?", (inst.id,))
            except Exception:
                pass
            self.db._ensure_vec_dim(len(new_emb))
            self.db.conn.execute(
                "INSERT INTO instance_embeddings (id, embedding) VALUES (?, ?)",
                (inst.id, _serialize_float_vec(new_emb)),
            )
        self.db.conn.commit()

        # Embed new instance
        instance.embedding = self.embedder.embed(instance.structural_signature)
        self.db.insert_instance(instance)

        if not existing:
            return []

        neighbors = self.db.find_nearest(
            query_embedding=instance.embedding,
            k=10,
            exclude_id=instance.id,
        )

        new_links = []
        for neighbor_id, distance in neighbors:
            if distance < self.PLAUSIBLE_THRESHOLD:
                neighbor = self.db.get_instance(neighbor_id)
                residual = Embedder.compute_residual(
                    instance.structural_signature,
                    neighbor.structural_signature,
                )
                confidence = max(0.0, 1.0 - (distance / self.PLAUSIBLE_THRESHOLD))  # normalize to 0-1 range
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
        """Apply all three pressures: upward (triangulation), downward (decay), lateral (clusters)."""
        self._apply_upward_pressure()
        self._apply_downward_pressure()

    def _apply_upward_pressure(self):
        """Triangulation: if A-B link has a shared neighbor C (A-C and B-C links), elevate A-B."""
        proposed_links = self.db.get_links_by_state(LinkState.PROPOSED)
        for link in proposed_links:
            if link.confidence <= 0.5:
                continue

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

            shared = source_neighbors & target_neighbors
            if shared:
                new_confidence = min(link.confidence + 0.1 * len(shared), 0.95)
                self.db.update_link(link.id, state=LinkState.CORROBORATED, confidence=new_confidence)
                self._rounds_since_corroboration.pop(link.id, None)

    def _apply_downward_pressure(self):
        """Decay: links that have not been corroborated over multiple rounds weaken."""
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
        """Lateral pressure: find clusters of structurally similar instances without a unifying pattern."""
        all_instances = self.db.list_instances()
        if len(all_instances) < self.MIN_CLUSTER_SIZE:
            return []

        clusters = self._find_connected_clusters(all_instances)
        proposals = []
        for cluster in clusters:
            if len(cluster) >= self.MIN_CLUSTER_SIZE:
                common = self._find_common_properties(cluster)
                if common:
                    proposals.append({
                        "instance_ids": [inst.id for inst in cluster],
                        "common_properties": common,
                        "size": len(cluster),
                    })
        return proposals

    def _find_connected_clusters(self, instances: list[Instance]) -> list[list[Instance]]:
        """BFS/DFS over structural similarity links to find connected components."""
        inst_map = {inst.id: inst for inst in instances}
        adjacency: dict[str, set[str]] = {inst.id: set() for inst in instances}

        for inst in instances:
            links = self.db.get_links_for_instance(inst.id)
            for link in links:
                if link.link_type == LinkType.STRUCTURALLY_SIMILAR and link.state in (
                    LinkState.PROPOSED, LinkState.CORROBORATED, LinkState.CONFIRMED
                ):
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
        """Find structural signature properties shared across all instances in the cluster."""
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
