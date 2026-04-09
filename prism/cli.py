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
