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
