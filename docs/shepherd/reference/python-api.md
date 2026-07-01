# Python API

> Page status: scaffold
> Source state: generated
> Applies to: Shepherd 0.1
> Owner: @docs-system-owner (TBD)
> Validation: scripts/gen_shepherd_api_inventory.py --check

*Reference. Exact, generated facts. The mental model lives in concepts, recipes in guides.*

!!! warning "Pre-rename surface — scaffolded entry"
    This is the curated entry page for the Python API reference (
    §"Generated API Reference"). The per-symbol facts are generated from the
    public facade into `reference/api/` and verified against
    `docs/_generated/shepherd/python-api/public-symbols.json`. Names render from
    the internal `shepherd` facade until the Shepherd rename; this page becomes
    release-ready when the facade is renamed and the symbols are documented to
    the public bar.

Shepherd's public surface is a small facade: you `import shepherd as shp` and
reach the task/workspace/delivery spine plus the effect and run vocabulary. The
generator targets **only** the accepted public facade — internal implementation
packages are never swept into this reference just because they are importable.

The exact, generated per-symbol pages live under `reference/api/` — one page per
public symbol, regenerated and drift-checked on every build (see
[`task`](api/task.md) for the shape). The machine-readable inventory is
`docs/_generated/shepherd/python-api/public-symbols.json`.
