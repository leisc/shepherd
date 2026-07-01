#!/usr/bin/env python3
"""Generate the per-symbol API reference pages (DESIGN §5.3 page contract).

Pulls from the REAL repo facade (read-only). Pages are committed; --check
regenerates in memory and fails on any difference. `_map.yml` is INPUT (the
curated See-also map) — never rewritten here.
"""

from __future__ import annotations

import sys

from _facade import API_DIR, FACADE_IMPORT, MAP_FILE, facade_map, page_filename, symbol_info

MODE_LINE = (
    "*Reference. Exact, generated facts. The mental model lives in "
    "concepts, recipes in guides.*"
)
BANNER = (
    '!!! warning "Pre-rename surface"\n'
    "    Generated from the internal `shepherd` facade; names and paths change "
    "at the Shepherd rename.\n"
)


def load_map() -> dict:
    if not MAP_FILE.exists():
        return {}
    import yaml

    return yaml.safe_load(MAP_FILE.read_text(encoding="utf-8")) or {}


def render(info: dict, see_also: dict | None) -> str:
    name, target, kind = info["name"], info.get("target", info["source"]), info["kind"]
    lines = [
        f"# `{FACADE_IMPORT}.{name}`",
        "",
        "> Page status: scaffold",
        "> Source state: generated",
        "> Applies to: Shepherd 0.1",
        "> Owner: @docs-system-owner (TBD)",
        "> Validation: scripts/gen_shepherd_api_inventory.py --check",
        "",
        MODE_LINE,
        "",
        BANNER,
        f'<span class="api-kind">{kind}</span>',
        "",
        f"::: {target}",
        "    options:",
        "      show_root_heading: true",
        "      heading_level: 2",
        "      show_root_full_path: false",
        "",
    ]
    if see_also:
        lines += ["## See also", ""]
        if see_also.get("concept"):
            lines.append(f"- Mental model: [{see_also['concept']}](../../{see_also['concept']})")
        if see_also.get("guide"):
            lines.append(f"- Recipe: [{see_also['guide']}](../../{see_also['guide']})")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    check = "--check" in sys.argv
    exports, _ = facade_map()
    infos = symbol_info()
    smap = load_map()
    stale = []
    expected: set[str] = set()
    API_DIR.mkdir(parents=True, exist_ok=True)
    for info in infos:
        fn = API_DIR / page_filename(info["name"], exports)
        expected.add(fn.name)
        content = render(info, smap.get(info["name"]))
        if check:
            if not fn.exists() or fn.read_text(encoding="utf-8") != content:
                stale.append(fn.name)
        else:
            fn.write_text(content, encoding="utf-8", newline="\n")
    # The generated per-symbol pages are the only *.md in API_DIR (_map.yml is the
    # sole hand-maintained input). Any *.md outside the current export set is an
    # orphan from a wider facade: flag it under --check, delete it on regen, so a
    # shrunk facade never leaves a page the strict build would try (and fail) to
    # render.
    orphans = sorted(p.name for p in API_DIR.glob("*.md") if p.name not in expected)
    if check:
        stale = sorted(set(stale) | set(orphans))
        if stale:
            print(f"DRIFT: {len(stale)} generated page(s) stale: {', '.join(stale)}")
            print("fix: ./check.sh regen   (see docs/_runbook.md)")
            return 1
        print(f"ok: {len(infos)} generated pages match the facade")
        return 0
    for name in orphans:
        (API_DIR / name).unlink()
    tail = f" (pruned {len(orphans)} stale)" if orphans else ""
    print(f"wrote {len(infos)} pages -> {API_DIR}{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
