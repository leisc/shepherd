"""Notebook presentation, kept apart from the Shepherd control surface.

Everything here just *draws*: it turns retained outputs, traces, and source into
HTML for a notebook cell. Nothing here is a Shepherd primitive -- the rule is
``launch.<verb>`` does something, ``viz.<verb>`` only shows it. The guides import this
as ``viz`` so a cell makes the split obvious at a glance.

These helpers are use-case-agnostic (a table, an artifact iframe, a stack, the
trace viewer, a source block). Use-case-specific cards live with their use case.
"""

from __future__ import annotations

import ast
import html as _html
import inspect
import textwrap
from collections.abc import Mapping, Sequence
from typing import Any, Literal

TraceDetail = Literal["summary", "events"]


def show(fragment: str) -> None:
    """Display an HTML fragment in a notebook with one small call."""
    from IPython.display import HTML, display

    display(HTML(fragment))


def workflow_overview(
    *,
    title: str,
    subtitle: str,
    steps: Sequence[tuple[str, str]],
) -> str:
    """Draw a compact use-case overview for a notebook introduction."""
    step_cards = []
    for index, (label, detail) in enumerate(steps, start=1):
        step_cards.append(
            '<div class="viz-flow-step">'
            f'<div class="viz-flow-index">{index}</div>'
            f'<div class="viz-flow-label">{_html.escape(label)}</div>'
            f'<div class="viz-flow-detail">{_html.escape(detail)}</div>'
            "</div>"
        )
    return f"""
    <style>
.viz-flow {{
        font: 14px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        border: 1px solid #d9dfec;
        border-radius: 8px;
        background: #fff;
        padding: 16px;
        margin: 8px 0 18px;
      }}
.viz-flow-title {{ font-size: 20px; font-weight: 800; color: #172033; margin-bottom: 4px; }}
.viz-flow-subtitle {{ color: #5b6474; margin-bottom: 14px; max-width: 860px; }}
.viz-flow-steps {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 10px;
      }}
.viz-flow-step {{
        min-width: 0;
        border: 1px solid #e2e7f1;
        background: #f8fafc;
        padding: 12px;
        display: grid;
        gap: 6px;
        min-height: 128px;
      }}
.viz-flow-index {{
        width: 28px;
        height: 28px;
        display: grid;
        place-items: center;
        border-radius: 50%;
        background: #172033;
        color: #fff;
        font-weight: 800;
      }}
.viz-flow-label {{ font-weight: 800; color: #172033; }}
.viz-flow-detail {{ color: #465266; line-height: 1.35; }}
    </style>
    <section class="viz-flow">
      <div class="viz-flow-title">{_html.escape(title)}</div>
      <div class="viz-flow-subtitle">{_html.escape(subtitle)}</div>
      <div class="viz-flow-steps">{"".join(step_cards)}</div>
    </section>
    """


def process_diagram(
    *,
    title: str,
    start: tuple[str, str],
    workspace: tuple[str, str],
    branches: Sequence[tuple[str, str]],
    review: tuple[str, str],
    output: tuple[str, str],
    branches_title: str = "Parallel task runs",
) -> str:
    """Draw a branching process diagram for a public recipe."""
    branch_nodes = "".join(
        '<div class="viz-process-branch">'
        f'<div class="viz-process-node-title">{_html.escape(label)}</div>'
        f'<div class="viz-process-node-text">{_html.escape(detail)}</div>'
        "</div>"
        for label, detail in branches
    )

    def node(kind: str, label: str, detail: str) -> str:
        return (
            f'<div class="viz-process-node viz-process-{kind}">'
            f'<div class="viz-process-node-title">{_html.escape(label)}</div>'
            f'<div class="viz-process-node-text">{_html.escape(detail)}</div>'
            "</div>"
        )

    return f"""
    <style>
.viz-process {{
        font: 14px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        border: 1px solid #cfd8e6;
        border-radius: 8px;
        background: #ffffff;
        padding: 18px;
        margin: 8px 0 18px;
        box-shadow: 0 10px 28px rgba(20, 32, 51,.08);
      }}
.viz-process-title {{
        font-size: 19px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 14px;
      }}
.viz-process-map {{
        display: grid;
        grid-template-columns: minmax(130px,.95fr) 26px minmax(140px, 1fr) 26px minmax(220px, 1.55fr) 26px minmax(150px, 1fr) 26px minmax(140px,.9fr);
        align-items: center;
        gap: 8px;
      }}
.viz-process-node,
.viz-process-forks {{
        min-width: 0;
        border: 1px solid #dfe6f2;
        border-radius: 8px;
        background: #fff;
        min-height: 126px;
        padding: 14px;
        box-sizing: border-box;
      }}
.viz-process-node {{
        display: grid;
        align-content: start;
        gap: 10px;
        border-top: 4px solid #4f7ccf;
      }}
.viz-process-workspace {{ border-top-color: #1d8f79; }}
.viz-process-review {{ border-top-color: #c17425; }}
.viz-process-output {{ border-top-color: #6b7280; }}
.viz-process-node-title {{
        font-weight: 800;
        color: #172033;
      }}
.viz-process-node-text {{
        color: #465266;
        line-height: 1.35;
      }}
.viz-process-arrow {{
        display: grid;
        place-items: center;
        color: #1d8f79;
        font-size: 24px;
        font-weight: 800;
      }}
.viz-process-forks {{
        background: #f7fafc;
        display: grid;
        gap: 10px;
      }}
.viz-process-fork-title {{
        color: #39445a;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0;
        text-transform: uppercase;
      }}
.viz-process-branch {{
        border: 1px solid #d2e9e3;
        border-left: 4px solid #1d8f79;
        border-radius: 8px;
        background: #fff;
        padding: 10px 12px;
        display: grid;
        gap: 5px;
      }}
      @media (max-width: 980px) {{
.viz-process-map {{
          grid-template-columns: 1fr;
        }}
.viz-process-arrow {{
          min-height: 22px;
          transform: rotate(90deg);
        }}
.viz-process-node,
.viz-process-forks {{
          min-height: auto;
        }}
      }}
    </style>
    <section class="viz-process">
      <div class="viz-process-title">{_html.escape(title)}</div>
      <div class="viz-process-map">
        {node("start", *start)}
        <div class="viz-process-arrow" aria-hidden="true">&rarr;</div>
        {node("workspace", *workspace)}
        <div class="viz-process-arrow" aria-hidden="true">&rarr;</div>
        <div class="viz-process-forks">
          <div class="viz-process-fork-title">{_html.escape(branches_title)}</div>
          {branch_nodes}
        </div>
        <div class="viz-process-arrow" aria-hidden="true">&rarr;</div>
        {node("review", *review)}
        <div class="viz-process-arrow" aria-hidden="true">&rarr;</div>
        {node("output", *output)}
      </div>
    </section>
    """


def pipeline_flow(
    *,
    title: str,
    stages: Sequence[tuple[str, str]],
) -> str:
    """Backward-compatible linear process diagram."""
    start = stages[0] if stages else ("Start", "")
    workspace = stages[1] if len(stages) > 1 else ("Workspace", "")
    review = stages[-2] if len(stages) > 2 else ("Review", "")
    output = stages[-1] if len(stages) > 1 else ("Output", "")
    branch_stages = stages[2:-2] or [("Task run", "Run the work in a retained branch.")]
    return process_diagram(
        title=title,
        start=start,
        workspace=workspace,
        branches=branch_stages,
        review=review,
        output=output,
    )


def prompt_box(prompt: str, *, title: str = "Prompt") -> str:
    """Show the user-facing input without exposing the backing object."""
    return f"""
    <div style="font:14px system-ui, -apple-system, sans-serif; border:1px solid #d9dfec;
                border-radius:8px; background:#fff; padding:14px 16px; margin:8px 0 18px;">
      <div style="font-weight:800; color:#172033; margin-bottom:8px;">{_html.escape(title)}</div>
      <pre style="white-space:pre-wrap; margin:0; color:#26364f; font:14px ui-monospace, SFMono-Regular, Menlo, monospace;">{_html.escape(prompt)}</pre>
    </div>
    """


def table(rows: Sequence[Mapping[str, object]]) -> str:
    """A compact HTML table for notebooks, no pandas required."""
    if not rows:
        return "<p>No rows.</p>"

    columns = list(rows[0])
    header = "".join(f"<th>{_html.escape(str(column))}</th>" for column in columns)
    body = []
    for row in rows:
        cells = "".join(f"<td>{_html.escape(_cell_text(row.get(column)))}</td>" for column in columns)
        body.append(f"<tr>{cells}</tr>")
    return f"""
    <style>
.viz-table {{
        border-collapse: collapse;
        width: 100%;
        font: 13px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
.viz-table th,
.viz-table td {{
        border: 1px solid #d9dfec;
        padding: 8px 10px;
        text-align: left;
        vertical-align: top;
      }}
.viz-table th {{ background: #f5f7fb; font-weight: 700; }}
.viz-table td {{ background: #fff; }}
    </style>
    <table class="viz-table">
      <thead><tr>{header}</tr></thead>
      <tbody>{"".join(body)}</tbody>
    </table>
    """


def artifact(
    html: str,
    *,
    label: str | None = None,
    height: int = 260,
    accent: str = "#d9dfec",
    max_width: int = 680,
) -> str:
    """Render one HTML artifact live in a sandboxed iframe, optionally captioned.

    The single-result primitive: one page, rendered as a viewer would see it,
    never a screenshot. The tile artifact is a 4:3 composition, so the iframe is
    responsive by aspect ratio and capped to a notebook-scale width; ``height``
    is a minimum height, not a clipping height. ``stack`` and side-by-side views
    are built from this.
    """
    doc = _html.escape(html, quote=True)
    border = _resolve_accent(accent)
    caption = f'<strong style="font:600 13px system-ui;">{_html.escape(label)}</strong>' if label else ""
    return (
        f'<section style="display:grid; gap:8px; width:min(100%, {max_width}px);">{caption}'
        f'<iframe sandbox="allow-same-origin" srcdoc="{doc}" style="width:100%; aspect-ratio:4 / 3; '
        f"height:auto; min-height:{height}px; border:1px solid {border}; "
        f'border-radius:8px; background:white; display:block;"></iframe></section>'
    )


def stack(items: Sequence[tuple[str, str]] | Mapping[str, str], *, height: int = 320) -> str:
    """Render several artifacts full-width, one below the next.

    ``items`` is ``[(label, html),...]`` (or a label->html mapping). Each page
    gets the full width, so the differences between attempts are legible -- the
    notebook can also call ``artifact`` in a loop to reveal them as forks return.
    """
    pairs = list(items.items()) if isinstance(items, Mapping) else list(items)
    sections = "".join(artifact(html, label=label, height=height) for label, html in pairs)
    return f'<div style="display:grid; gap:20px;">{sections}</div>'


def side_by_side(sections: Sequence[str]) -> str:
    """Lay out a few rendered sections in a responsive row (e.g. before / after)."""
    return (
        '<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); '
        f'gap:16px;">{"".join(sections)}</div>'
    )


def run_artifacts(
    runs: Mapping[str, Any],
    *,
    selected: str | None = None,
    failed: Sequence[str] = (),
    height: int = 260,
) -> str:
    """Render retained workspace artifacts from completed runs as a board."""
    failed_set = set(failed)
    sections = []
    for name, run in runs.items():
        accent = "green" if name == selected else ("red" if name in failed_set else "#d9dfec")
        label = name
        if name == selected:
            label = f"{name} - selected"
        elif name in failed_set:
            label = f"{name} - discarded"
        sections.append(artifact(_run_output_text(run), label=label, height=height, accent=accent))
    return side_by_side(sections)


def run_artifact(run: Any, *, label: str | None = None, height: int = 260, accent: str = "#d9dfec") -> str:
    """Render the retained artifact from one completed run."""
    return artifact(_run_output_text(run), label=label or getattr(run, "name", None), height=height, accent=accent)


def variant_outputs(view: Any, *, height: int = 260) -> str:
    """Render the branch outputs from an explicit Variant Studio view."""
    sections = [artifact(candidate["html"], label=str(candidate["id"]), height=height) for candidate in view.attempts]
    return side_by_side(sections)


def variant_output(view: Any, variant: str, *, height: int = 260) -> str:
    """Render one branch output from an explicit Variant Studio view."""
    for candidate in view.attempts:
        if str(candidate["id"]) == variant:
            return artifact(candidate["html"], label=variant, height=height)
    raise ValueError(f"Unknown variant: {variant}")


def _run_output_text(run: Any, path: str = "index.html") -> str:
    output_method = getattr(run, "output", None)
    if callable(output_method):
        return output_method().read_text(path)
    raise TypeError("expected a public WorkspaceRun with output()")


def _compact_payload(value: Mapping[str, object]) -> str:
    compact = {key: item for key, item in value.items() if key not in {"id", "kind", "flow_id", "run_ref", "metadata"}}
    if not compact:
        return ""
    return ", ".join(f"{key}={item!r}" for key, item in compact.items())


_ACCENTS = {
    "green": "#21a67a",
    "pass": "#21a67a",
    "ok": "#21a67a",
    "red": "#c4515f",
    "fail": "#c4515f",
    "miss": "#c4515f",
    "warn": "#d39423",
}


def _resolve_accent(value: str) -> str:
    """Map a status word (pass/fail/green/red) to a color; pass a hex value through."""
    return _ACCENTS.get(str(value).lower(), str(value))


def card(
    title: str,
    *,
    subtitle: str | None = None,
    chips: Sequence[tuple[str, str]] = (),
    fields: Sequence[tuple[str, str]] = (),
    accent: str = "#d9dfec",
) -> str:
    """One result, summarized.

    The card has a title, optional subtitle, status chips, and key/value fields.

    The generic "summarize one thing" block -- a model's verdict with a pass/fail
    chip per item, or an inspector's diagnosis as labelled fields. ``chips`` is
    ``[(label, status)]`` (``status`` pass/fail colors the chip); ``fields`` is
    ``[(key, value)]`` rendered as rows; ``accent`` is a status word or a hex color.
    """
    bar = _resolve_accent(accent)
    sub = f'<div style="color:#5b6474; margin:2px 0 8px;">{_html.escape(subtitle)}</div>' if subtitle else ""
    chip_spans = ""
    for label, status in chips:
        color = _resolve_accent(status)
        chip_spans += (
            '<span style="display:inline-block; margin:2px 6px 2px 0; padding:2px 9px; '
            f'border-radius:999px; background:{color}1a; color:{color}; font-weight:600;">'
            f"{_html.escape(str(label))}: {_html.escape(str(status))}</span>"
        )
    chips_block = f"<div>{chip_spans}</div>" if chips else ""
    field_rows = "".join(
        '<div style="display:grid; grid-template-columns:160px 1fr; gap:10px; padding:6px 0; '
        'border-bottom:1px solid #eef1f7;">'
        f'<span style="font-weight:700; color:#39445a;">{_html.escape(str(key))}</span>'
        f'<span style="color:#172033;">{_html.escape(str(value))}</span></div>'
        for key, value in fields
    )
    return (
        '<div style="font:13px system-ui, -apple-system, sans-serif; border:1px solid #d9dfec; '
        f"border-left:4px solid {bar}; border-radius:8px; padding:12px 14px; margin:8px 0; "
        'background:#fff; max-width:760px;">'
        f'<div style="font-weight:800;">{_html.escape(title)}</div>'
        f"{sub}{chips_block}{field_rows}</div>"
    )


def compare(
    items: Mapping[str, str] | Sequence[tuple[str, str]],
    *,
    height: int = 300,
    accents: Mapping[str, str] | None = None,
) -> str:
    """Render named artifacts side by side for a before/after comparison.

    ``items`` maps a label to its HTML; ``accents`` optionally maps a label to a
    border color (a status word or hex), e.g. ``"red"`` for the broken draft and
    ``"green"`` for the fix. Built from ``artifact`` + ``side_by_side``.
    """
    pairs = list(items.items()) if isinstance(items, Mapping) else list(items)
    accents = accents or {}
    sections = [
        artifact(html, label=label, height=height, accent=_resolve_accent(accents.get(label, "#d9dfec")))
        for label, html in pairs
    ]
    return side_by_side(sections)


def review_summary(candidates: Sequence[Mapping[str, object]], *, selected: str) -> str:
    """Show a compact pass/fail review and the selected candidate."""
    rows = []
    for candidate in candidates:
        issues = candidate.get("issues") or []
        rows.append(
            {
                "candidate": candidate.get("id", ""),
                "verdict": candidate.get("verdict", ""),
                "issues": ", ".join(str(issue) for issue in issues) if isinstance(issues, list) else str(issues),
            }
        )
    banner = (
        '<div style="font:14px system-ui, -apple-system, sans-serif; border:1px solid #b9e3d4; '
        "background:#f0fbf7; color:#11684d; padding:10px 12px; border-radius:8px; "
        f'margin:8px 0 12px;"><strong>Selected:</strong> {_html.escape(selected)}</div>'
    )
    return banner + table(rows)


def variant_selection(view: Any) -> str:
    """Render the reviewer result from an explicit Variant Studio view."""
    return review_summary(view.candidates, selected=view.selected)


def trace(
    flow: Any,
    runs: Mapping[str, Any],
    *,
    notes: Mapping[str, str] | None = None,
    height: str = "600px",
    detail: TraceDetail = "summary",
) -> str:
    """Render a public flow trace as one self-contained inline viewer."""
    if detail not in {"summary", "events"}:
        raise ValueError(f"unknown trace detail mode: {detail!r}")
    trace_method = getattr(flow, "trace", None)
    if not callable(trace_method):
        raise TypeError("expected a public Flow with trace()")
    run_notes = {run.run_ref: name for name, run in runs.items() if hasattr(run, "run_ref")}
    if notes:
        run_notes.update(notes)
    return flow_trace(trace_method(), notes=run_notes, height=height)


def flow_trace(
    trace_payload: Mapping[str, object], *, notes: Mapping[str, str] | None = None, height: str = "600px"
) -> str:
    """Render a public flow trace projection as a compact HTML table."""
    raw_events = trace_payload.get("events", [])
    raw_edges = trace_payload.get("edges", [])
    events = raw_events if isinstance(raw_events, Sequence) and not isinstance(raw_events, str) else ()
    edges = raw_edges if isinstance(raw_edges, Sequence) and not isinstance(raw_edges, str) else ()
    notes = notes or {}
    event_rows = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        run_ref = str(event.get("run_ref", ""))
        label = notes.get(run_ref, notes.get(str(event.get("name", "")), ""))
        event_rows.append(
            "<tr>"
            f"<td>{_html.escape(str(event.get('kind', '')))}</td>"
            f"<td>{_html.escape(run_ref or str(event.get('flow_id', '')))}</td>"
            f"<td>{_html.escape(label)}</td>"
            f"<td>{_html.escape(_compact_payload(event))}</td>"
            "</tr>"
        )
    edge_rows = []
    for edge in edges:
        if not isinstance(edge, Mapping):
            continue
        edge_rows.append(
            "<tr>"
            f"<td>{_html.escape(str(edge.get('kind', '')))}</td>"
            f"<td>{_html.escape(str(edge.get('source', '')))}</td>"
            f"<td>{_html.escape(str(edge.get('target', '')))}</td>"
            "</tr>"
        )
    html = f"""
    <style>
.trace {{
        font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        color: #172033;
      }}
.trace h3 {{ font: 700 14px system-ui, -apple-system, sans-serif; margin: 10px 0 6px; }}
.trace table {{ border-collapse: collapse; width: 100%; margin-bottom: 14px; }}
.trace th,
.trace td {{ border: 1px solid #d9dfec; padding: 6px 8px; text-align: left; vertical-align: top; }}
.trace th {{ background: #f5f7fb; }}
    </style>
    <section class="trace">
      <h3>Events</h3>
      <table>
        <thead><tr><th>kind</th><th>run/flow</th><th>label</th><th>payload</th></tr></thead>
        <tbody>{"".join(event_rows)}</tbody>
      </table>
      <h3>Edges</h3>
      <table>
        <thead><tr><th>kind</th><th>source</th><th>target</th></tr></thead>
        <tbody>{"".join(edge_rows)}</tbody>
      </table>
    </section>
    """
    return _trace_iframe(html, height=height)


def source(obj: object) -> Any:
    """Show an object's source as a syntax-highlighted block (the reveal move).

    Used to surface the bodyless ``@task``, the child-run body, and the critic so
    the reader sees the real, small primitives directly.
    """
    from IPython.display import Code

    return Code(_source_text(obj), language="python")


def task_contract(obj: object) -> Any:
    """Show a Shepherd task's public contract without fixture/backing code."""
    from IPython.display import Code

    return Code(_contract_text(obj), language="python")


# --- internals --------------------------------------------------------------


def _contract_text(obj: object) -> str:
    src = textwrap.dedent(_source_text(obj))
    try:
        module = ast.parse(src)
        fn = next(node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    except (SyntaxError, StopIteration):
        return src

    lines = src.splitlines()
    doc = ast.get_docstring(fn, clean=True) or inspect.getdoc(obj) or ""
    doc_block = (
        ' """' + doc.replace('"""', '\\"\\"\\"') + '"""'
        if "\n" not in doc
        else (' """' + doc.splitlines()[0] + "\n" + textwrap.indent("\n".join(doc.splitlines()[1:]), " ") + '\n """')
    )
    contract_lines = lines[: max(0, fn.body[0].lineno - 1)] if fn.body else lines
    return "\n".join([*contract_lines, doc_block, "..."])


def _trace_iframe(document_html: str, *, height: str = "600px") -> str:
    """Wrap a full, self-contained HTML document for inline notebook display."""
    srcdoc = _html.escape(document_html, quote=True)
    return (
        '<div class="shepherd-trace-viewer-embed" '
        'style="border:1px solid #d9dfec;border-radius:8px;overflow:hidden;">'
        f'<iframe srcdoc="{srcdoc}" '
        f'style="width:100%;height:{height};border:0;display:block;" '
        'loading="lazy"></iframe>'
        "</div>"
    )


def _source_text(obj: object) -> str:
    metadata = getattr(obj, "metadata", None)
    src = getattr(metadata, "source", None)
    if isinstance(src, str) and src.strip():
        return src.strip()
    return inspect.getsource(obj).strip()


def _cell_text(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)
