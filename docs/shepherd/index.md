---
hide:
  - navigation
  - toc
---

<!--
Page-metadata block, kept in an HTML comment so the membership gate
(scripts/check_shepherd_docs.py) still reads the `> Key: value` lines while the
landing renders without a visible status banner.
> Page status: release-ready
> Source state: shipped-source
> Applies to: Shepherd 0.1
> Owner: @docs-system-owner (TBD)
> Validation: scripts/check_shepherd_docs.py
-->

<div class="shp-hero" markdown>

# Program meta-agents in Python

Write agents as simple typed functions, and meta-agents as functions that take your agents as input.

[Get started](tutorials/first-shepherd-app.md){ .md-button .md-button--primary }
[Quickstart](start/quickstart.md){ .md-button }
[Concepts](concepts/tasks.md){ .md-button }

</div>

```python title="hello.py"
--8<-- "quickstart/hello.py:hello"
```

## Find your path

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Build your first agent**

    ---

    A typed task, a workspace, and a small working reviewer. Offline and
    deterministic.

    [:octicons-arrow-right-24: First Shepherd app](tutorials/first-shepherd-app.md)

-   :material-bug-check:{ .lg .middle } **Debug and test a run**

    ---

    Read typed failures, keep runs deterministic, and test model-backed
    code without live calls.

    [:octicons-arrow-right-24: Debug your first run](guides/debug-your-first-run.md)

-   :material-lightbulb-on:{ .lg .middle } **Understand & evaluate**

    ---

    The mental model behind tasks, effects, and runs, plus a record of what
    these docs can claim today.

    [:octicons-arrow-right-24: Concepts: Tasks](concepts/tasks.md)

</div>

## Why Shepherd

- **Typed.** A task is a function with a signature and a docstring. The return
  type is the contract the model must satisfy.
- **Observable.** Every run records what was sent and returned, so you debug by
  reading a trace instead of guessing.
- **Composable.** Tasks are values. Pass them, supervise them, and build bigger
  programs out of small ones.

!!! info "Every page here is backed by tested code"
    A page goes public only after its examples run as tests against real
    Shepherd code. The [source-state inventory](reference/source-state.md)
    records where each claim comes from.
