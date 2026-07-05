"""Build Shepherd-facing notebooks over the current Shepherd workspace-control API."""

# ruff: noqa: INP001

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECK_MODE = False

CellSpec = tuple[str, str] | tuple[str, str, dict[str, object]]
HIDE_INPUT = {"jupyter": {"source_hidden": True}, "tags": ["hide-input"]}


def nb(cells: list[CellSpec]) -> dict[str, object]:
    rendered = []
    for index, cell_spec in enumerate(cells, start=1):
        kind, source = cell_spec[:2]
        metadata = dict(cell_spec[2]) if len(cell_spec) > 2 else {}
        cell: dict[str, object] = {
            "cell_type": kind,
            "id": f"cell-{index:02d}",
            "metadata": metadata,
            "source": source.strip("\n").splitlines(keepends=True),
        }
        if kind == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        rendered.append(cell)
    return {
        "cells": rendered,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(name: str, cells: list[CellSpec]) -> None:
    path = HERE / name
    text = json.dumps(nb(cells), indent=1) + "\n"
    if CHECK_MODE:
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"{path.relative_to(HERE)} is stale; run python _build_notebooks.py")
        print(f"checked {path.relative_to(HERE)} ({len(cells)} cells)")
        return
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(HERE)} ({len(cells)} cells)")


SETUP = """
import pathlib
import sys


def _find_visual_artifact_example_root():
    cwd = pathlib.Path.cwd().resolve()
    candidates = []
    for base in (cwd, *cwd.parents):
        candidates.append(base)
        candidates.append(base / "examples" / "notebooks" / "visual_artifact")
    for candidate in candidates:
        if (candidate / "shepherd_usecases" / "visual_artifact" / "launch.py").exists():
            return candidate
    raise RuntimeError(
        "Cannot find examples/notebooks/visual_artifact. "
        "Launch JupyterLab from the repository root with `make notebooks`."
    )


example_root = _find_visual_artifact_example_root()
if str(example_root) not in sys.path:
    sys.path.insert(0, str(example_root))
try:
    from shepherd_usecases.visual_artifact import launch
    from shepherd_usecases.visual_artifact import viz
except Exception as exc:
    raise RuntimeError(
        "Could not import the visual-artifact notebook helpers. "
        "Launch JupyterLab from the repository root with `make notebooks`."
    ) from exc

launch.bootstrap(example_root=example_root)
"""


def setup_cells() -> list[CellSpec]:
    return [
        ("markdown", "## Setup\n\nLoad the launch helpers."),
        ("code", SETUP, HIDE_INPUT),
    ]


def workspace_cell(usecase: str) -> CellSpec:
    return (
        "code",
        f'workspace = launch.open_workspace("{usecase}", prompt=prompt, metadata={{"usecase": "{usecase}"}})\n',
    )


def uc1() -> None:
    cells: list[CellSpec] = [
        (
            "markdown",
            """
# Find the best visual attempt

In this guide, you'll run a few variations of one task and keep the best result in a few lines.
The sample task is an infographic prompt. The same pattern works for bug fixes, UI directions,
prompts, queries, migrations, and other work where stronger alternatives are worth comparing.
""",
        ),
        *setup_cells(),
        (
            "markdown",
            """
## What this run does

Variant Studio is a best-of-N run: it turns one prompt into several isolated attempts, each in its
own run, then hands the finished attempts to a reviewer that keeps the strongest.
""",
        ),
        (
            "markdown",
            """
## The input

For this example, the run starts with one prompt and two short variant instructions.
""",
        ),
        (
            "code",
            'prompt = launch.default_prompt()\nvariants = launch.variant_prompts()\n{"prompt": prompt, "variants": variants}\n',
        ),
        (
            "markdown",
            """
## Run Shepherd

First, open a workspace for the prompt.
""",
        ),
        workspace_cell("visual-variant-studio"),
        (
            "markdown",
            """
Now run one attempt per variant from that workspace. Each run is isolated, so the two attempts
write their artifacts without seeing each other.
""",
        ),
        (
            "code",
            """
attempts = {}
for variant, instruction in variants.items():
    attempts[variant] = launch.run_static(
        workspace,
        name=variant,
        output_path=launch.ARTIFACT_PATH,
        output_text=launch.variant_html(prompt, variant),
        metadata={"variant": variant, "instruction": instruction},
    )

viz.show(viz.run_artifacts(attempts))
""",
        ),
        (
            "markdown",
            """
With both attempts finished, run the reviewer task. It receives the completed attempt runs as
artifact references — `candidate_refs` — so it can read each one's artifact and pick the best.
""",
        ),
        (
            "code",
            """
candidate_refs = {
    f"candidate_{name}": launch.artifact_ref(run, label=name)
    for name, run in attempts.items()
}
reviewer = launch.run_with_artifact_refs(
    workspace,
    name="review",
    refs=candidate_refs,
    output_path=launch.VERDICT_PATH,
    output_content=launch.review_content(prompt, attempts),
    after=list(attempts.values()),
)
selection = launch.selection_from_review(reviewer)
viz.show(viz.review_summary(selection.candidates, selected=selection.selected))
""",
        ),
        (
            "code",
            """
for name, run in attempts.items():
    if name == selection.selected:
        run.output().select()
    else:
        run.output().discard()
reviewer.output().release()
""",
        ),
        (
            "markdown",
            """
## Trace

How does Shepherd keep the attempts isolated and let one task read another's work? The trace below
captures every action an agent took while executing the tasks above, plus the
relationships among those actions. Click the nodes to learn more about the events.
""",
        ),
        (
            "code",
            """
notes = {run.run_ref: name for name, run in attempts.items()}
notes[reviewer.run_ref] = f"review: selected {selection.selected}"
viz.show(viz.trace(workspace.flow, {**attempts, "review": reviewer}, notes=notes, height="620px", detail="events"))
""",
        ),
        (
            "markdown",
            """
## Optional Live Claude Run

The default path above is deterministic. Set the flag below only on a machine with the local Claude CLI,
credentials, and native jail support.
""",
        ),
        (
            "code",
            """
RUN_LIVE_CLAUDE = False
CLAUDE_MODEL = None

if RUN_LIVE_CLAUDE:
    launch.require_claude()
    live_workspace = launch.open_workspace("visual-variant-studio-live", prompt=prompt, metadata={"usecase": "uc1-live"})
    try:
        live_attempts = {}
        for variant, instruction in variants.items():
            live_attempts[variant] = launch.run_claude_artifact(
                live_workspace,
                name=f"live-{variant}",
                prompt=prompt,
                variant=variant,
                instruction=instruction,
                model=CLAUDE_MODEL,
                metadata={"variant": variant, "mode": "live-claude"},
            )
        live_refs = {
            f"candidate_{name}": launch.artifact_ref(run, label=name)
            for name, run in live_attempts.items()
        }
        live_reviewer = launch.run_claude_review(
            live_workspace,
            name="live-review",
            prompt=prompt,
            refs=live_refs,
            after=list(live_attempts.values()),
            model=CLAUDE_MODEL,
            metadata={"mode": "live-claude"},
        )
        live_selection = launch.selection_from_review(live_reviewer)
        viz.show(viz.review_summary(live_selection.candidates, selected=live_selection.selected))
        for name, run in live_attempts.items():
            if name == live_selection.selected:
                run.output().select()
            else:
                run.output().discard()
        live_reviewer.output().release()
        viz.show(viz.trace(live_workspace.flow, {**live_attempts, "review": live_reviewer}, height="620px", detail="events"))
    finally:
        live_workspace.close()
else:
    launch.claude_preflight()
""",
        ),
        ("code", "workspace.close()\n"),
        (
            "markdown",
            "## Want to understand how it works?\n\nIf you want to understand how this works in greater detail, open the\n[Variant Studio internals notebook](visual_variant_studio_internals.ipynb).",
        ),
        (
            "markdown",
            "## Next steps\n\n- [Find the cheapest model that still does the job](model_right_sizing_lab.ipynb)\n- [Recover a pipeline when one step drifts](visual_pipeline_recovery.ipynb)",
        ),
    ]
    write("visual_variant_studio.ipynb", cells)


def uc1_internals() -> None:
    cells: list[CellSpec] = [
        (
            "markdown",
            """
# Variant Studio Internals

The short [Variant Studio guide](visual_variant_studio.ipynb) ran one prompt through two attempts
and a reviewer, then kept the best tile. This notebook rebuilds that same run in minimal form — one
attempt, one reviewer — and opens it up: the API objects Shepherd uses behind the recipe, and the
flow trace it records along the way.

Primitives exposed here: a flow, retained runs, durable argument refs, artifact refs, a regenerated
trace projection, and a task contract.
""",
        ),
        *setup_cells(),
        (
            "markdown",
            """
## Build a run to inspect

`launch.open_workspace` opens the controlling scope — everything the run produces lives inside it.
`launch.run_static` executes a `contour-map` attempt in that workspace and records its output.
`launch.artifact_ref` mints a durable content-addressed handle to the artifact the attempt wrote.
`launch.run_with_artifact_ref` runs the reviewer in the same workspace, receiving that handle as
the named argument `candidate` — which is what draws the dependency arrow from the attempt into the
review in the trace viewer.
""",
        ),
        ("code", "prompt = launch.default_prompt()\n"),
        workspace_cell("visual-variant-studio-internals"),
        (
            "code",
            """
attempt = launch.run_static(
    workspace,
    name="contour-map",
    output_path=launch.ARTIFACT_PATH,
    output_text=launch.variant_html(prompt, "contour-map"),
    metadata={"variant": "contour-map"},
)
artifact_ref = launch.artifact_ref(attempt, label="contour-map")
reviewer = launch.run_with_artifact_ref(
    workspace,
    name="review",
    ref_name="candidate",
    artifact_ref=artifact_ref,
    output_path=launch.VERDICT_PATH,
    output_content=launch.review_content(prompt, {"contour-map": attempt}),
    after=[attempt],
)
""",
        ),
        (
            "markdown",
            """
That is the whole run. The rest of the notebook opens up what is behind it: the run record and
argument refs, the regenerated flow trace, and the task contract.
""",
        ),
        (
            "markdown",
            """
## Run record and argument refs

Every run Shepherd executes produces a **run record**: a content-addressed snapshot of the arguments
it received. `launch.run_record` returns that record for any run in the workspace.

- `args_ref` identifies the argument bundle by content address in the flow.
- `args_digest` is the hash — two runs given identical arguments share one record.
- `input_refs` are the artifact refs passed in as named arguments: here, the one `artifact_ref`
  the reviewer received as `candidate`.

`launch.run_args` unpacks the raw argument map, exposing those `input_refs` as dereferenceable
handles. Together `record` and `args` let any downstream run re-cite exactly what the reviewer saw.
""",
        ),
        (
            "code",
            """
record = launch.run_record(workspace, reviewer)
args = launch.run_args(workspace, reviewer)
{
    "run_ref": reviewer.run_ref,
    "args_ref": record.args_ref,
    "args_digest": record.args_digest,
    "input_ref_count": len(args["input_refs"]),
    "artifact_ref": artifact_ref.to_json(),
}
""",
        ),
        (
            "markdown",
            """
## Flow trace

`workspace.flow.trace()` regenerates the full trace from the live flow object — not from a stored
snapshot — so it always reflects the current workspace state. `viz.flow_trace` renders it as an
interactive viewer: each run is a lane, the effects it performed line up inside it in order, and
an arrow between lanes means one run drew on another.

Here the attempt lane and the reviewer lane are both visible, with a dependency arrow from the
attempt into the review — because the reviewer received `artifact_ref` and read it. Click any node
to open the effect it represents.
""",
        ),
        ("code", 'viz.show(viz.flow_trace(workspace.flow.trace(), height="620px"))\n'),
        (
            "markdown",
            """
## Task contract

`tasks.static_artifact_task` is the underlying task function Shepherd called for the attempt run.
`viz.task_contract` renders its contract: the typed inputs the task declares, the artifact path it
writes to, and the docstring instruction the LLM receives when the task executes live.

The contract marks the boundary between the workspace-control layer and the generation step.
Everything above the contract — the workspace, the run record, the artifact ref — is Shepherd's
domain. Everything inside the task body is the generation step's domain.
""",
        ),
        (
            "code",
            """
from shepherd_usecases.visual_artifact import tasks
viz.task_contract(tasks.static_artifact_task)
""",
        ),
        (
            "markdown",
            """
## Cleanup

`attempt.output().discard()` releases the attempt's retained output without selecting it.
`reviewer.output().release()` releases the verdict. `workspace.close()` closes the controlling
scope. The run ends, but its trace stays in the flow: the full record of effects each run
performed and the dependencies between them.
""",
        ),
        ("code", "attempt.output().discard()\nreviewer.output().release()\nworkspace.close()\n"),
    ]
    write("visual_variant_studio_internals.ipynb", cells)


def uc2() -> None:
    cells: list[CellSpec] = [
        (
            "markdown",
            """
# Find the cheapest model that still does the job

In this guide, you'll run the same evaluator under a few model choices and keep the cheapest model
that still catches the bad output. The sample task is a visual QA check that flags broken
infographic tiles. The same pattern works for classifiers, extractors, routers, and other repeated
steps where cost matters.
""",
        ),
        *setup_cells(),
        (
            "markdown",
            """
## What this run does

Right-sizing runs the same evaluator under each model choice, grades every run against a
deterministic oracle, then keeps the cheapest model that still caught the bad tile.
""",
        ),
        (
            "markdown",
            """
## The input

The run starts with one prompt and a few model choices. `model_choices` maps each tier name to a
model identifier; only the model changes between evaluator runs.
""",
        ),
        ("code", "prompt = launch.default_prompt()\nmodel_choices = launch.model_choices()\nmodel_choices\n"),
        (
            "markdown",
            """
## Run Shepherd

Open a workspace for this comparison. Every evaluator run and the selector attach to it.
""",
        ),
        workspace_cell("model-right-sizing-lab"),
        (
            "markdown",
            """
Run the evaluator once per model choice and grade the results. Each run goes through
`launch.run_static` with a different model tier; `grade_runs` checks each verdict against a
deterministic oracle — a check that already knows which tile is broken — and marks each run as
passed or missed.
""",
        ),
        (
            "code",
            """
runs = {}
for config_name, model in model_choices.items():
    runs[config_name] = launch.run_static(
        workspace,
        name=f"rightsize-{config_name}",
        output_path=launch.VERDICT_PATH,
        output_content=launch.evaluator_content(config_name, model),
        model=model,
        metadata={"model_tier": config_name, "model": model},
    )

graded = launch.grade_runs(runs)
viz.show(viz.table([
    {
        "config": name,
        "model": item.model,
        "cost": item.cost,
        "catches hard fail": item.catches_hard_fail,
        "state": "passed" if item.passed else "missed",
    }
    for name, item in graded.items()
]))
""",
        ),
        (
            "markdown",
            """
With every run graded, run the selector. It receives each run's verdict as an artifact ref in
`verdict_refs` and reads them to pick the cheapest model that passed. `read_json` retrieves the
selector's decision from `DECISION_PATH`.
""",
        ),
        (
            "code",
            """
verdict_refs = {
    f"verdict_{name}": launch.artifact_ref(item.run, launch.VERDICT_PATH, label=name)
    for name, item in graded.items()
}
selector = launch.run_with_artifact_refs(
    workspace,
    name="selector",
    refs=verdict_refs,
    output_path=launch.DECISION_PATH,
    output_content=launch.selector_content(graded),
    after=[item.run for item in graded.values()],
)
decision = launch.read_json(selector, launch.DECISION_PATH)
decision
""",
        ),
        (
            "markdown",
            """
Mark the winning run as selected, discard the rest, and release the selector's output.
""",
        ),
        (
            "code",
            """
for name, item in graded.items():
    if name == decision["kept"]:
        item.run.output().select()
    else:
        item.run.output().discard()
selector.output().release()
""",
        ),
        (
            "markdown",
            """
## Trace

How does Shepherd run one task under many models and let a selector read them all? The trace below
captures every action an agent took while executing the tasks above, plus the relationships among
those actions. Click the nodes to learn more about the events.
""",
        ),
        (
            "code",
            'viz.show(viz.trace(workspace.flow, runs | {"selector": selector}, height="620px", detail="events"))\n',
        ),
        ("code", "workspace.close()\n"),
        (
            "markdown",
            "## Want to understand how it works?\n\nIf you want to understand how this works in greater detail, open the\n[Model Right-Sizing internals notebook](model_right_sizing_internals.ipynb).",
        ),
        (
            "markdown",
            "## Next steps\n\n- [Find the best approach by trying several alternatives](visual_variant_studio.ipynb)\n- [Recover a pipeline when one step drifts](visual_pipeline_recovery.ipynb)",
        ),
    ]
    write("model_right_sizing_lab.ipynb", cells)


def uc2_internals() -> None:
    cells: list[CellSpec] = [
        (
            "markdown",
            """
# Model Right-Sizing Internals

The short [Model Right-Sizing guide](model_right_sizing_lab.ipynb) ran one evaluator under a few
model choices and kept the cheapest that still caught the bad tile. This notebook rebuilds that run
and opens it up: the one task that runs under each model, the mechanical oracle that grades them,
and the selector that reads every run and decides.
""",
        ),
        *setup_cells(),
        (
            "markdown",
            """
## Build a run to inspect

Start by building the run this notebook takes apart. `launch.open_workspace` opens a workspace and
one flow, holding the prompt and the fixed candidate tiles every model is graded on. Each run runs
the same evaluator, changing one thing: the `model` it runs under. `launch.grade_runs` then scores
the finished runs against a mechanical oracle. `runs` holds the completed evaluator runs and
`graded` their oracle scores; the cells below read both.
""",
        ),
        ("code", "prompt = launch.default_prompt()\nmodel_choices = launch.model_choices()\nmodel_choices\n"),
        workspace_cell("model-right-sizing-internals"),
        (
            "code",
            """
runs = {}
for config_name, model in model_choices.items():
    runs[config_name] = launch.run_static(
        workspace,
        name=f"rightsize-{config_name}",
        output_path=launch.VERDICT_PATH,
        output_content=launch.evaluator_content(config_name, model),
        model=model,
        metadata={"model_tier": config_name, "model": model},
    )

graded = launch.grade_runs(runs)
""",
        ),
        (
            "markdown",
            """
## The oracle every run is graded against

You could judge each run with a reviewer: a run that reads the output and forms an opinion.
Right-sizing can't rely on that. The model is the thing on trial, so its judge has to be fixed and
model-free. That judge is a mechanical oracle, a deterministic gate that already knows which
candidate tile is broken. Grading a run is then a check for agreement: whether this model caught
the failure the oracle knows about. Each `GradedRun` carries the model it ran under, its cost tier,
whether it caught the hard failure, and whether it passed. Here is how the runs scored:
""",
        ),
        (
            "code",
            """
viz.show(viz.table([
    {
        "config": name,
        "model": item.model,
        "cost": item.cost,
        "catches hard fail": item.catches_hard_fail,
        "state": "passed" if item.passed else "missed",
    }
    for name, item in graded.items()
]))
""",
        ),
        (
            "markdown",
            """
## The selector that decides

After every evaluator run finishes, the selector runs. It takes an artifact ref to each evaluator's
`verdict.json`, follows those runs with `after`, grades them against that same oracle, and writes
one `decision.json`: the cheapest model that still passed. Because it cites every verdict, the
decision stays tied to exactly the runs it was based on.
""",
        ),
        (
            "code",
            """
verdict_refs = {
    f"verdict_{name}": launch.artifact_ref(item.run, launch.VERDICT_PATH, label=name)
    for name, item in graded.items()
}
selector = launch.run_with_artifact_refs(
    workspace,
    name="selector",
    refs=verdict_refs,
    output_path=launch.DECISION_PATH,
    output_content=launch.selector_content(graded),
    after=[item.run for item in graded.values()],
)
decision = launch.read_json(selector, launch.DECISION_PATH)
decision
""",
        ),
        (
            "markdown",
            """
## The durable run record

Each run leaves a record in the workspace: a durable `args_ref` naming the exact arguments it ran
on, a digest of those arguments, and the input refs it cited. The selector cited every verdict, so
its record carries them as input refs. `launch.run_record` and `launch.run_args` read that record
back out of the workspace.
""",
        ),
        (
            "code",
            """
record = launch.run_record(workspace, selector)
args = launch.run_args(workspace, selector)
{
    "run_ref": selector.run_ref,
    "args_ref": record.args_ref,
    "args_digest": record.args_digest,
    "input_ref_count": len(args["input_refs"]),
}
""",
        ),
        (
            "markdown",
            """
## Flow trace

The selector joins the evaluator runs as one more run, with an edge from each evaluator into it,
because it was handed those verdicts and read them. `workspace.flow.trace()` regenerates that
projection from the retained flow; the edges trace exactly what the decision was based on.
""",
        ),
        ("code", 'viz.show(viz.flow_trace(workspace.flow.trace(), height="620px"))\n'),
        (
            "markdown",
            """
## The task that runs under each model

`tasks.static_artifact_task` is the single task the workspace ran under each model. Nothing about
the task or its inputs changed between runs; only the `model` argument to `launch.run_static` did.
This is what right-sizing does: hold the task and its inputs fixed, vary the model, and see which
model still does the job. The contract below shows the task's inputs and docstring, without the
provider-owned body.
""",
        ),
        (
            "code",
            """
from shepherd_usecases.visual_artifact import tasks
viz.task_contract(tasks.static_artifact_task)
""",
        ),
        (
            "markdown",
            """
## Cleanup

All the runs — the evaluators and the selector — ran inside one workspace. Keep the decided tier as
the selected output, discard the rest, and release the selector. `workspace.close()` then releases
the workspace. The decision is made, but the trace stays: every run, the model each used, and the
verdicts the selector read to choose.
""",
        ),
        (
            "code",
            """
for name, item in graded.items():
    if name == decision["kept"]:
        item.run.output().select()
    else:
        item.run.output().discard()
selector.output().release()
workspace.close()
""",
        ),
    ]
    write("model_right_sizing_internals.ipynb", cells)


def uc3() -> None:
    cells: list[CellSpec] = [
        (
            "markdown",
            """
# Recover a pipeline when one step drifts

In this guide, you'll run a short pipeline, catch a bad middle step, and retry from the last good
boundary. The retry cites the retained plan artifact and causally follows the failed draft and
inspector — it does not write into an existing output as if it were a branch. The sample task is
an infographic tile. The same pattern works for code generation, data transforms, migrations, and
other pipelines where one bad step can spoil useful work.
""",
        ),
        *setup_cells(),
        (
            "markdown",
            """
## What this run does

Pipeline Recovery runs a two-step pipeline: a plan, then a draft. A reviewer catches the draft
going wrong, an inspector reads the trace to diagnose it, and the draft is retried as a new run
that cites the retained plan artifact (`plan_ref`), the last good boundary.
""",
        ),
        (
            "markdown",
            """
## The input

For this example, the run starts from one plain prompt. `launch.plan_for` turns it into a `brief`
and an initial `plan`; both are used as inputs to the pipeline's first task.
""",
        ),
        ("code", "prompt = launch.default_prompt()\nbrief, plan = launch.plan_for(prompt)\nplan\n"),
        (
            "markdown",
            """
## Run Shepherd

First, open a workspace for the prompt. Every run in this guide belongs to it.
""",
        ),
        workspace_cell("visual-pipeline-recovery"),
        (
            "markdown",
            """
Now run the pipeline: first the plan, then the draft. The plan is the last good boundary; the draft
is the step that drifts.
""",
        ),
        (
            "code",
            """
plan_run = launch.run_static(
    workspace,
    name="plan",
    output_path=launch.PLAN_PATH,
    output_content=plan,
    metadata={"logical_boundary": "plan"},
)
plan_ref = launch.artifact_ref(plan_run, launch.PLAN_PATH, label="retry-plan")

draft_v1 = launch.run_with_artifact_ref(
    workspace,
    name="draft-v1",
    ref_name="plan",
    artifact_ref=plan_ref,
    output_path=launch.ARTIFACT_PATH,
    output_text=launch.draft_html(brief, corrupt=True),
    after=[plan_run],
    metadata={"failed_run": "draft-v1"},
)
viz.show(viz.run_artifact(draft_v1, label="draft v1: wrong direction", accent="red"))
""",
        ),
        (
            "markdown",
            """
Two tasks examine the failed draft. First a reviewer flags it — an agent that reads the draft and
judges whether it met the brief. Then an inspector reads the reviewer's verdict and writes a
structured diagnosis: what drifted and what to change on retry. The reviewer's output is retained
read-only through `review_ref`; the inspector does not write into it.
""",
        ),
        (
            "code",
            """
draft_ref = launch.artifact_ref(draft_v1, label="failed-draft")
reviewer = launch.run_with_artifact_ref(
    workspace,
    name="review",
    ref_name="candidate",
    artifact_ref=draft_ref,
    output_path=launch.VERDICT_PATH,
    output_content=launch.review_content(prompt, {"draft_v1": draft_v1}),
    after=[draft_v1],
)
selection = launch.selection_from_review(reviewer)
issues = list(selection.candidates[0].get("issues", []))
review_ref = launch.artifact_ref(reviewer, launch.VERDICT_PATH, label="draft-review")
inspector = launch.run_with_artifact_ref(
    workspace,
    name="inspector",
    ref_name="review",
    artifact_ref=review_ref,
    output_path=launch.DIAGNOSIS_PATH,
    output_content=launch.diagnosis_content(issues),
    after=[reviewer],
)
launch.read_json(inspector, launch.DIAGNOSIS_PATH)
""",
        ),
        (
            "markdown",
            """
The inspector's diagnosis is packaged as `diagnosis_ref`. The retry is a new run — not a branch
of an existing one. It cites both `plan_ref` (the retained plan artifact, the last good boundary)
and `diagnosis_ref` as artifact refs, and declares `after=[plan_run, inspector]` so the workspace
trace records the causal chain. The failed draft stays untouched as evidence.
""",
        ),
        (
            "code",
            """
diagnosis_ref = launch.artifact_ref(inspector, launch.DIAGNOSIS_PATH, label="retry-diagnosis")
retry = launch.run_with_artifact_refs(
    workspace,
    name="retry",
    refs={"plan": plan_ref, "diagnosis": diagnosis_ref},
    output_path=launch.ARTIFACT_PATH,
    output_text=launch.draft_html(brief, corrupt=False),
    after=[plan_run, inspector],
    metadata={"retry_run": "retry-from-plan"},
)
viz.show(viz.side_by_side([
    viz.run_artifact(draft_v1, label="failed draft", accent="red"),
    viz.run_artifact(retry, label="retry from cited plan", accent="green"),
]))
""",
        ),
        (
            "markdown",
            """
The two draft outputs now sit side by side: the failed run is still available as evidence, and the
retry is the result of the new run with the fix applied. Now formalize the decision: discard the
failed draft, release the supporting runs, and select the retry.
""",
        ),
        (
            "code",
            """
draft_v1.output().discard()
plan_run.output().release()
reviewer.output().release()
inspector.output().release()
retry.output().select()
""",
        ),
        (
            "markdown",
            """
## Trace

`workspace.flow` captures every action taken across all runs above, plus the causal relationships
among them — including the link from `plan_run` and `inspector` to `retry`. Click the nodes to
explore individual events.
""",
        ),
        (
            "code",
            """
viz.show(viz.trace(
    workspace.flow,
    {"plan": plan_run, "draft_v1": draft_v1, "review": reviewer, "inspector": inspector, "retry": retry},
    height="680px",
    detail="events",
))
""",
        ),
        ("code", "workspace.close()\n"),
        (
            "markdown",
            "## Want to understand how it works?\n\nIf you want to understand how this works in greater detail, open the\n[Pipeline Recovery internals notebook](visual_pipeline_recovery_internals.ipynb).",
        ),
        (
            "markdown",
            "## Next steps\n\n- [Find the best approach by trying several alternatives](visual_variant_studio.ipynb)\n- [Find the cheapest model that still does the job](model_right_sizing_lab.ipynb)",
        ),
    ]
    write("visual_pipeline_recovery.ipynb", cells)


def uc3_internals() -> None:
    cells: list[CellSpec] = [
        (
            "markdown",
            """
# Pipeline Recovery Internals

The short [Pipeline Recovery guide](visual_pipeline_recovery.ipynb) ran a plan, caught a draft that
drifted, and retried from the last good boundary. This notebook rebuilds that run and opens it up:
a plan held as a logical boundary, a failed draft, a reviewer and an inspector, durable artifact
refs, and a retry that CITES the retained plan instead of writing into it. Along the way it reveals
the retained run record, the durable argument refs, and a regenerated trace projection.
""",
        ),
        *setup_cells(),
        (
            "markdown",
            """
## Build a run to inspect

Start by building the run this notebook takes apart. `launch.plan_for` turns the prompt into a brief
and a plan; `launch.open_workspace` opens one flow to hold every run. The cells below run each step
by name and read them back to draw the trace and to drive the retry.
""",
        ),
        ("code", "prompt = launch.default_prompt()\nbrief, plan = launch.plan_for(prompt)\nplan\n"),
        workspace_cell("visual-pipeline-recovery-internals"),
        (
            "markdown",
            """
## The plan boundary and the draft that drifted

The pipeline is two runs made in order. `plan_run` writes the plan and stands as the logical
boundary; `launch.artifact_ref` freezes a durable citation to it. The draft run follows
`after=[plan_run]` and cites that plan ref, so it builds on the plan instead of starting from
scratch. `corrupt=True` makes this draft drift uphill, away from the minimum. The failed artifact
renders in red.
""",
        ),
        (
            "code",
            """
plan_run = launch.run_static(
    workspace,
    name="plan",
    output_path=launch.PLAN_PATH,
    output_content=plan,
    metadata={"logical_boundary": "plan"},
)
plan_ref = launch.artifact_ref(plan_run, launch.PLAN_PATH, label="retry-plan")

draft_v1 = launch.run_with_artifact_ref(
    workspace,
    name="draft-v1",
    ref_name="plan",
    artifact_ref=plan_ref,
    output_path=launch.ARTIFACT_PATH,
    output_text=launch.draft_html(brief, corrupt=True),
    after=[plan_run],
    metadata={"failed_run": "draft-v1"},
)
viz.show(viz.run_artifact(draft_v1, label="draft v1: drifts uphill", accent="red"))
""",
        ),
        (
            "markdown",
            """
## The reviewer and inspector read what the draft recorded

Neither run re-executes the draft; each cites a retained artifact and reads it. The reviewer cites
the failed draft and writes a verdict; `launch.selection_from_review` reads that verdict back and
the draft's issues fall out of it. The inspector then cites the review, classifies the drift with
`launch.diagnosis_content`, and writes a diagnosis. Citing the runs as durable refs is what lets
each read the other's output without running anything again.
""",
        ),
        (
            "code",
            """
draft_ref = launch.artifact_ref(draft_v1, label="failed-draft")
reviewer = launch.run_with_artifact_ref(
    workspace,
    name="review",
    ref_name="candidate",
    artifact_ref=draft_ref,
    output_path=launch.VERDICT_PATH,
    output_content=launch.review_content(prompt, {"draft_v1": draft_v1}),
    after=[draft_v1],
)
selection = launch.selection_from_review(reviewer)
issues = list(selection.candidates[0].get("issues", []))
review_ref = launch.artifact_ref(reviewer, launch.VERDICT_PATH, label="draft-review")
inspector = launch.run_with_artifact_ref(
    workspace,
    name="inspector",
    ref_name="review",
    artifact_ref=review_ref,
    output_path=launch.DIAGNOSIS_PATH,
    output_content=launch.diagnosis_content(issues),
    after=[reviewer],
)
launch.read_json(inspector, launch.DIAGNOSIS_PATH)
""",
        ),
        (
            "markdown",
            """
## The retry cites the retained plan

The retry is a NEW run. It cites the retained plan ref and the diagnosis ref, follows
`after=[plan_run, inspector]`, and renders `corrupt=False` so it lands back on track. The plan is
never reopened as a writable branch: the retry follows it by citation. The failed draft and the
retry render side by side, red and green.
""",
        ),
        (
            "code",
            """
diagnosis_ref = launch.artifact_ref(inspector, launch.DIAGNOSIS_PATH, label="retry-diagnosis")
retry = launch.run_with_artifact_refs(
    workspace,
    name="retry",
    refs={"plan": plan_ref, "diagnosis": diagnosis_ref},
    output_path=launch.ARTIFACT_PATH,
    output_text=launch.draft_html(brief, corrupt=False),
    after=[plan_run, inspector],
    metadata={"retry_run": "retry-from-plan"},
)
viz.show(viz.side_by_side([
    viz.run_artifact(draft_v1, label="failed draft", accent="red"),
    viz.run_artifact(retry, label="retry from cited plan", accent="green"),
]))
""",
        ),
        (
            "markdown",
            """
## What the retry actually is

The retry's structure lives in its retained run record and durable args, not in any mutated output.
`launch.run_record` returns the record with its `args_ref` and `args_digest`; `launch.run_args`
resolves the durable arg payload so its input refs can be counted, one per cited artifact.
`launch.changed_paths` shows exactly what the retry wrote, which is a fresh artifact, not an edit of
the plan.
""",
        ),
        (
            "code",
            """
record = launch.run_record(workspace, retry)
args = launch.run_args(workspace, retry)
{
    "run_ref": retry.run_ref,
    "args_ref": record.args_ref,
    "args_digest": record.args_digest,
    "input_ref_count": len(args["input_refs"]),
    "changed_paths": launch.changed_paths(retry),
}
""",
        ),
        (
            "markdown",
            """
## Flow trace

`workspace.flow.trace()` returns the flow's projection, plan to retry. `viz.flow_trace` renders it
as a self-contained table of events and edges. Follow the edges: the retry links back to the plan
and the inspector, and the failed draft stays recorded as evidence.
""",
        ),
        ("code", 'viz.show(viz.flow_trace(workspace.flow.trace(), height="680px"))\n'),
        (
            "markdown",
            """
## Task contract

Every run above went through one provider-owned task. `viz.task_contract` shows that task's public
contract, its inputs and docstring, without the provider fixture body.
""",
        ),
        (
            "code",
            """
from shepherd_usecases.visual_artifact import tasks
viz.task_contract(tasks.static_artifact_task)
""",
        ),
        (
            "markdown",
            """
## Cleanup

Retention is expressed with the output verbs, not by overwriting anything. The failed draft is
discarded, the plan, review, and diagnosis are released, and the retry is selected as the kept
result. `workspace.close()` releases the flow. Nothing above was overwritten: the failed run, its
trace, and the diagnosis all remain, which is what let the inspector work from recorded output in
the first place.
""",
        ),
        (
            "code",
            """
draft_v1.output().discard()
plan_run.output().release()
reviewer.output().release()
inspector.output().release()
retry.output().select()
workspace.close()
""",
        ),
    ]
    write("visual_pipeline_recovery_internals.ipynb", cells)


def template() -> None:
    write("_usecase_template.ipynb", [("markdown", "# Visual artifact notebook template"), ("code", SETUP)])


BUILDERS = {
    "uc1": uc1,
    "uc1-internals": uc1_internals,
    "uc2": uc2,
    "uc2-internals": uc2_internals,
    "uc3": uc3,
    "uc3-internals": uc3_internals,
    "template": template,
}


def main(argv: list[str]) -> None:
    global CHECK_MODE

    args = argv[1:]
    if "--check" in args:
        CHECK_MODE = True
        args = [arg for arg in args if arg != "--check"]
    targets = args or ["uc1", "uc1-internals", "uc2", "uc2-internals", "uc3", "uc3-internals", "template"]
    for target in targets:
        try:
            BUILDERS[target]()
        except KeyError as exc:
            raise SystemExit(f"unknown notebook target: {target}") from exc


if __name__ == "__main__":
    main(sys.argv)
