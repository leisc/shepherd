# CLI

> Page status: release-ready
> Source state: shipped-source
> Applies to: Shepherd v0.2.0
> Owner: @docs-system-owner (TBD)
> Validation: `shepherd --help` (0.2.0 wheel)

*Reference. Exact, generated facts. The mental model lives in concepts, recipes in guides.*

The `shepherd` command (also installed as `sp`) ships in 0.2.0. Read-only listings
accept `--json` for a durable machine payload.

## `shepherd`

```text
Usage: shepherd [OPTIONS] COMMAND [ARGS]...

  Shepherd — effect-based AI agent framework.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  demo     Emit checked-in quickstart demo scripts.
  doctor   Check whether the current directory is ready for the quickstart.
  init     Initialize PATH as a Shepherd workspace.
  package  Create and manage Shepherd extension packages.
  run      Inspect runs and settle retained outputs; start is a fenced entry point.
  task     Manage and inspect task-library entries.
```

## First run — `init`, `doctor`, `demo`

```text
shepherd init [PATH]              # create/reuse .vcscore; validate the workspace-control substrate
shepherd doctor                   # check the current directory is ready for the quickstart
shepherd doctor claude            # check the live Claude runtime lane is available
shepherd demo write NAME          # emit a checked-in demo script to stdout (e.g. quickstart, agent-task)
```

`init` takes `--backend [auto|clonefile|fuse|kernel|copy]` and `--adopt [none|git-head|worktree]`.
`doctor` takes `--json` and the same `--backend`.

## `shepherd run` — inspect and settle

`run` is the inspection and settlement group over retained run outputs.

```text
Usage: shepherd run COMMAND [ARGS]...

Commands:
  show          Show one run record.
  list          List run summaries from the selected ledger.
  changeset     Inspect the read-only changeset view for one retained output.
  outputs       List product run outputs after a run.
  trace         Print the materialized trace for a run.
  trace-revision  Print the run-trace summary for one trace revision.
  vcscore       Show the vcs-core citations carried by a run.
  select        Select one retained output into its live parent world.
  release       Release one retained output (consume without applying).
  discard       Discard one retained output as non-application.
  start         Run the fenced compatibility start path (see below).
```

**Read vs. settle — the identity rule.** Read commands (`show`, `changeset`, `trace`, …)
accept selectors: `--latest` and a unique short run-id prefix. Settlement commands
(`select` / `release` / `discard`) require an **exact** run identity and reject selectors —
settling the wrong run is not a mistake a prefix should be able to make. Settlement is
**consume-once**: after one of the three records its outcome, the others refuse for that output.

```bash
shepherd run changeset --latest              # what the latest run produced (read-only)
shepherd run changeset --latest --read out.py  # print one changed file's content
shepherd run select <exact-run-ref>          # advance the selected binding to the run's basis
shepherd run discard <exact-run-ref>         # drop it; trace + changeset evidence remain queryable
```

`changeset`/`select` take `--output-name` (default `workspace`) and `--binding` to scope to one
bound repository; `changeset` also takes `--state` and `--json`.

### `shepherd run start` is fenced

`run start` is a **fenced compatibility entry point**, not the normal launch path. It fails
closed unless you opt in with `SHEPHERD_ENABLE_FENCED_RUN_START=1`. The sanctioned Python launch
is `workspace.run(...)`; the sanctioned CLI inspection is the `run` read/settle commands above.

## `shepherd task` — the task library

```text
Usage: shepherd task COMMAND [ARGS]...

Commands:
  list      List task summaries from the selected task ledger.
  register  Register a task import path as an active task version.
  resolve   Resolve a task ref to an exact artifact lock.
  show      Show one task definition and its signature/permission surface.
```

`shepherd task show <name>` renders the task's **signature and permission surface** expanded —
for a task with per-binding grants it prints, e.g., `docs read-only / backend read-write`, so you
can read exactly what a task may touch before you run it.

## `shepherd package` — extension packages

```text
shepherd package init NAME        # scaffold a new Shepherd extension package
```
