# CLI

> Page status: scaffold
> Source state: checked-fixture
> Applies to: Shepherd v0.1.1-dev
> Owner: @docs-system-owner (TBD)
> Validation: scripts/gen_cli_reference.py --check

*Reference. Exact, generated facts. The mental model lives in concepts, recipes in guides.*

!!! warning "CLI not shipped yet"
    The Shepherd CLI has not shipped. This page previews the planned command
    surface; the commands below are not runnable yet.

The command groups follow: first-run (`init`, `doctor`, `demo`),
`provider`, `placement`, `workflow`, and `run`/`runs`. Read-only listings
support `--json`.

## `shepherd`

```text
Usage: shepherd [OPTIONS] COMMAND [ARGS]...

 Build and run agent systems.

Commands:
 init Create or update project configuration.
 doctor Report providers, placements, and capability gaps.
 demo Run the packaged first-run demo (--offline by default).
 provider List, show, login, and check model providers.
 placement List, show, check, and configure runnable environments.
 workflow List, show, install, configure, and run packaged workflows.
 run Run a task or workflow by id.
 runs List and inspect recorded runs.
```

## `shepherd demo`

```text
Usage: shepherd demo [--offline]

 Run the packaged first-run demo.

 --offline Use the deterministic offline provider (no credentials, no cost).
```

## `shepherd workflow`

```text
Usage: shepherd workflow COMMAND [ARGS]...

Commands:
 list List installed workflow manifests. [--json]
 show Show one workflow's manifest. [--json]
 install Install a workflow package from PyPI.
 configure Write project/local workflow configuration.
 run Run a workflow by canonical id.
```
