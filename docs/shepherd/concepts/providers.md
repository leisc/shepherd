# Providers

> Page status: release-ready
> Source state: shipped-source
> Applies to: Shepherd 0.1
> Owner: @docs-system-owner (TBD)
> Validation: scripts/check_shepherd_docs.py

*Concept. The mental model behind Shepherd. Steps live in the tutorial, signatures in the reference.*

A **provider** is the binding between a task and the model backend that answers
it. A task declares *what* it wants, a typed contract and a docstring, but
never *who* answers. That choice is made once, in the workspace, and every task
call in scope inherits it.

```python
import shepherd as shp
from shepherd.providers import claude

with shp.workspace(model=claude("sonnet-4-5")):
    ...  # every task call in here is answered by that provider + model
```

`claude("sonnet-4-5")` is a provider *selection*, an inert token naming a
backend and a model. You hand it to the workspace; you do not call it yourself.
The task signatures stay untouched: point the same tasks at a different backend
by changing only that one argument.

## The provider is chosen by the workspace, never by the task

This is the load-bearing split. A task is **model-agnostic** by construction,
nothing in its signature names a provider, and nothing should. The *caller's*
workspace supplies the provider, the same way it supplies the working root and
any shared context. One consequence you can rely on: the same task is a unit you
can re-target without editing it, and a task called with no workspace open fails
immediately rather than reaching for a hidden default.

That is why "which model" is a property of a *run*, not of a *task*. The task is
the contract; the provider is the situation it runs in.

## The offline provider is a real provider

Every documented example runs against a **recorded,
deterministic offline provider**, no credentials, no network. It is not a mock
bolted on for tests; it is a provider like any other, selected the same way, and
it is the one the docs and CI use so that what you read is what ran. Live
providers exist alongside it; they cost money and vary run to run, which is
exactly why everyday development and CI stay on the offline one.

## What a provider is *not*

- **Not credentials.** Selecting a provider in code (`claude("sonnet-4-5")`) is
  separate from *recording* its API key. The public docs here cover provider
  selection, not live credential management.
- **Not a global.** There is no module-level "current provider" you set once and
  forget. The provider lives in the workspace scope, so it is explicit and
  local, a `with` block, not a singleton.
- **Not something a task reaches for.** A task does not inspect or pick its
  provider at runtime. If you need different tasks on different models, you
  scope them to different workspaces, see
  [Route tasks to models](../guides/route-tasks-to-models.md).

## Where providers sit

A provider is the executor a [workspace](workspaces.md) binds; calling a
[task](tasks.md) inside that workspace produces a [run](runs.md) answered by
that provider. To see one selected and pinned in working code, start with the
[first Shepherd app tutorial](../tutorials/first-shepherd-app.md).
