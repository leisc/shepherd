# Permissions

> Page status: scaffold
> Source state: preview
> Applies to: Shepherd 0.1
> Owner: @docs-system-owner (TBD)
> Validation: scripts/check_shepherd_docs.py

*Concept. The mental model behind Shepherd. Steps live in the tutorial, signatures in the reference.*

!!! warning "Not shipped yet"
    The permission *vocabulary* here, `may=`, the per-parameter `May[...]`
    grants, named profiles, and clamp-at-spawn, is implemented in the runtime
    and partly enforced, but not yet re-exported on the public `shepherd`
    surface. `may=Permissive` works today on the top-level import (and
    `may=ReadOnly`, with `ReadOnly` imported from the `shepherd.profiles`
    submodule); the per-parameter `May[...]` typing, the richer profile names shown below
    (e.g. `ModelCalls`), and full clamp-at-spawn enforcement are design
    vocabulary you can build against, not yet the importable surface.

A task's **permissions are part of its signature**. Just as the return type
declares what you get back, a task's grants declare what it is allowed to do,
which kinds of effects it may perform, and over which resources. Reading the
signature *is* reading the permission surface; there is no second, hidden policy
file that the code merely approximates.

```python
@shp.task(may=ModelCalls)
def support_agent(
    mail:   May[Mailbox, ReadOnly],
    outbox: May[Mailbox, EmailSend.where(to__all__endswith="@acme.com")],
    ticket: Ticket,
) -> Resolution:
    """Read context from the mailbox; draft a reply to the customer."""
```

Two homes for authority, and they never overlap:

- **`may=` is the task-wide ceiling.** It carries the authority that has no
  single parameter to live on, model calls, and the whole-task limit that
  every parameter is also held under. `may=ModelCalls` says this task may call
  the model and do nothing else to the world on its own.
- **`May[Resource, ...]` is per-parameter authority.** Where authority is about
  a *specific* resource the task is handed, it rides that parameter. `mail` may
  only be read; `outbox` may send, but only to `@acme.com` addresses. The
  constraint is right there in the type.

## Authority narrows, never widens

The load-bearing rule is **clamp at spawn**: when a resource is passed into a
task, its authority is intersected with the grant on that parameter. You can
hand in something broader, a fully writable mailbox, and the task still runs
with only what its signature allows. You can never hand in something that makes
the task *more* powerful than its declaration. Permissions only ever shrink as
they cross into a task, so a task's declared surface is a true upper bound on
what it can do, regardless of who called it or what they passed.

This is why permissions do not depend on caller discipline. A careful caller
and a careless one get the same enforced surface, because the surface is a
property of the *task definition*, checked at the boundary, not a convention
the caller is trusted to honor.

## Least privilege is a value-level move

Narrowing a resource before you pass it on, handing a child task a read-only
view of something you can write, is ordinary hygiene, done with attenuation
(`handle.readonly()`, `handle.allow_only(...)`). It is how you practice least
privilege, but it is never what *enforces* the limit: enforcement is the clamp,
which happens whether or not the caller attenuated anything.

## What permissions are *not*

- **Not runtime trust.** A task does not ask, at runtime, whether it is allowed
  to do something and hope the answer is yes. The allowed surface is fixed at
  definition time and bounds the body no matter how it tries to act.
- **Not capabilities discovered in production.** Authority is declared up front
  and defaults to deny: a kind of effect, or a resource, that the signature
  never names is refused, not silently permitted.
- **Not a sandbox bolted on beside the code.** The grant lives on the parameter,
  so the security surface and the program are the same artifact. There is no
  separate policy document to drift out of sync.

## Where permissions sit

Permissions are the authority half of a [task's](tasks.md) contract: the
signature says both *what it computes* (parameters and return type) and *what it
may touch*. What actually crosses the boundary at runtime are
[effects](effects.md), and the resources they act on are the
[runtime substrate](runtime-substrate.md), handles that carry their own
authority. Permissions are the rules; effects are the traffic; the substrate is
the world being governed.
