"""SessionState — the shared per-session state VcsCore collaborators read.

A frozen-reference struct of the reference-stable members set in
``VcsCore.__init__`` (D-C, amended 2026-07-05 after the forward-exploration pass).
P3 collaborators take ``state: SessionState`` plus their own few injected callables
for non-stable / behavioural dependencies, instead of ``owner: VcsCore`` — which is
what breaks the ``_vcscore_* <-> vcscore`` import cycles.

This is a leaf module: it imports VcsCore-side types only under ``TYPE_CHECKING``,
so no collaborator that depends on it acquires a runtime edge back to ``vcscore``.
Only the reference-stable members live here; lazily-rebound state (``_ground``,
``_world_storage_manager``) and owner methods are injected per collaborator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable

    from vcs_core._patch_manager import PatchManager
    from vcs_core._world_storage_manager import WorldStorageManager
    from vcs_core.recording import RecordingPipeline
    from vcs_core.store import Store
    from vcs_core.types import ScopeInfo


@dataclass(frozen=True)
class SessionState:
    """Reference-stable session members, constructed once by ``VcsCore.__init__``.

    ``active_scopes`` is a live dict mutated in place by the owner; the binding is
    stable so holding the reference is correct. ``world_storage`` is the lazy
    accessor (the manager itself is built on first use, so it is a callable, not a
    field) per the D-C state-vs-method finding.
    """

    store: Store
    active_scopes: dict[str, ScopeInfo]
    lock: threading.RLock
    pipeline: RecordingPipeline
    patch_manager: PatchManager
    repo_path: str
    session_id: str
    world_storage: Callable[[], WorldStorageManager]
