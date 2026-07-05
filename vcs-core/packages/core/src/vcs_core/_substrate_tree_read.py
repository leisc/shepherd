"""Read workspace bytes from a tree-backed substrate revision.

Tranche 3 of the tree-backed workspace migration uses this helper to source
materialization bytes from the substrate's embedded ``workspace/`` Git tree
rather than from the scalar coordinator workspace. The two are byte-equivalent
because alternates make the same blobs visible from both stores; the change is
about which surface materialization depends on.

This module is private. Callers should go through
:meth:`vcs_core.vcscore.VcsCore._read_v2_workspace_file_for_materialization`,
which selects the substrate tree only when the ground world's workspace head
is tree-backed and falls back to scalar otherwise.
"""

from __future__ import annotations

import pygit2


def read_substrate_workspace_file(
    substrate_repo: pygit2.Repository, head_oid: str, path: str
) -> tuple[bytes, int] | None:
    """Read ``path`` from the ``workspace/`` subtree of a substrate revision.

    Returns ``(content, filemode)`` on success, or ``None`` when:

    - the commit is absent from the substrate ODB;
    - the commit has no ``workspace/`` entry (digest-only revision);
    - the workspace tree is unreachable from the substrate ODB (e.g. the
      alternates link to coord has been removed);
    - the path does not resolve to a blob in the workspace tree.

    All None returns are graceful: callers fall back to the scalar workspace
    tree. ``repo.get(...)`` (not ``repo[...]``) is used so a missing object
    yields ``None`` instead of raising ``KeyError`` — important for the
    alternates-removed case where the commit tree references workspace blobs
    that live in coord's ODB.
    """
    try:
        commit = substrate_repo[pygit2.Oid(hex=head_oid)]
    except (KeyError, ValueError):
        return None
    if not isinstance(commit, pygit2.Commit):
        return None
    try:
        workspace_entry = commit.tree["workspace"]
    except KeyError:
        return None
    if workspace_entry.filemode != pygit2.GIT_FILEMODE_TREE:
        return None
    workspace_tree = substrate_repo.get(workspace_entry.id)
    if not isinstance(workspace_tree, pygit2.Tree):
        return None
    return _read_path(substrate_repo, workspace_tree, path)


def _read_path(repo: pygit2.Repository, root: pygit2.Tree, path: str) -> tuple[bytes, int] | None:
    parts = path.split("/")
    current: pygit2.Tree = root
    for part in parts[:-1]:
        try:
            sub_entry = current[part]
        except KeyError:
            return None
        if sub_entry.filemode != pygit2.GIT_FILEMODE_TREE:
            return None
        sub_tree = repo.get(sub_entry.id)
        if not isinstance(sub_tree, pygit2.Tree):
            return None
        current = sub_tree
    try:
        leaf_entry = current[parts[-1]]
    except KeyError:
        return None
    if leaf_entry.filemode not in (
        pygit2.GIT_FILEMODE_BLOB,
        pygit2.GIT_FILEMODE_BLOB_EXECUTABLE,
    ):
        return None
    blob = repo.get(leaf_entry.id)
    if not isinstance(blob, pygit2.Blob):
        return None
    return bytes(blob.data), int(leaf_entry.filemode)
