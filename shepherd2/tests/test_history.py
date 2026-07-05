from __future__ import annotations

from shepherd2 import (
    AppendBatch,
    AppendContext,
    AppendGroup,
    FactDraft,
    ReadContext,
    SQLiteTraceStore,
    complete_execution_batch,
    create_execution_batch,
    create_execution_relation_batch,
    execution_id_for,
    project_effective_history_from_store,
    project_execution_relations,
    publish_execution_frontier,
    relation_id_for,
)

TRUSTED = AppendContext(
    actor_ref="runtime:test",
    presented_witness_refs=("trusted:internal",),
    schema_version_set="shepherd2-slice-a",
    trust_mode="internal",
)
READER = ReadContext(actor_ref="reader")


def test_parent_owned_relation_projects_child_from_retained_frontier() -> None:
    store = SQLiteTraceStore()
    parent_id = execution_id_for("intent:parent:create")
    child_id = execution_id_for("intent:child:create")
    relation_id = relation_id_for("intent:parent:spawn-child")

    parent_create = store.append(
        TRUSTED,
        create_execution_batch(
            append_intent_id="intent:parent:create",
            execution_id=parent_id,
            task_ref="Parent",
            inputs={"x": 10},
        ),
    )
    relation = store.append(
        TRUSTED,
        create_execution_relation_batch(
            append_intent_id="intent:parent:spawn-child",
            relation_id=relation_id,
            relation_kind="spawned",
            parent_execution_id=parent_id,
            child_execution_id=child_id,
            child_frontier_id="frontier:child-terminal",
            caused_by=(parent_create.fact_ids[-1],),
        ),
    )
    child_create = store.append(
        TRUSTED,
        create_execution_batch(
            append_intent_id="intent:child:create",
            execution_id=child_id,
            task_ref="Child",
            inputs={"x": 10},
            parent_execution_id=parent_id,
            caused_by=relation.fact_ids,
        ),
    )
    child_terminal = store.append(
        TRUSTED,
        complete_execution_batch(
            append_intent_id="intent:child:complete",
            execution_id=child_id,
            outputs={"y": 11},
            caused_by=(child_create.fact_ids[-1],),
        ),
    )
    child_cutoff = publish_execution_frontier(
        store,
        TRUSTED,
        frontier_id="frontier:child-terminal",
        target_execution_id=child_id,
        through_fact_id=child_terminal.fact_ids[-1],
        publisher_execution_id=parent_id,
        caused_by=relation.fact_ids,
    )
    parent_terminal = store.append(
        TRUSTED,
        complete_execution_batch(
            append_intent_id="intent:parent:complete",
            execution_id=parent_id,
            outputs={"done": True},
            caused_by=(child_cutoff.created_by_fact_id or child_terminal.fact_ids[-1],),
        ),
    )
    parent_cutoff = publish_execution_frontier(
        store,
        TRUSTED,
        frontier_id="frontier:parent-terminal",
        target_execution_id=parent_id,
        through_fact_id=parent_terminal.fact_ids[-1],
    )

    parent_slice = store.resolve_frontier(READER, parent_cutoff.frontier_id)
    relations = project_execution_relations(parent_slice, parent_id)
    history = project_effective_history_from_store(store, READER, parent_cutoff)

    assert len(relations) == 1
    assert relations[0].relation_id == relation_id
    assert relations[0].relation_kind == "spawned"
    assert relations[0].parent_execution_id == parent_id
    assert relations[0].child_execution_id == child_id
    assert relations[0].created_fact_id == relation.fact_ids[0]
    assert history.root.outputs == {"done": True}
    assert len(history.children) == 1
    assert history.children[0].relation == relations[0]
    assert history.children[0].execution.status == "succeeded"
    assert history.children[0].execution.outputs == {"y": 11}


def test_abandoned_relation_stays_retained_but_drops_effective_child() -> None:
    store = SQLiteTraceStore()
    parent_id = execution_id_for("intent:parent:create")
    child_id = execution_id_for("intent:child:create")
    relation_id = relation_id_for("intent:parent:spawn-child")
    parent_create = store.append(
        TRUSTED,
        create_execution_batch(
            append_intent_id="intent:parent:create",
            execution_id=parent_id,
            task_ref="Parent",
            inputs={},
        ),
    )
    spawned = store.append(
        TRUSTED,
        create_execution_relation_batch(
            append_intent_id="intent:parent:spawn-child",
            relation_id=relation_id,
            relation_kind="spawned",
            parent_execution_id=parent_id,
            child_execution_id=child_id,
            child_frontier_id="frontier:child-terminal",
            caused_by=(parent_create.fact_ids[-1],),
        ),
    )
    abandoned = store.append(
        TRUSTED,
        create_execution_relation_batch(
            append_intent_id="intent:parent:abandon-child",
            relation_id=relation_id,
            relation_kind="abandoned",
            parent_execution_id=parent_id,
            child_execution_id=child_id,
            child_frontier_id="frontier:child-terminal",
            caused_by=spawned.fact_ids,
        ),
    )
    parent_terminal = store.append(
        TRUSTED,
        complete_execution_batch(
            append_intent_id="intent:parent:complete",
            execution_id=parent_id,
            outputs={},
            caused_by=abandoned.fact_ids,
        ),
    )
    parent_cutoff = publish_execution_frontier(
        store,
        TRUSTED,
        frontier_id="frontier:parent-terminal",
        target_execution_id=parent_id,
        through_fact_id=parent_terminal.fact_ids[-1],
    )

    history = project_effective_history_from_store(store, READER, parent_cutoff)

    assert [relation.relation_kind for relation in history.relations] == ["spawned", "abandoned"]
    assert history.children == ()


def test_project_effective_history_includes_parent_published_facts() -> None:
    store = SQLiteTraceStore()
    parent_id = execution_id_for("intent:parent:create")
    parent_create = store.append(
        TRUSTED,
        create_execution_batch(
            append_intent_id="intent:parent:create",
            execution_id=parent_id,
            task_ref="Parent",
            inputs={},
        ),
    )
    store.append(
        TRUSTED,
        AppendBatch(
            append_intent_id="intent:parent:publish",
            groups=(
                AppendGroup(
                    trace_owner_id=parent_id,
                    causal_parents=(parent_create.fact_ids[-1],),
                    fact_drafts=(
                        FactDraft(
                            kind_label="fact_published",
                            mode="capture",
                            schema_ref="shepherd2.runtime.published_fact.v1",
                            payload={"kind": "note", "data": {"body": "hello"}},
                        ),
                    ),
                ),
            ),
        ),
    )
    parent_terminal = store.append(
        TRUSTED,
        complete_execution_batch(
            append_intent_id="intent:parent:complete",
            execution_id=parent_id,
            outputs={},
        ),
    )
    parent_cutoff = publish_execution_frontier(
        store,
        TRUSTED,
        frontier_id="frontier:parent-terminal",
        target_execution_id=parent_id,
        through_fact_id=parent_terminal.fact_ids[-1],
    )

    history = project_effective_history_from_store(store, READER, parent_cutoff)

    assert len(history.published_facts) == 1
    assert history.published_facts[0].kind == "note"
    assert history.published_facts[0].data == {"body": "hello"}
