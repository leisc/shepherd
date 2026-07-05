"""Shared publisher for the selected Shepherd run ledger.

``shepherd.runs`` is product/control authority for run lifecycle rows, status
summaries, launch identity, and output citations. It is not retained-output
custody, settlement authority, or trace authority. Mutations should write
addressable keyed records through :class:`RunLedgerStore`; synthesized full
payloads are compatibility/query material only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from vcs_core.spi import KeyedJsonPut, KeyedJsonTreeDraft

from shepherd_dialect.workspace_control.drivers import mint_ledger_write_authority
from shepherd_dialect.workspace_control.flow_context import FLOW_RUN_SCHEMA, FLOW_SCHEMA
from shepherd_dialect.workspace_control.input_refs import RUN_ARGS_SCHEMA, run_args_ref
from shepherd_dialect.workspace_control.ledger_contracts import (
    FLOW_RUNS,
    FLOWS,
    RUN_ARGS,
    RUN_LEDGER_BINDING,
    RUN_LEDGER_SCHEMA,
    RUN_LEDGER_STORAGE_SHAPE,
    RUN_RECORDS,
)

if TYPE_CHECKING:
    from vcs_core.keyed_json_tree import KeyedJsonTreeStore

    from shepherd_dialect.workspace_control.schemas import RunRecord, RunSummary

JsonObject = dict[str, object]
_RUN_RECORDS = RUN_RECORDS
_RUN_ARGS = RUN_ARGS
_FLOWS = FLOWS
_FLOW_RUNS = FLOW_RUNS
_AUXILIARY_STORES = (_RUN_ARGS, _FLOWS, _FLOW_RUNS)


class RunLedgerPublishError(ValueError):
    """Raised when the selected run ledger cannot publish a revision."""


def canonical_digest(value: object) -> str:
    """Return the canonical JSON SHA-256 digest used by workspace-control ledgers."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def utc_now() -> str:
    """Return the ledger timestamp format used by workspace-control records."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class RunLedgerStore:
    """Typed run-ledger facade over the selected ``shepherd.runs`` binding.

    The selected revision payload is a compact manifest for keyed ledgers.
    Product mutation paths should use this facade instead of synthesizing and
    mutating a whole compatibility payload.
    """

    def __init__(self, mg: Any, *, scope: Any = None) -> None:
        self._mg = mg
        self._scope = scope

    @property
    def scope(self) -> Any:
        return self._scope

    def _write_scope(self) -> Any:
        return self._mg.ground if self._scope is None else self._scope

    def get(self, run_ref: str) -> RunRecord | None:
        """Return one exact run record without reading the synthesized ledger payload."""
        from shepherd_dialect.workspace_control.schemas import RunRecord

        raw = _read_run_record_json(self._mg, run_ref, scope=self._scope)
        if raw is not None:
            return RunRecord.from_json(raw)
        _manifest, _head, legacy_payload = selected_run_ledger_manifest_with_head(self._mg, scope=self._scope)
        raw = _legacy_run_json(legacy_payload, run_ref)
        return None if raw is None else RunRecord.from_json(raw)

    def latest(self) -> RunRecord | None:
        """Return the latest run record when the selected manifest names one."""
        manifest, _head, legacy_payload = selected_run_ledger_manifest_with_head(self._mg, scope=self._scope)
        if legacy_payload is not None:
            latest = _latest_run_ref_from_payload(run_ledger_payload(legacy_payload))
        else:
            latest = _manifest_latest(manifest)
        return None if latest is None else self.get(latest)

    def list_runs(
        self,
        *,
        status: str | None = None,
        task_id: str | None = None,
        max_count: int | None = None,
    ) -> tuple[RunSummary, ...]:
        """Return run summaries using keyed record reads for the normal v2 shape."""
        if max_count is not None and max_count < 0:
            raise ValueError("max_count must be non-negative")
        records = self.list_run_records()
        summaries = []
        for record in records:
            if status is not None and record.status != status:
                continue
            if task_id is not None and record.task_id != task_id:
                continue
            summaries.append(record.summary())
        if max_count is not None:
            summaries = summaries[-max_count:] if max_count else []
        return tuple(summaries)

    def list_run_records(self) -> tuple[RunRecord, ...]:
        """Return current run rows, ordered like the compatibility ledger projection."""
        from shepherd_dialect.workspace_control.schemas import RunRecord

        rows = _list_json_records(self._mg, _RUN_RECORDS, scope=self._scope)
        if not rows:
            _manifest, _head, legacy_payload = selected_run_ledger_manifest_with_head(self._mg, scope=self._scope)
            rows = _legacy_records_for_store(legacy_payload, _RUN_RECORDS)
        records = tuple(RunRecord.from_json(row) for row in rows)
        return tuple(sorted(records, key=lambda record: (record.started_at or "", record.run_ref)))

    def put_current(
        self,
        record: RunRecord,
        *,
        args_payload: Mapping[str, object] | None = None,
        flow_run_payload: Mapping[str, object] | None = None,
    ) -> str:
        """Append or replace one current run record."""
        return _publish_run_record_impl(
            self._mg,
            record,
            scope=self._write_scope(),
            args_payload=args_payload,
            flow_run_payload=flow_run_payload,
        )

    def append_resolution(self, run_ref: str, resolution: Any) -> str:
        """Append one task-resolution record to an existing run row."""
        return _update_run_record_impl(
            self._mg,
            run_ref,
            lambda record: replace(
                record,
                task_resolutions=_append_by_attr(record.task_resolutions, resolution, "resolution_id"),
            ),
            scope=self._write_scope(),
            missing_message=f"cannot record task resolution for missing run {run_ref!r}",
        )

    def append_execution(self, run_ref: str, execution: Any) -> str:
        """Append one task-execution record to an existing run row."""
        return _update_run_record_impl(
            self._mg,
            run_ref,
            lambda record: replace(
                record,
                task_executions=_append_by_attr(record.task_executions, execution, "execution_id"),
            ),
            scope=self._write_scope(),
            missing_message=f"cannot record task execution for missing run {run_ref!r}",
        )

    def _put_auxiliary(self, store: KeyedJsonTreeStore, key: str, payload: Mapping[str, object]) -> str:
        """Append or replace one keyed auxiliary run-ledger record."""
        _require_auxiliary_store(store)
        return _publish_auxiliary_record_impl(self._mg, store, key, payload, scope=self._write_scope())

    def put_args(self, args_ref: str, payload: Mapping[str, object]) -> str:
        """Append or replace one persisted run-argument payload by exact args ref."""
        _validate_run_args_payload(payload)
        if _required_payload_id(payload, "args_ref") != args_ref:
            raise RunLedgerPublishError("run args payload args_ref disagrees with keyed args ref")
        return self._put_auxiliary(_RUN_ARGS, args_ref, payload)

    def put_flow(self, flow_id: str, payload: Mapping[str, object]) -> str:
        """Append or replace one workflow metadata record by exact flow id."""
        _validate_flow_payload(payload)
        if _required_payload_id(payload, "flow_id") != flow_id:
            raise RunLedgerPublishError("flow payload flow_id disagrees with keyed flow id")
        return self._put_auxiliary(_FLOWS, flow_id, payload)

    def put_flow_run(self, run_ref: str, payload: Mapping[str, object]) -> str:
        """Append or replace one workflow/run edge record by exact run ref."""
        _validate_flow_run_payload(payload)
        if _required_payload_id(payload, "run_ref") != run_ref:
            raise RunLedgerPublishError("flow run payload run_ref disagrees with keyed run ref")
        return self._put_auxiliary(_FLOW_RUNS, run_ref, payload)

    def _get_auxiliary(self, store: KeyedJsonTreeStore, key: str) -> dict[str, object] | None:
        """Return one keyed auxiliary record without synthesizing the full run ledger."""
        _require_auxiliary_store(store)
        raw = _read_json_record(self._mg, store, key, scope=self._scope)
        if raw is not None:
            return raw
        _manifest, _head, legacy_payload = selected_run_ledger_manifest_with_head(self._mg, scope=self._scope)
        return _legacy_auxiliary_json(legacy_payload, store, key)

    def _list_auxiliary(self, store: KeyedJsonTreeStore) -> tuple[dict[str, object], ...]:
        """Return keyed auxiliary records without synthesizing unrelated stores."""
        _require_auxiliary_store(store)
        rows = _list_json_records(self._mg, store, scope=self._scope)
        if rows:
            return rows
        _manifest, _head, legacy_payload = selected_run_ledger_manifest_with_head(self._mg, scope=self._scope)
        return _legacy_records_for_store(legacy_payload, store)

    def get_args(self, args_ref: str) -> dict[str, object] | None:
        """Return one persisted run-argument payload by exact args ref."""
        return self._get_auxiliary(_RUN_ARGS, args_ref)

    def get_flow(self, flow_id: str) -> dict[str, object] | None:
        """Return one workflow metadata record by exact flow id."""
        return self._get_auxiliary(_FLOWS, flow_id)

    def list_flows(self) -> tuple[dict[str, object], ...]:
        """Return workflow metadata records visible in the selected run ledger."""
        return self._list_auxiliary(_FLOWS)

    def list_flow_runs(self, *, flow_id: str | None = None) -> tuple[dict[str, object], ...]:
        """Return workflow/run edge records, optionally filtered to one flow."""
        rows = self._list_auxiliary(_FLOW_RUNS)
        if flow_id is None:
            return rows
        return tuple(row for row in rows if row.get("flow_id") == flow_id)


def publish_run_record(
    mg: Any,
    record: RunRecord,
    *,
    scope: Any = None,
    args_payload: Mapping[str, object] | None = None,
    flow_run_payload: Mapping[str, object] | None = None,
) -> str:
    """Append or replace one run record in the selected ``shepherd.runs`` ledger."""
    return RunLedgerStore(mg, scope=scope).put_current(
        record,
        args_payload=args_payload,
        flow_run_payload=flow_run_payload,
    )


def _publish_run_record_impl(
    mg: Any,
    record: RunRecord,
    *,
    scope: Any,
    args_payload: Mapping[str, object] | None = None,
    flow_run_payload: Mapping[str, object] | None = None,
) -> str:
    target_scope = scope
    manifest, expected_head, legacy_payload = selected_run_ledger_manifest_with_head(mg, scope=target_scope)
    existing = _read_run_record_json(mg, record.run_ref, scope=target_scope)
    puts = _legacy_puts(legacy_payload)
    if args_payload is not None:
        _validate_run_args_payload(args_payload)
        _validate_run_args_record(record, args_payload)
        _upsert_put(puts, _put(_RUN_ARGS, _required_payload_id(args_payload, "args_ref"), args_payload))
    if flow_run_payload is not None:
        _validate_flow_run_payload(flow_run_payload)
        if _required_payload_id(flow_run_payload, "run_ref") != record.run_ref:
            raise RunLedgerPublishError("flow run payload run_ref disagrees with run record")
        _upsert_put(puts, _put(_FLOW_RUNS, record.run_ref, flow_run_payload))

    _upsert_put(puts, _put(_RUN_RECORDS, record.run_ref, record.to_json()))
    record_count = _manifest_count(manifest, legacy_payload=legacy_payload)
    if existing is None and not _legacy_has_run(legacy_payload, record.run_ref):
        record_count += 1
    manifest = _run_ledger_manifest(record_count=record_count, latest_run_ref=record.run_ref)
    content = KeyedJsonTreeDraft(manifest=manifest, base_head=expected_head, puts=tuple(puts))
    outcome = _exec_publish(mg, target_scope, payload=manifest, content=content, expected_head=expected_head)
    if not outcome.oids:
        raise RunLedgerPublishError("run-ledger publish produced no revision oid")
    return outcome.oids[0]


def publish_record(mg: Any, record: RunRecord, *, scope: Any = None) -> str:
    """Append or replace one run record in the selected ``shepherd.runs`` ledger."""
    return publish_run_record(mg, record, scope=scope)


def publish_terminal_run_record(mg: Any, record: RunRecord, *, scope: Any = None) -> RunRecord:
    """Publish a terminal row without adding a self-referential finish revision."""
    target_scope = mg.ground if scope is None else scope
    record = merge_current_run_state(mg, record, scope=target_scope)
    publish_run_record(mg, record, scope=target_scope)
    return record


def publish_terminal(mg: Any, record: RunRecord, *, scope: Any = None) -> RunRecord:
    """Publish a terminal run record through the shared run-ledger service."""
    return publish_terminal_run_record(mg, record, scope=scope)


def publish_flow_record(mg: Any, flow: Mapping[str, object], *, scope: Any = None) -> str:
    """Append or replace one workflow metadata record in the selected run ledger."""
    flow_id = _required_payload_id(flow, "flow_id")
    return RunLedgerStore(mg, scope=scope).put_flow(flow_id, flow)


def publish_flow_run_record(mg: Any, flow_run: Mapping[str, object], *, scope: Any = None) -> str:
    """Append or replace one workflow/run edge record in the selected run ledger."""
    return RunLedgerStore(mg, scope=scope).put_flow_run(_required_payload_id(flow_run, "run_ref"), flow_run)


def append_resolution(mg: Any, run_ref: str, resolution: Any, *, scope: Any = None) -> str:
    """Append one task-resolution record to an existing run row."""
    return RunLedgerStore(mg, scope=scope).append_resolution(run_ref, resolution)


def append_execution(mg: Any, run_ref: str, execution: Any, *, scope: Any = None) -> str:
    """Append one task-execution record to an existing run row."""
    return RunLedgerStore(mg, scope=scope).append_execution(run_ref, execution)


def merge_current_run_state(mg: Any, record: RunRecord, *, scope: Any = None) -> RunRecord:
    """Preserve append-only child records already accumulated on an existing row."""
    current = RunLedgerStore(mg, scope=scope).get(record.run_ref)
    if current is None:
        return record
    return replace(
        record,
        task_resolutions=_merge_by_attr(current.task_resolutions, record.task_resolutions, "resolution_id"),
        task_executions=_merge_by_attr(current.task_executions, record.task_executions, "execution_id"),
        pending_effects=current.pending_effects or record.pending_effects,
    )


def _update_run_record_impl(
    mg: Any,
    run_ref: str,
    update: Any,
    *,
    scope: Any = None,
    missing_message: str,
) -> str:
    from shepherd_dialect.workspace_control.schemas import RunRecord

    target_scope = mg.ground if scope is None else scope
    manifest, expected_head, legacy_payload = selected_run_ledger_manifest_with_head(mg, scope=target_scope)
    raw = _read_run_record_json(mg, run_ref, scope=target_scope)
    if raw is None:
        raw = _legacy_run_json(legacy_payload, run_ref)
    if raw is None:
        raise RunLedgerPublishError(missing_message)
    record = update(RunRecord.from_json(raw))
    puts = _legacy_puts(legacy_payload)
    _upsert_put(puts, _put(_RUN_RECORDS, run_ref, record.to_json()))
    manifest = _run_ledger_manifest(
        record_count=_manifest_count(manifest, legacy_payload=legacy_payload),
        latest_run_ref=_manifest_latest(manifest),
    )
    content = KeyedJsonTreeDraft(manifest=manifest, base_head=expected_head, puts=tuple(puts))
    outcome = _exec_publish(mg, target_scope, payload=manifest, content=content, expected_head=expected_head)
    if not outcome.oids:
        raise RunLedgerPublishError("run-ledger publish produced no revision oid")
    return outcome.oids[0]


def _exec_publish(
    mg: Any,
    scope: Any,
    *,
    payload: JsonObject,
    content: KeyedJsonTreeDraft,
    expected_head: str | None,
) -> Any:
    return mg.exec(
        RUN_LEDGER_BINDING,
        "publish",
        scope=scope,
        payload=payload,
        content=content,
        expected_head=expected_head,
        authority=mint_ledger_write_authority(),
    )


def _publish_auxiliary_record_impl(
    mg: Any,
    store: KeyedJsonTreeStore,
    key: str,
    payload: Mapping[str, object],
    *,
    scope: Any = None,
) -> str:
    _require_auxiliary_store(store)
    target_scope = mg.ground if scope is None else scope
    manifest, expected_head, legacy_payload = selected_run_ledger_manifest_with_head(mg, scope=target_scope)
    puts = _legacy_puts(legacy_payload)
    _upsert_put(puts, _put(store, key, payload))
    manifest = _run_ledger_manifest(
        record_count=_manifest_count(manifest, legacy_payload=legacy_payload),
        latest_run_ref=_manifest_latest(manifest),
    )
    content = KeyedJsonTreeDraft(manifest=manifest, base_head=expected_head, puts=tuple(puts))
    outcome = _exec_publish(mg, target_scope, payload=manifest, content=content, expected_head=expected_head)
    if not outcome.oids:
        raise RunLedgerPublishError("run-ledger publish produced no revision oid")
    return outcome.oids[0]


def _put(store: KeyedJsonTreeStore, key: str, payload: Mapping[str, object]) -> KeyedJsonPut:
    return KeyedJsonPut(key=key, path=store.path_for_key(key), payload=dict(payload))


def _upsert_put(puts: list[KeyedJsonPut], item: KeyedJsonPut) -> None:
    puts[:] = [existing for existing in puts if existing.path != item.path]
    puts.append(item)


def _read_run_record_json(mg: Any, run_ref: str, *, scope: Any = None) -> dict[str, object] | None:
    return _read_json_record(mg, _RUN_RECORDS, run_ref, scope=scope)


def _read_json_record(
    mg: Any,
    store: KeyedJsonTreeStore,
    key: str,
    *,
    scope: Any = None,
) -> dict[str, object] | None:
    reader = getattr(mg, "read_selected_binding_json_entry", None)
    if not callable(reader):
        return None
    return reader(RUN_LEDGER_BINDING, store.entry_path_for_key(key), scope=scope)


def _list_json_records(mg: Any, store: KeyedJsonTreeStore, *, scope: Any = None) -> tuple[dict[str, object], ...]:
    reader = getattr(mg, "read_selected_binding_json_entries", None)
    if not callable(reader):
        return ()
    return tuple(value for _path, value in reader(RUN_LEDGER_BINDING, store.prefix_path(), scope=scope))


def _json_record_map(rows: tuple[dict[str, object], ...], key_field: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for row in rows:
        key = row.get(key_field)
        if isinstance(key, str) and key:
            result[key] = dict(row)
    return result


def _legacy_puts(legacy_payload: Mapping[str, object] | None) -> list[KeyedJsonPut]:
    if legacy_payload is None:
        return []
    payload = run_ledger_payload(legacy_payload)
    puts: list[KeyedJsonPut] = []
    for raw in payload.get("runs", []):
        if isinstance(raw, Mapping):
            _upsert_put(puts, _put(_RUN_RECORDS, _required_payload_id(raw, "run_ref"), raw))
    for raw in _legacy_map_values(payload, "args"):
        _upsert_put(puts, _put(_RUN_ARGS, _required_payload_id(raw, "args_ref"), raw))
    for raw in _legacy_map_values(payload, "flows"):
        _upsert_put(puts, _put(_FLOWS, _required_payload_id(raw, "flow_id"), raw))
    for raw in _legacy_map_values(payload, "flow_runs"):
        _upsert_put(puts, _put(_FLOW_RUNS, _required_payload_id(raw, "run_ref"), raw))
    return puts


def _legacy_auxiliary_json(
    legacy_payload: Mapping[str, object] | None,
    store: KeyedJsonTreeStore,
    key: str,
) -> dict[str, object] | None:
    key_field = _key_field_for_store(store)
    for row in _legacy_records_for_store(legacy_payload, store):
        if row.get(key_field) == key:
            return row
    return None


def _legacy_records_for_store(
    legacy_payload: Mapping[str, object] | None,
    store: KeyedJsonTreeStore,
) -> tuple[dict[str, object], ...]:
    if legacy_payload is None:
        return ()
    payload = run_ledger_payload(legacy_payload)
    if store == _RUN_RECORDS:
        raw_runs = payload.get("runs", [])
        if not isinstance(raw_runs, list | tuple):
            return ()
        return tuple(dict(raw) for raw in raw_runs if isinstance(raw, Mapping))
    field_name = _legacy_field_for_store(store)
    return tuple(dict(raw) for raw in _legacy_map_values(payload, field_name))


def _legacy_field_for_store(store: KeyedJsonTreeStore) -> str:
    _require_auxiliary_store(store)
    if store == _RUN_ARGS:
        return "args"
    if store == _FLOWS:
        return "flows"
    if store == _FLOW_RUNS:
        return "flow_runs"
    raise RunLedgerPublishError(f"unsupported run-ledger auxiliary store: {store.record_root!r}")


def _key_field_for_store(store: KeyedJsonTreeStore) -> str:
    if store == _RUN_RECORDS:
        return "run_ref"
    _require_auxiliary_store(store)
    if store == _FLOW_RUNS:
        return "run_ref"
    if store == _RUN_ARGS:
        return "args_ref"
    if store == _FLOWS:
        return "flow_id"
    raise RunLedgerPublishError(f"unsupported run-ledger auxiliary store: {store.record_root!r}")


def _require_auxiliary_store(store: KeyedJsonTreeStore) -> None:
    if store not in _AUXILIARY_STORES:
        raise RunLedgerPublishError(f"unsupported run-ledger auxiliary store: {store.record_root!r}")


def _legacy_map_values(payload: Mapping[str, object], field_name: str) -> tuple[Mapping[str, object], ...]:
    raw = payload.get(field_name, {})
    if not isinstance(raw, Mapping):
        return ()
    return tuple(value for value in raw.values() if isinstance(value, Mapping))


def _legacy_has_run(legacy_payload: Mapping[str, object] | None, run_ref: str) -> bool:
    return _legacy_run_json(legacy_payload, run_ref) is not None


def _legacy_run_json(legacy_payload: Mapping[str, object] | None, run_ref: str) -> dict[str, object] | None:
    if legacy_payload is None:
        return None
    for raw in run_ledger_payload(legacy_payload).get("runs", []):
        if isinstance(raw, Mapping) and raw.get("run_ref") == run_ref:
            return dict(raw)
    return None


def _run_ledger_manifest(*, record_count: int, latest_run_ref: str | None) -> JsonObject:
    return {
        "schema": RUN_LEDGER_SCHEMA,
        "storage_shape": RUN_LEDGER_STORAGE_SHAPE,
        "record_count": record_count,
        "latest_run_ref": latest_run_ref,
    }


def _run_ledger_manifest_from_payload(payload: Mapping[str, object]) -> JsonObject:
    if payload.get("schema") != RUN_LEDGER_SCHEMA:
        raise RunLedgerPublishError(f"unsupported ledger schema: {payload.get('schema')!r}")
    if payload.get("storage_shape") != RUN_LEDGER_STORAGE_SHAPE:
        raise RunLedgerPublishError(f"unsupported run ledger storage shape: {payload.get('storage_shape')!r}")
    raw_count = payload.get("record_count", 0)
    if not isinstance(raw_count, int) or raw_count < 0:
        raise RunLedgerPublishError("run ledger manifest record_count must be a non-negative integer")
    latest = payload.get("latest_run_ref")
    if latest is not None and (not isinstance(latest, str) or not latest):
        raise RunLedgerPublishError("run ledger manifest latest_run_ref must be null or a non-empty string")
    return _run_ledger_manifest(record_count=raw_count, latest_run_ref=latest)


def _manifest_count(manifest: Mapping[str, object], *, legacy_payload: Mapping[str, object] | None) -> int:
    if legacy_payload is not None:
        return len(run_ledger_payload(legacy_payload)["runs"])
    raw = manifest.get("record_count", 0)
    if not isinstance(raw, int) or raw < 0:
        raise RunLedgerPublishError("run ledger manifest record_count must be a non-negative integer")
    return raw


def _manifest_latest(manifest: Mapping[str, object]) -> str | None:
    latest = manifest.get("latest_run_ref")
    if latest is None:
        return None
    if not isinstance(latest, str) or not latest:
        raise RunLedgerPublishError("run ledger manifest latest_run_ref must be null or a non-empty string")
    return latest


def _latest_run_ref_from_payload(payload: Mapping[str, object]) -> str | None:
    runs = payload.get("runs", [])
    if not isinstance(runs, list | tuple) or not runs:
        return None
    last = runs[-1]
    if isinstance(last, Mapping):
        raw = last.get("run_ref")
        if isinstance(raw, str) and raw:
            return raw
    return None


def _validate_run_args_record(record: RunRecord, args_payload: Mapping[str, object]) -> None:
    args_ref = args_payload.get("args_ref")
    if not isinstance(args_ref, str) or not args_ref:
        raise RunLedgerPublishError("run args payload requires args_ref")
    if record.args_ref is not None and record.args_ref != args_ref:
        raise RunLedgerPublishError("run args payload args_ref disagrees with run record")
    if args_payload.get("run_ref") != record.run_ref:
        raise RunLedgerPublishError("run args payload run_ref disagrees with run record")
    if args_payload.get("args_digest") != record.args_digest:
        raise RunLedgerPublishError("run args payload args_digest disagrees with run record")


def _required_payload_id(payload: Mapping[str, object], field_name: str) -> str:
    raw = payload.get(field_name)
    if not isinstance(raw, str) or not raw:
        raise RunLedgerPublishError(f"payload field {field_name!r} must be a non-empty string")
    return raw


def _append_by_attr(existing: tuple[Any, ...], item: Any, attr: str) -> tuple[Any, ...]:
    if any(getattr(candidate, attr) == getattr(item, attr) for candidate in existing):
        return existing
    return (*existing, item)


def _merge_by_attr(left: tuple[Any, ...], right: tuple[Any, ...], attr: str) -> tuple[Any, ...]:
    merged = left
    seen = {getattr(item, attr) for item in merged}
    for item in right:
        key = getattr(item, attr)
        if key in seen:
            continue
        merged = (*merged, item)
        seen.add(key)
    return merged


def selected_run_ledger_payload_with_head(mg: Any, *, scope: Any = None) -> tuple[JsonObject, str | None]:
    """Read the selected run-ledger payload plus the selected-head identity."""
    manifest, head, legacy_payload = selected_run_ledger_manifest_with_head(mg, scope=scope)
    if legacy_payload is not None:
        return run_ledger_payload(legacy_payload), head
    payload = dict(manifest)
    payload["runs"] = list(_list_json_records(mg, _RUN_RECORDS, scope=scope))
    payload["args"] = _json_record_map(_list_json_records(mg, _RUN_ARGS, scope=scope), "args_ref")
    payload["flows"] = _json_record_map(_list_json_records(mg, _FLOWS, scope=scope), "flow_id")
    payload["flow_runs"] = _json_record_map(_list_json_records(mg, _FLOW_RUNS, scope=scope), "run_ref")
    return run_ledger_payload(payload), head


def selected_run_ledger_manifest_with_head(
    mg: Any,
    *,
    scope: Any = None,
) -> tuple[JsonObject, str | None, Mapping[str, object] | None]:
    """Read the selected run-ledger manifest, selected head, and optional legacy payload."""
    target_scope = scope
    reader = getattr(mg, "read_selected_binding_revision_with_head", None)
    if callable(reader):
        selected = reader(RUN_LEDGER_BINDING, scope=target_scope)
        if selected is None:
            return _run_ledger_manifest(record_count=0, latest_run_ref=None), None, None
        if selected.payload.get("storage_shape") == RUN_LEDGER_STORAGE_SHAPE:
            return _run_ledger_manifest_from_payload(selected.payload), selected.head, None
        return (
            _run_ledger_manifest(
                record_count=len(run_ledger_payload(selected.payload)["runs"]),
                latest_run_ref=_latest_run_ref_from_payload(run_ledger_payload(selected.payload)),
            ),
            selected.head,
            selected.payload,
        )
    payload = mg.read_selected_binding_revision(RUN_LEDGER_BINDING, scope=target_scope)
    head = None
    if payload is None:
        return _run_ledger_manifest(record_count=0, latest_run_ref=None), head, None
    if payload.get("storage_shape") == RUN_LEDGER_STORAGE_SHAPE:
        return _run_ledger_manifest_from_payload(payload), head, None
    normalized = run_ledger_payload(payload)
    return (
        _run_ledger_manifest(
            record_count=len(normalized["runs"]),
            latest_run_ref=_latest_run_ref_from_payload(normalized),
        ),
        head,
        payload,
    )


def run_ledger_payload(payload: Mapping[str, object] | None) -> JsonObject:
    """Normalize an optional selected run-ledger payload into a mutable payload."""
    if payload is None:
        return {"schema": RUN_LEDGER_SCHEMA, "runs": []}
    if payload.get("schema") != RUN_LEDGER_SCHEMA:
        raise RunLedgerPublishError(f"unsupported ledger schema: {payload.get('schema')!r}")
    if payload.get("storage_shape") == RUN_LEDGER_STORAGE_SHAPE:
        normalized_manifest = _run_ledger_manifest_from_payload(payload)
        normalized_manifest.setdefault("runs", [])
        normalized_manifest.setdefault("args", {})
        normalized_manifest.setdefault("flows", {})
        normalized_manifest.setdefault("flow_runs", {})
        return normalized_manifest
    runs = payload.get("runs", [])
    if not isinstance(runs, list | tuple):
        raise RunLedgerPublishError("run ledger payload field 'runs' must be a list")
    normalized = dict(payload)
    normalized["schema"] = RUN_LEDGER_SCHEMA
    normalized["runs"] = list(runs)
    _normalize_object_map(normalized, "args", validator=_validate_run_args_payload)
    _normalize_object_map(normalized, "flows", validator=_validate_flow_payload)
    _normalize_object_map(normalized, "flow_runs", validator=_validate_flow_run_payload)
    return normalized


def _validate_run_args_payload(args_payload: Mapping[str, object]) -> None:
    if args_payload.get("schema") != RUN_ARGS_SCHEMA:
        raise RunLedgerPublishError(f"unsupported run args schema: {args_payload.get('schema')!r}")
    args_ref = _required_payload_id(args_payload, "args_ref")
    run_ref = _required_payload_id(args_payload, "run_ref")
    args_digest = _required_payload_id(args_payload, "args_digest")
    if args_payload.get("payload_digest") != args_digest:
        raise RunLedgerPublishError("run args payload_digest must equal args_digest")
    payload = args_payload.get("payload")
    if not isinstance(payload, Mapping):
        raise RunLedgerPublishError("run args payload field 'payload' must be an object")
    if canonical_digest(payload) != args_digest:
        raise RunLedgerPublishError("run args payload digest disagrees with canonical payload")
    if run_args_ref(run_ref=run_ref, args_digest=args_digest) != args_ref:
        raise RunLedgerPublishError("run args payload args_ref disagrees with run_ref and args_digest")
    input_refs = args_payload.get("input_refs", [])
    if not isinstance(input_refs, list | tuple):
        raise RunLedgerPublishError("run args payload input_refs must be a list")


def _validate_flow_payload(flow: Mapping[str, object]) -> None:
    if flow.get("schema") != FLOW_SCHEMA:
        raise RunLedgerPublishError(f"unsupported flow schema: {flow.get('schema')!r}")
    _required_payload_id(flow, "flow_id")
    _required_payload_id(flow, "name")
    metadata = flow.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise RunLedgerPublishError("flow metadata must be an object")


def _validate_flow_run_payload(flow_run: Mapping[str, object]) -> None:
    if flow_run.get("schema") != FLOW_RUN_SCHEMA:
        raise RunLedgerPublishError(f"unsupported flow run schema: {flow_run.get('schema')!r}")
    _required_payload_id(flow_run, "flow_id")
    _required_payload_id(flow_run, "run_ref")
    _required_payload_id(flow_run, "name")
    sequence = flow_run.get("sequence")
    if not isinstance(sequence, int) or sequence < 0:
        raise RunLedgerPublishError("flow run sequence must be a non-negative integer")
    after = flow_run.get("after", [])
    if not isinstance(after, list | tuple) or any(not isinstance(item, str) or not item for item in after):
        raise RunLedgerPublishError("flow run after must be a list of non-empty run refs")
    metadata = flow_run.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise RunLedgerPublishError("flow run metadata must be an object")


def _normalize_object_map(payload: JsonObject, field_name: str, *, validator: Any) -> None:
    raw = payload.get(field_name, {})
    if not isinstance(raw, Mapping):
        raise RunLedgerPublishError(f"run ledger payload field {field_name!r} must be an object")
    normalized: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise RunLedgerPublishError(f"run ledger payload field {field_name!r} keys must be non-empty strings")
        if not isinstance(value, Mapping):
            raise RunLedgerPublishError(f"run ledger payload field {field_name!r} values must be objects")
        validator(value)
        normalized[key] = dict(value)
    if normalized:
        payload[field_name] = normalized
