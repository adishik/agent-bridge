from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3

import pytest

from agent_bridge.hub_store import HubStore


def _clock() -> datetime:
    return datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def _tables(path: Path) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        return {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()


def test_hub_database_owns_only_schema_metadata_and_account_settings(tmp_path) -> None:
    path = tmp_path / "hub.sqlite3"
    store = HubStore(path, clock=_clock)

    assert _tables(path) == {"schema_metadata", "account_settings"}
    assert not {
        "sessions",
        "tasks",
        "events",
        "agent_runs",
        "baselines",
        "repositories",
        "repository_paths",
    } & _tables(path)

    store.close()


def test_usage_credits_default_to_unacknowledged(tmp_path) -> None:
    store = HubStore(tmp_path / "hub.sqlite3", clock=_clock)

    assert store.usage_credits_acknowledged() is False

    store.close()


def test_usage_credits_acknowledgement_is_durable_and_uses_injected_clock(tmp_path) -> None:
    path = tmp_path / "hub.sqlite3"
    store = HubStore(path, clock=_clock)

    store.acknowledge_usage_credits()
    store.close()

    connection = sqlite3.connect(path)
    try:
        stored_at = connection.execute(
            "SELECT value FROM account_settings WHERE key = ?",
            ("usage_credits_acknowledged_at",),
        ).fetchone()
    finally:
        connection.close()

    assert stored_at == ("2026-08-11T12:00:00+00:00",)
    reopened = HubStore(path, clock=_clock)
    assert reopened.usage_credits_acknowledged() is True
    reopened.close()


def test_usage_credits_acknowledgement_is_idempotent(tmp_path) -> None:
    path = tmp_path / "hub.sqlite3"
    timestamps = iter(
        (
            datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
            datetime(2026, 8, 11, 13, 0, tzinfo=timezone.utc),
        )
    )
    store = HubStore(path, clock=lambda: next(timestamps))

    store.acknowledge_usage_credits()
    store.acknowledge_usage_credits()

    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT key, value FROM account_settings WHERE key = ?",
            ("usage_credits_acknowledged_at",),
        ).fetchall()
    finally:
        connection.close()
    assert rows == [("usage_credits_acknowledged_at", "2026-08-11T12:00:00+00:00")]
    store.close()


def test_failed_acknowledgement_transaction_rolls_back(tmp_path) -> None:
    path = tmp_path / "hub.sqlite3"
    store = HubStore(path, clock=_clock)
    store._connection.execute(
        """
        CREATE TRIGGER reject_usage_acknowledgement
        BEFORE INSERT ON account_settings
        WHEN NEW.key = 'usage_credits_acknowledged_at'
        BEGIN
            SELECT RAISE(ABORT, 'forced acknowledgement failure');
        END
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="forced acknowledgement failure"):
        store.acknowledge_usage_credits()

    assert store._connection.in_transaction is False
    assert store.usage_credits_acknowledged() is False
    store._connection.execute("DROP TRIGGER reject_usage_acknowledgement")
    store.acknowledge_usage_credits()
    assert store.usage_credits_acknowledged() is True
    store.close()


def test_naive_clock_value_rolls_back_and_connection_remains_usable(tmp_path) -> None:
    timestamps = iter(
        (
            datetime(2026, 8, 11, 12, 0),
            datetime(2026, 8, 11, 13, 0, tzinfo=timezone.utc),
        )
    )
    store = HubStore(tmp_path / "hub.sqlite3", clock=lambda: next(timestamps))

    with pytest.raises(ValueError, match="timezone-aware"):
        store.acknowledge_usage_credits()

    assert store._connection.in_transaction is False
    assert store.usage_credits_acknowledged() is False
    store.acknowledge_usage_credits()
    assert store.usage_credits_acknowledged() is True
    store.close()


@pytest.mark.parametrize("malformed_value", ["", "false", "not-a-timestamp"])
def test_malformed_acknowledgement_values_fail_closed(tmp_path, malformed_value: str) -> None:
    path = tmp_path / "hub.sqlite3"
    store = HubStore(path, clock=_clock)
    store._connection.execute(
        "INSERT INTO account_settings (key, value) VALUES (?, ?)",
        ("usage_credits_acknowledged_at", malformed_value),
    )

    assert store.usage_credits_acknowledged() is False
    store.close()


def test_hub_database_file_is_private_to_its_owner(tmp_path) -> None:
    path = tmp_path / "hub.sqlite3"
    store = HubStore(path, clock=_clock)

    assert os.stat(path).st_mode & 0o777 == 0o600

    store.close()


def test_close_is_idempotent(tmp_path) -> None:
    store = HubStore(tmp_path / "hub.sqlite3", clock=_clock)

    store.close()
    store.close()
