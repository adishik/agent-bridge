"""SQLite persistence for hub-wide account settings only."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import sqlite3


Clock = Callable[[], datetime]
_USAGE_CREDITS_ACKNOWLEDGED_AT = "usage_credits_acknowledged_at"
_MIGRATION_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS schema_metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS account_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
)


class HubStore:
    """A connection-owned boundary for non-project account settings."""

    def __init__(self, path: Path, *, clock: Clock) -> None:
        self._path = path
        self._clock = clock
        self._closed = False
        self._connection = sqlite3.connect(str(path), isolation_level=None)
        self._connection.execute("PRAGMA foreign_keys = ON")
        try:
            os.chmod(path, 0o600)
            self._migrate()
        except BaseException:
            self._connection.close()
            self._closed = True
            raise

    def usage_credits_acknowledged(self) -> bool:
        """Return false unless a valid acknowledgement timestamp is persisted."""
        row = self._connection.execute(
            "SELECT value FROM account_settings WHERE key = ?",
            (_USAGE_CREDITS_ACKNOWLEDGED_AT,),
        ).fetchone()
        if row is None:
            return False
        try:
            acknowledged_at = datetime.fromisoformat(row[0])
        except (TypeError, ValueError):
            return False
        return acknowledged_at.tzinfo is not None

    def acknowledge_usage_credits(self) -> None:
        """Persist the first account-level usage-credit acknowledgement."""
        with self._immediate_transaction():
            existing = self._connection.execute(
                "SELECT 1 FROM account_settings WHERE key = ?",
                (_USAGE_CREDITS_ACKNOWLEDGED_AT,),
            ).fetchone()
            if existing is not None:
                return
            acknowledged_at = self._clock()
            if not isinstance(acknowledged_at, datetime):
                raise ValueError("clock must return a datetime")
            if acknowledged_at.utcoffset() is None:
                raise ValueError("clock must return a timezone-aware datetime")
            self._connection.execute(
                "INSERT INTO account_settings (key, value) VALUES (?, ?)",
                (_USAGE_CREDITS_ACKNOWLEDGED_AT, acknowledged_at.isoformat()),
            )

    def close(self) -> None:
        """Release the owned connection; repeated calls are safe."""
        if self._closed:
            return
        self._connection.close()
        self._closed = True

    def _migrate(self) -> None:
        self._connection.execute("BEGIN")
        try:
            for statement in _MIGRATION_STATEMENTS:
                self._connection.execute(statement)
            self._connection.execute(
                "INSERT OR IGNORE INTO schema_metadata (key, value) VALUES (?, ?)",
                ("schema_version", "1"),
            )
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    @contextmanager
    def _immediate_transaction(self) -> Iterator[None]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()
