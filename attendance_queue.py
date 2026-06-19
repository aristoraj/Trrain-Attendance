"""
SQLite / PostgreSQL attendance outbox with async Zoho sync.

Flow per verify request:
  1. Face matched → is_already_marked() check (in-memory O(1), then DB <1ms)
  2. enqueue() → write to DB instantly → return success to student
  3. Background worker drains PENDING rows → posts to Zoho → marks POSTED
  4. On Zoho failure: exponential backoff retry (5s, 15s, 45s, 135s, 405s)
  5. After 5 failed attempts: mark FAILED — visible at /admin/sync-status

Also manages the face_embeddings table:
  - Multi-source: 'enrollment' photo + up to 3 'verified_N' live captures per student
  - SQLite by default; PostgreSQL when DATABASE_URL env var is set
"""

import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get(
    "ATTENDANCE_DB_PATH",
    os.path.join(os.path.dirname(__file__), "data", "attendance_queue.db"),
)
DATABASE_URL = os.environ.get("DATABASE_URL")   # Render managed PostgreSQL
MAX_ATTEMPTS = 5
WORKER_POLL_INTERVAL = 2


# ── Thin connection wrapper — uniform interface for sqlite3 and psycopg2 ──────

class _ConnWrapper:
    """
    Wraps a raw sqlite3 or psycopg2 connection so that conn.execute(sql, params)
    works identically for both backends.
    """

    def __init__(self, raw_conn, is_postgres: bool):
        self._raw = raw_conn
        self._pg = is_postgres
        self._cur = raw_conn.cursor() if is_postgres else None

    def execute(self, sql: str, params=()):
        if self._pg:
            self._cur.execute(sql, params)
            return self._cur
        else:
            return self._raw.execute(sql, params)

    def commit(self):
        self._raw.commit()

    def rollback(self):
        self._raw.rollback()

    def close(self):
        if self._pg and self._cur:
            try:
                self._cur.close()
            except Exception:
                pass
        try:
            self._raw.close()
        except Exception:
            pass


class AttendanceQueue:
    """
    Thread-safe DB outbox + background Zoho sync worker.

    Supports SQLite (default, single instance) and PostgreSQL (multi-instance,
    set DATABASE_URL env var). The face_embeddings table stores multiple
    angle-variant embeddings per student for better accuracy.
    """

    def __init__(self, zoho_api):
        self._zoho = zoho_api
        self._lock = threading.Lock()

        self._is_postgres = bool(DATABASE_URL)
        # SQL placeholder: ? for SQLite, %s for PostgreSQL
        self._ph = "%s" if self._is_postgres else "?"

        # In-memory fast-path dedup {date_str: set_of_student_ids}
        self._global_marked: dict[str, set] = {}

        # Optional callback fired after worker posts a record and gets zoho_id.
        # Signature: on_record_posted(student_id, student_name, zoho_id, env)
        self.on_record_posted = None


        if self._is_postgres:
            logger.info("AttendanceQueue: using PostgreSQL (DATABASE_URL set).")
        else:
            logger.info(f"AttendanceQueue: using SQLite at {DB_PATH}.")

        self._init_db()
        self._rebuild_dedup_from_db()

        self._worker = threading.Thread(target=self._drain_loop, daemon=True)
        self._worker.start()
        logger.info("AttendanceQueue ready — background sync worker started.")

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _q(self, sql: str) -> str:
        """Convert ? placeholders to %s for PostgreSQL."""
        if self._is_postgres:
            return sql.replace("?", "%s")
        return sql

    @contextmanager
    def _db(self):
        if self._is_postgres:
            import psycopg2
            import psycopg2.extras
            # Render provides postgres:// URLs; psycopg2 needs postgresql://
            dsn = DATABASE_URL.replace("postgres://", "postgresql://", 1)
            raw = psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            import sqlite3
            raw = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
            raw.row_factory = sqlite3.Row

        conn = _ConnWrapper(raw, self._is_postgres)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _set_wal_mode(self):
        """Enable WAL journal mode for safe concurrent writes (SQLite only)."""
        if self._is_postgres:
            return   # PostgreSQL handles concurrency natively
        import sqlite3
        for attempt in range(20):
            raw = None
            try:
                raw = sqlite3.connect(DB_PATH, timeout=2)
                row = raw.execute("PRAGMA journal_mode=WAL").fetchone()
                raw.close()
                if row and row[0] == "wal":
                    return
                break
            except sqlite3.OperationalError:
                if raw:
                    try:
                        raw.close()
                    except Exception:
                        pass
                time.sleep(0.15 * (attempt + 1))
        else:
            logger.warning("Could not set WAL journal mode — falling back to default.")

    def _table_exists(self, conn, table_name: str) -> bool:
        if self._is_postgres:
            row = conn.execute(
                self._q("SELECT 1 FROM information_schema.tables WHERE table_name=?"),
                (table_name,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            ).fetchone()
        return row is not None

    def _migrate_embeddings_schema(self, conn):
        """Migrate face_embeddings v1 (single PRIMARY KEY) → v2 (multi-source)."""
        # Use a savepoint on PostgreSQL so a migration error doesn't abort the
        # outer _init_db() transaction (InFailedSqlTransaction cascade).
        if self._is_postgres:
            conn.execute("SAVEPOINT migrate_embeddings")
        try:
            if not self._table_exists(conn, "face_embeddings"):
                if self._is_postgres:
                    conn.execute("RELEASE SAVEPOINT migrate_embeddings")
                return   # fresh install — no migration needed

            if self._is_postgres:
                row = conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name='face_embeddings' AND column_name='source'"
                ).fetchone()
                has_source = row is not None
            else:
                info = conn.execute("PRAGMA table_info(face_embeddings)").fetchall()
                has_source = any(r["name"] == "source" for r in info)

            if has_source:
                if self._is_postgres:
                    conn.execute("RELEASE SAVEPOINT migrate_embeddings")
                return   # already v2 — nothing to do

            logger.info("Migrating face_embeddings to multi-source schema...")
            conn.execute("ALTER TABLE face_embeddings RENAME TO face_embeddings_v1")
            self._create_embeddings_table(conn)
            conn.execute(self._q(
                "INSERT INTO face_embeddings (student_id, source, embedding, updated_at) "
                "SELECT student_id, 'enrollment', embedding, updated_at FROM face_embeddings_v1"
            ))
            conn.execute("DROP TABLE face_embeddings_v1")
            if self._is_postgres:
                conn.execute("RELEASE SAVEPOINT migrate_embeddings")
            logger.info("face_embeddings migration complete.")
        except Exception as e:
            if self._is_postgres:
                try:
                    conn.execute("ROLLBACK TO SAVEPOINT migrate_embeddings")
                    conn.execute("RELEASE SAVEPOINT migrate_embeddings")
                except Exception:
                    pass
            logger.warning(f"face_embeddings migration skipped: {e}")

    def _create_student_cache_table(self, conn):
        serial = "BIGSERIAL" if self._is_postgres else "INTEGER"
        autoincrement = "" if self._is_postgres else "AUTOINCREMENT"
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS student_cache (
                id             {serial} PRIMARY KEY {autoincrement},
                student_id     TEXT NOT NULL,
                scope_key      TEXT NOT NULL,
                name           TEXT NOT NULL DEFAULT '',
                student_number TEXT NOT NULL DEFAULT '',
                has_embedding  BOOLEAN NOT NULL DEFAULT FALSE,
                updated_at     TEXT NOT NULL,
                UNIQUE(student_id, scope_key)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sc_scope "
            "ON student_cache(scope_key)"
        )
        # Migrate existing rows to add has_embedding and batch_id columns
        for col, definition in [
            ("has_embedding", "BOOLEAN NOT NULL DEFAULT FALSE"),
            ("batch_id",      "TEXT    NOT NULL DEFAULT ''"),
        ]:
            if self._is_postgres:
                conn.execute(f"SAVEPOINT add_{col}")
                try:
                    conn.execute(f"ALTER TABLE student_cache ADD COLUMN {col} {definition}")
                    conn.execute(f"RELEASE SAVEPOINT add_{col}")
                except Exception:
                    conn.execute(f"ROLLBACK TO SAVEPOINT add_{col}")
                    conn.execute(f"RELEASE SAVEPOINT add_{col}")
            else:
                try:
                    conn.execute(f"ALTER TABLE student_cache ADD COLUMN {col} {definition}")
                except Exception:
                    pass  # Already exists
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sc_batch "
            "ON student_cache(batch_id, scope_key)"
        )

    def _create_webhook_sync_log_table(self, conn):
        """
        Tracks every feature-enable / feature-disable webhook call.
        Used for admin visibility and startup recovery of interrupted syncs.
        """
        serial = "BIGSERIAL" if self._is_postgres else "INTEGER"
        autoincrement = "" if self._is_postgres else "AUTOINCREMENT"
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS webhook_sync_log (
                id          {serial} PRIMARY KEY {autoincrement},
                event       TEXT NOT NULL,
                email       TEXT NOT NULL,
                centre_id   TEXT NOT NULL,
                scope_key   TEXT NOT NULL,
                env         TEXT NOT NULL DEFAULT '',
                status      TEXT NOT NULL DEFAULT 'pending',
                error_msg   TEXT,
                started_at  TEXT NOT NULL,
                finished_at TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_wsl_centre "
            "ON webhook_sync_log(centre_id, env)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_wsl_status "
            "ON webhook_sync_log(status)"
        )

    def _create_checkin_state_table(self, conn):
        """
        Tracks per-student per-day check-in/check-out state.
        Purely additive — existing attendance_queue flow is unchanged.
        """
        serial = "BIGSERIAL" if self._is_postgres else "INTEGER"
        autoincrement = "" if self._is_postgres else "AUTOINCREMENT"
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS checkin_state (
                id             {serial} PRIMARY KEY {autoincrement},
                student_id     TEXT NOT NULL,
                date_str       TEXT NOT NULL,
                checkin_at     TEXT NOT NULL,
                is_checked_out INTEGER NOT NULL DEFAULT 0,
                checkout_at    TEXT,
                environment    TEXT NOT NULL DEFAULT '',
                zoho_record_id TEXT NOT NULL DEFAULT '',
                UNIQUE(student_id, date_str)
            )
        """)
        # Migration: add zoho_record_id to existing tables
        if self._is_postgres:
            try:
                conn.execute("SAVEPOINT add_cs_zoho_id")
                conn.execute("ALTER TABLE checkin_state ADD COLUMN zoho_record_id TEXT NOT NULL DEFAULT ''")
                conn.execute("RELEASE SAVEPOINT add_cs_zoho_id")
            except Exception:
                conn.execute("ROLLBACK TO SAVEPOINT add_cs_zoho_id")
                conn.execute("RELEASE SAVEPOINT add_cs_zoho_id")
        else:
            try:
                conn.execute("ALTER TABLE checkin_state ADD COLUMN zoho_record_id TEXT NOT NULL DEFAULT ''")
            except Exception:
                pass
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cs_student_date "
            "ON checkin_state(student_id, date_str)"
        )

    def get_checkin_status(self, student_id: str, date_str: str) -> dict:
        """
        Returns check-in state for a student on a given date.
        Return dict: {status: 'none'|'checked_in'|'checked_out', checkin_at?, checkout_at?}
        """
        with self._db() as conn:
            row = conn.execute(
                self._q("SELECT * FROM checkin_state WHERE student_id=? AND date_str=?"),
                (student_id, date_str),
            ).fetchone()
        if not row:
            return {"status": "none"}
        zoho_record_id = row["zoho_record_id"] if "zoho_record_id" in row.keys() else ""
        if row["is_checked_out"]:
            return {
                "status":        "checked_out",
                "checkin_at":    row["checkin_at"],
                "checkout_at":   row["checkout_at"],
                "zoho_record_id": zoho_record_id,
            }
        return {"status": "checked_in", "checkin_at": row["checkin_at"], "zoho_record_id": zoho_record_id}

    def record_checkin(self, student_id: str, student_name: str,
                       date_str: str, environment: str = "",
                       zoho_record_id: str = "") -> bool:
        """
        Records a check-in. Returns True if inserted, False if already exists.
        Called from both the SDK path (/api/record-checkin) and server fallback (/api/post-attendance).
        """
        now = datetime.now().isoformat()
        try:
            with self._db() as conn:
                conn.execute(
                    self._q("""
                        INSERT INTO checkin_state (student_id, date_str, checkin_at, environment, zoho_record_id)
                        VALUES (?, ?, ?, ?, ?)
                    """),
                    (student_id, date_str, now, environment, zoho_record_id),
                )
            logger.info(f"Check-in recorded: {student_id} on {date_str} zoho_id='{zoho_record_id}'")
            return True
        except Exception:
            return False  # UNIQUE constraint = already recorded (idempotent)

    def update_zoho_record_id(self, student_id: str, date_str: str, zoho_record_id: str) -> None:
        """Store the Zoho Creator record ID in checkin_state after server-side queue posts attendance."""
        if not zoho_record_id or zoho_record_id == "unknown":
            return
        with self._db() as conn:
            conn.execute(
                self._q("UPDATE checkin_state SET zoho_record_id=? WHERE student_id=? AND date_str=?"),
                (zoho_record_id, student_id, date_str),
            )
        logger.info(f"Stored Zoho record ID {zoho_record_id} for {student_id} on {date_str}")

    def record_checkout(self, student_id: str, date_str: str) -> bool:
        """
        Marks the student as checked out. Returns True if updated, False if already checked out.
        """
        now = datetime.now().isoformat()
        with self._db() as conn:
            cur = conn.execute(
                self._q("""
                    UPDATE checkin_state SET is_checked_out=1, checkout_at=?
                    WHERE student_id=? AND date_str=? AND is_checked_out=0
                """),
                (now, student_id, date_str),
            )
        return cur.rowcount > 0

    def _create_batch_status_table(self, conn):
        """
        Tracks the last-known status, start date, and end date of every batch
        that has been scanned for a given scope. Used to detect when a batch
        transitions out of Ongoing so its students can be removed automatically.
        """
        conn.execute("""
            CREATE TABLE IF NOT EXISTS batch_status (
                batch_id    TEXT NOT NULL,
                scope_key   TEXT NOT NULL,
                batch_name  TEXT NOT NULL DEFAULT '',
                status      TEXT NOT NULL DEFAULT 'Ongoing',
                start_date  TEXT NOT NULL DEFAULT '',
                end_date    TEXT NOT NULL DEFAULT '',
                updated_at  TEXT NOT NULL,
                PRIMARY KEY (batch_id, scope_key)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bs_scope "
            "ON batch_status(scope_key)"
        )

    def save_batch_statuses(self, scope_key: str, batches: list) -> None:
        """
        Persist the current list of Ongoing batches for a scope.
        Each item in `batches`: {id, name, status, start_date, end_date}
        """
        now = datetime.now().isoformat()
        with self._db() as conn:
            for b in batches:
                if self._is_postgres:
                    conn.execute(self._q("""
                        INSERT INTO batch_status (batch_id, scope_key, batch_name, status, start_date, end_date, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(batch_id, scope_key) DO UPDATE
                            SET batch_name=excluded.batch_name,
                                status=excluded.status,
                                start_date=excluded.start_date,
                                end_date=excluded.end_date,
                                updated_at=excluded.updated_at
                    """), (b["id"], scope_key, b.get("name",""), b.get("status","Ongoing"),
                           b.get("start_date",""), b.get("end_date",""), now))
                else:
                    conn.execute(self._q("""
                        INSERT OR REPLACE INTO batch_status
                            (batch_id, scope_key, batch_name, status, start_date, end_date, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """), (b["id"], scope_key, b.get("name",""), b.get("status","Ongoing"),
                           b.get("start_date",""), b.get("end_date",""), now))

    def get_known_batch_ids(self, scope_key: str) -> list[str]:
        """Return batch IDs previously recorded as Ongoing for this scope."""
        with self._db() as conn:
            rows = conn.execute(
                self._q("SELECT batch_id FROM batch_status WHERE scope_key=?"),
                (scope_key,)
            ).fetchall()
        return [r["batch_id"] for r in rows]

    def remove_students_by_batch(self, batch_id: str, scope_key: str) -> int:
        """
        Remove all student_cache rows for a batch that is no longer Ongoing.
        The face_embeddings rows are retained (in case the student re-enrols)
        but they become unreachable for attendance matching without a student_cache row.
        """
        with self._db() as conn:
            cur = conn.execute(
                self._q("DELETE FROM student_cache WHERE batch_id=? AND scope_key=?"),
                (batch_id, scope_key)
            )
            count = cur.rowcount if hasattr(cur, "rowcount") else 0
        if count:
            logger.info(f"Removed {count} student(s) for completed batch {batch_id} in scope '{scope_key}'.")
        return count

    def remove_batch_status(self, batch_id: str, scope_key: str) -> None:
        """Remove the batch_status row once students have been cleaned up."""
        with self._db() as conn:
            conn.execute(
                self._q("DELETE FROM batch_status WHERE batch_id=? AND scope_key=?"),
                (batch_id, scope_key)
            )

    # ── Webhook sync log ──────────────────────────────────────────────────────

    def log_webhook_sync(self, event: str, email: str, centre_id: str,
                         scope_key: str, env: str = "") -> int:
        """Insert a new webhook_sync_log row with status='running'. Returns the row id."""
        now = datetime.now().isoformat()
        sql = self._q("""
            INSERT INTO webhook_sync_log
                (event, email, centre_id, scope_key, env, status, started_at)
            VALUES (?, ?, ?, ?, ?, 'running', ?)
        """)
        if self._is_postgres:
            sql += " RETURNING id"
        with self._db() as conn:
            cur = conn.execute(sql, (event, email, centre_id, scope_key, env, now))
            return cur.fetchone()["id"] if self._is_postgres else cur.lastrowid

    def update_webhook_sync_status(self, log_id: int, status: str,
                                   error_msg: str = None) -> None:
        """Update status and finished_at on an existing webhook_sync_log row."""
        now = datetime.now().isoformat()
        with self._db() as conn:
            conn.execute(
                self._q(
                    "UPDATE webhook_sync_log "
                    "SET status=?, finished_at=?, error_msg=? WHERE id=?"
                ),
                (status, now, error_msg, log_id),
            )

    def get_incomplete_syncs(self) -> list:
        """
        Return webhook_sync_log rows still in 'running' or 'deleting' state.
        Called at startup to re-trigger any sync that was interrupted by a restart.
        """
        with self._db() as conn:
            rows = conn.execute(
                "SELECT id, event, email, centre_id, scope_key, env "
                "FROM webhook_sync_log WHERE status IN ('running', 'deleting') "
                "ORDER BY started_at ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_scope_keys_overlapping_centres(self, centre_ids: list, env: str = "") -> list:
        """
        Return all scope_keys in student_cache whose centre ID set has ANY
        overlap with the given centre_ids.

        Needed for the disable webhook: the User Management record may have
        had centres removed before the webhook fires, so the payload's
        centre_ids can be a subset of what was originally used to build the
        scope key in the DB. Scanning by overlap rather than exact match
        ensures all scopes are found and cleaned up.
        """
        target_ids = set(str(c) for c in centre_ids)
        with self._db() as conn:
            rows = conn.execute(
                "SELECT DISTINCT scope_key FROM student_cache"
            ).fetchall()
        matches = []
        for row in rows:
            sk = row["scope_key"]
            # Scope key format: "{env}:C:{id1},{id2},..." or "C:{id1},{id2},..."
            try:
                c_part = sk.split(":C:", 1)[1] if ":C:" in sk else sk.split("C:", 1)[1]
                sk_ids = set(c_part.split(","))
                if sk_ids & target_ids:   # any overlap → this scope belongs to the user
                    matches.append(sk)
            except (IndexError, ValueError):
                continue
        return matches

    def get_webhook_sync_log(self, limit: int = 20) -> list:
        """Return recent webhook_sync_log entries for the admin panel."""
        with self._db() as conn:
            rows = conn.execute(
                self._q(
                    "SELECT id, event, email, centre_id, env, status, "
                    "error_msg, started_at, finished_at "
                    "FROM webhook_sync_log ORDER BY id DESC LIMIT ?"
                ),
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_center_data(self, scope_key: str) -> dict:
        """
        Hard-delete all student_cache and batch_status rows for a scope_key.
        Also removes orphaned face_embeddings (students not present in any
        other scope's student_cache after this deletion).
        Returns counts for logging.

        Uses set-based SQL rather than a per-student Python loop so the entire
        operation runs as a short transaction — avoids holding a long SQLite
        write lock that would block concurrent requests.
        """
        with self._db() as conn:
            cur = conn.execute(
                self._q("DELETE FROM student_cache WHERE scope_key=?"),
                (scope_key,),
            )
            deleted_students = cur.rowcount if hasattr(cur, "rowcount") else 0

            cur = conn.execute(
                self._q("DELETE FROM batch_status WHERE scope_key=?"),
                (scope_key,),
            )
            deleted_batches = cur.rowcount if hasattr(cur, "rowcount") else 0

            # Remove embeddings for students that no longer appear in ANY scope.
            # Single subquery — no Python loop, no per-row round-trips.
            cur = conn.execute(
                "DELETE FROM face_embeddings WHERE student_id NOT IN "
                "(SELECT DISTINCT student_id FROM student_cache)"
            )
            deleted_embeddings = cur.rowcount if hasattr(cur, "rowcount") else 0

        return {
            "deleted_students":   deleted_students,
            "deleted_batches":    deleted_batches,
            "deleted_embeddings": deleted_embeddings,
        }

    def _create_daily_caches_table(self, conn):
        """
        Single key-value store for daily-TTL caches (centres, batches, feature-access).
        Survives server restarts so a single Zoho API call per day is enough.
        """
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_cache (
                cache_key  TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

    _DAILY_CACHE_TTL = 86400   # 24 hours in seconds

    def get_daily_cache(self, key: str):
        """Return cached value (parsed JSON) if fresh, else None.
        Keys prefixed with 'catalogued:' never expire — they persist until
        explicitly cleared (e.g. /admin/clear-daily-cache).
        """
        with self._db() as conn:
            row = conn.execute(
                self._q("SELECT value_json, updated_at FROM daily_cache WHERE cache_key=?"),
                (key,)
            ).fetchone()
        if not row:
            return None
        try:
            # 'catalogued:' keys never expire — only cleared by admin action
            if not key.startswith("catalogued:"):
                ts = datetime.fromisoformat(row["updated_at"]).timestamp()
                if (datetime.now().timestamp() - ts) > self._DAILY_CACHE_TTL:
                    return None
            import json as _json
            return _json.loads(row["value_json"])
        except Exception:
            return None

    def set_daily_cache(self, key: str, value) -> None:
        """Store value (serialised to JSON) with current timestamp."""
        import json as _json
        now = datetime.now().isoformat()
        val = _json.dumps(value)
        with self._db() as conn:
            if self._is_postgres:
                conn.execute(self._q("""
                    INSERT INTO daily_cache (cache_key, value_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE
                        SET value_json=excluded.value_json,
                            updated_at=excluded.updated_at
                """), (key, val, now))
            else:
                conn.execute(self._q("""
                    INSERT OR REPLACE INTO daily_cache (cache_key, value_json, updated_at)
                    VALUES (?, ?, ?)
                """), (key, val, now))

    def clear_daily_cache(self, key_prefix: str = "") -> int:
        """Delete daily cache entries matching prefix (empty = all)."""
        with self._db() as conn:
            if key_prefix:
                cur = conn.execute(
                    self._q("DELETE FROM daily_cache WHERE cache_key LIKE ?"),
                    (key_prefix + "%",)
                )
            else:
                cur = conn.execute("DELETE FROM daily_cache")
            return cur.rowcount if hasattr(cur, "rowcount") else 0

    def _create_embeddings_table(self, conn):
        serial = "BIGSERIAL" if self._is_postgres else "INTEGER"
        autoincrement = "" if self._is_postgres else "AUTOINCREMENT"
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id          {serial} PRIMARY KEY {autoincrement},
                student_id  TEXT    NOT NULL,
                source      TEXT    NOT NULL DEFAULT 'enrollment',
                embedding   TEXT    NOT NULL,
                det_score   REAL,
                photo_url   TEXT,
                updated_at  TEXT    NOT NULL,
                UNIQUE(student_id, source)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_emb_student "
            "ON face_embeddings(student_id)"
        )

    def _migrate_photo_url_column(self, conn):
        """Add photo_url column to face_embeddings if it doesn't exist yet."""
        if self._is_postgres:
            conn.execute("SAVEPOINT add_photo_url_col")
            try:
                conn.execute("ALTER TABLE face_embeddings ADD COLUMN photo_url TEXT")
                conn.execute("RELEASE SAVEPOINT add_photo_url_col")
                logger.info("face_embeddings: added photo_url column.")
            except Exception:
                conn.execute("ROLLBACK TO SAVEPOINT add_photo_url_col")
                conn.execute("RELEASE SAVEPOINT add_photo_url_col")
        else:
            try:
                conn.execute("ALTER TABLE face_embeddings ADD COLUMN photo_url TEXT")
                logger.info("face_embeddings: added photo_url column.")
            except Exception:
                pass  # Already exists

    def _init_db(self):
        self._set_wal_mode()
        serial = "BIGSERIAL" if self._is_postgres else "INTEGER"
        autoincrement = "" if self._is_postgres else "AUTOINCREMENT"
        with self._db() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS attendance_queue (
                    id            {serial} PRIMARY KEY {autoincrement},
                    student_id    TEXT    NOT NULL,
                    student_name  TEXT    NOT NULL,
                    date_str      TEXT    NOT NULL,
                    status        TEXT    NOT NULL DEFAULT 'PENDING',
                    attempts      INTEGER NOT NULL DEFAULT 0,
                    last_error    TEXT,
                    created_at    TEXT    NOT NULL,
                    updated_at    TEXT    NOT NULL,
                    next_retry_at TEXT    NOT NULL,
                    environment   TEXT    NOT NULL DEFAULT ''
                )
            """)
            # Migrate existing tables that pre-date the environment column.
            # On PostgreSQL a failed ALTER aborts the whole transaction, so
            # wrap it in a savepoint so the outer _init_db() transaction survives.
            if self._is_postgres:
                try:
                    conn.execute("SAVEPOINT add_env_col")
                    conn.execute("ALTER TABLE attendance_queue ADD COLUMN environment TEXT NOT NULL DEFAULT ''")
                    conn.execute("RELEASE SAVEPOINT add_env_col")
                except Exception:
                    conn.execute("ROLLBACK TO SAVEPOINT add_env_col")
                    conn.execute("RELEASE SAVEPOINT add_env_col")
                try:
                    conn.execute("SAVEPOINT add_dsid_col")
                    conn.execute("ALTER TABLE attendance_queue ADD COLUMN device_session_id TEXT NOT NULL DEFAULT ''")
                    conn.execute("RELEASE SAVEPOINT add_dsid_col")
                except Exception:
                    conn.execute("ROLLBACK TO SAVEPOINT add_dsid_col")
                    conn.execute("RELEASE SAVEPOINT add_dsid_col")
                try:
                    conn.execute("SAVEPOINT add_action_col")
                    conn.execute("ALTER TABLE attendance_queue ADD COLUMN action_field TEXT NOT NULL DEFAULT ''")
                    conn.execute("RELEASE SAVEPOINT add_action_col")
                except Exception:
                    conn.execute("ROLLBACK TO SAVEPOINT add_action_col")
                    conn.execute("RELEASE SAVEPOINT add_action_col")
                try:
                    conn.execute("SAVEPOINT add_checkin_time_col")
                    conn.execute("ALTER TABLE attendance_queue ADD COLUMN checkin_time TEXT NOT NULL DEFAULT ''")
                    conn.execute("RELEASE SAVEPOINT add_checkin_time_col")
                except Exception:
                    conn.execute("ROLLBACK TO SAVEPOINT add_checkin_time_col")
                    conn.execute("RELEASE SAVEPOINT add_checkin_time_col")
            else:
                try:
                    conn.execute("ALTER TABLE attendance_queue ADD COLUMN environment TEXT NOT NULL DEFAULT ''")
                except Exception:
                    pass  # Column already exists
                try:
                    conn.execute("ALTER TABLE attendance_queue ADD COLUMN device_session_id TEXT NOT NULL DEFAULT ''")
                except Exception:
                    pass
                try:
                    conn.execute("ALTER TABLE attendance_queue ADD COLUMN action_field TEXT NOT NULL DEFAULT ''")
                except Exception:
                    pass
                try:
                    conn.execute("ALTER TABLE attendance_queue ADD COLUMN checkin_time TEXT NOT NULL DEFAULT ''")
                except Exception:
                    pass
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_status_retry "
                "ON attendance_queue(status, next_retry_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_student_date "
                "ON attendance_queue(student_id, date_str)"
            )
            self._migrate_embeddings_schema(conn)
            self._create_embeddings_table(conn)
            self._migrate_photo_url_column(conn)
            self._create_student_cache_table(conn)
            self._create_daily_caches_table(conn)
            self._create_batch_status_table(conn)
            self._create_webhook_sync_log_table(conn)
            self._create_checkin_state_table(conn)

    def _rebuild_dedup_from_db(self):
        today = datetime.now(_IST).strftime("%d-%b-%Y")
        with self._db() as conn:
            rows = conn.execute(
                self._q(
                    "SELECT student_id FROM attendance_queue "
                    "WHERE date_str=? AND status IN ('PENDING','PROCESSING','POSTED','SDK_POSTED')"
                ),
                (today,),
            ).fetchall()
        with self._lock:
            for row in rows:
                self._mark_in_memory(row["student_id"], today)
        logger.info(f"Dedup set: {len(rows)} students already marked today.")

    # ── Public API ────────────────────────────────────────────────────────────

    def is_already_marked(self, student_id: str, date_str: str) -> bool:
        with self._lock:
            if student_id in self._global_marked.get(date_str, set()):
                return True

        with self._db() as conn:
            count = conn.execute(
                self._q(
                    "SELECT COUNT(*) AS cnt FROM attendance_queue "
                    "WHERE student_id=? AND date_str=? "
                    "AND status IN ('PENDING','PROCESSING','POSTED','SDK_POSTED')"
                ),
                (student_id, date_str),
            ).fetchone()["cnt"]

        if count > 0:
            with self._lock:
                self._mark_in_memory(student_id, date_str)
            return True
        return False

    def mark_attended(self, student_id: str, student_name: str, date_str: str) -> None:
        """Mark attendance posted via Zoho SDK — writes DB record for dedup durability."""
        now = datetime.now().isoformat()
        sql = self._q("""
            INSERT INTO attendance_queue
                (student_id, student_name, date_str,
                 status, attempts, created_at, updated_at, next_retry_at, environment)
            VALUES (?, ?, ?, 'SDK_POSTED', 0, ?, ?, ?, 'sdk')
        """)
        with self._db() as conn:
            conn.execute(sql, (student_id, student_name, date_str, now, now, now))
        with self._lock:
            self._mark_in_memory(student_id, date_str)
        logger.info(f"SDK attendance marked for {student_name} ({date_str})")

    def enqueue_if_not_marked(self, student_id: str, student_name: str,
                              date_str: str, environment: str = "",
                              device_session_id: str = "",
                              action_field: str = "",
                              checkin_time: str = "") -> tuple:
        """
        Atomic dedup-check + enqueue in one call.
        Returns (queue_id, is_duplicate).

        Holding self._lock while marking in-memory blocks any concurrent
        in-process request from passing the dedup check before this one
        has written to the DB, closing the TOCTOU window.
        The subsequent DB COUNT check covers cross-process duplicates
        (e.g. multiple Render instances sharing the same PostgreSQL).
        """
        with self._lock:
            if student_id in self._global_marked.get(date_str, set()):
                return None, True
            # Claim the slot in memory immediately so concurrent threads
            # hitting this block next will see it as already marked.
            self._mark_in_memory(student_id, date_str)

        # Cross-process durability check (DB is authoritative across instances)
        with self._db() as conn:
            count = conn.execute(
                self._q(
                    "SELECT COUNT(*) AS cnt FROM attendance_queue "
                    "WHERE student_id=? AND date_str=? "
                    "AND status IN ('PENDING','PROCESSING','POSTED','SDK_POSTED')"
                ),
                (student_id, date_str),
            ).fetchone()["cnt"]

        if count > 0:
            logger.info(f"DB dedup blocked duplicate for {student_name} ({date_str})")
            return None, True

        now = datetime.now().isoformat()
        sql = self._q("""
            INSERT INTO attendance_queue
                (student_id, student_name, date_str,
                 status, attempts, created_at, updated_at, next_retry_at,
                 environment, device_session_id, action_field, checkin_time)
            VALUES (?, ?, ?, 'PENDING', 0, ?, ?, ?, ?, ?, ?, ?)
        """)
        if self._is_postgres:
            sql += " RETURNING id"
        with self._db() as conn:
            cur = conn.execute(sql, (
                student_id, student_name, date_str, now, now, now,
                environment, device_session_id, action_field, checkin_time,
            ))
            rec_id = cur.fetchone()["id"] if self._is_postgres else cur.lastrowid

        logger.info(f"Queued attendance for {student_name} (queue #{rec_id})")
        return rec_id, False

    def enqueue(self, student_id: str, student_name: str, date_str: str,
                environment: str = "") -> int:
        """Legacy single-step enqueue (no dedup). Use enqueue_if_not_marked for verify flow."""
        now = datetime.now().isoformat()
        sql = self._q("""
            INSERT INTO attendance_queue
                (student_id, student_name, date_str,
                 status, attempts, created_at, updated_at, next_retry_at, environment)
            VALUES (?, ?, ?, 'PENDING', 0, ?, ?, ?, ?)
        """)
        if self._is_postgres:
            sql += " RETURNING id"
        with self._db() as conn:
            cur = conn.execute(sql, (student_id, student_name, date_str, now, now, now, environment))
            rec_id = cur.fetchone()["id"] if self._is_postgres else cur.lastrowid
        with self._lock:
            self._mark_in_memory(student_id, date_str)
        logger.info(f"Queued attendance for {student_name} (queue #{rec_id})")
        return rec_id

    def get_status_summary(self) -> dict:
        since = (datetime.now() - timedelta(days=1)).strftime("%d-%b-%Y")
        with self._db() as conn:
            rows = conn.execute(
                self._q(
                    "SELECT status, COUNT(*) as cnt FROM attendance_queue "
                    "WHERE date_str >= ? GROUP BY status"
                ),
                (since,),
            ).fetchall()
            counts = {row["status"]: row["cnt"] for row in rows}

            failed = conn.execute(
                "SELECT id, student_name, date_str, attempts, last_error, created_at "
                "FROM attendance_queue WHERE status='FAILED' "
                "ORDER BY created_at DESC LIMIT 50"
            ).fetchall()

            pending_old = conn.execute(
                self._q(
                    "SELECT id, student_name, date_str, attempts, created_at "
                    "FROM attendance_queue WHERE status='PENDING' "
                    "AND created_at < ? ORDER BY created_at ASC LIMIT 20"
                ),
                ((datetime.now() - timedelta(minutes=5)).isoformat(),),
            ).fetchall()

        return {
            "pending":        counts.get("PENDING", 0),
            "posted":         counts.get("POSTED",  0),
            "failed":         counts.get("FAILED",  0),
            "failed_records": [dict(r) for r in failed],
            "stuck_pending":  [dict(r) for r in pending_old],
        }

    def get_today_attendance(self, date_str: str, device_session_id: str = None) -> list:
        """
        Return today's attendance records that are not FAILED.
        If device_session_id is provided, only return records from that device session
        so users sharing the same login across multiple locations see only their own entries.
        """
        with self._db() as conn:
            if device_session_id:
                rows = conn.execute(
                    self._q(
                        "SELECT student_name, status, created_at FROM attendance_queue "
                        "WHERE date_str=? AND status NOT IN ('FAILED') "
                        "AND device_session_id=? "
                        "ORDER BY created_at ASC"
                    ),
                    (date_str, device_session_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    self._q(
                        "SELECT student_name, status, created_at FROM attendance_queue "
                        "WHERE date_str=? AND status NOT IN ('FAILED') "
                        "ORDER BY created_at ASC"
                    ),
                    (date_str,),
                ).fetchall()
        return [
            {
                "name":   row["student_name"],
                "status": row["status"],
                "time":   row["created_at"][11:16],  # HH:MM
            }
            for row in rows
        ]

    def clear_today_attendance(self, student_id: str = None) -> int:
        """
        Delete today's attendance records and clear the in-memory dedup set.
        Pass student_id to clear a single student; omit to clear everyone for today.
        Used for testing only — does NOT affect Zoho Creator (delete there separately).
        """
        today = datetime.now(_IST).strftime("%d-%b-%Y")
        with self._db() as conn:
            if student_id:
                cur = conn.execute(
                    self._q("DELETE FROM attendance_queue WHERE date_str=? AND student_id=?"),
                    (today, student_id),
                )
                conn.execute(
                    self._q("DELETE FROM checkin_state WHERE date_str=? AND student_id=?"),
                    (today, student_id),
                )
            else:
                cur = conn.execute(
                    self._q("DELETE FROM attendance_queue WHERE date_str=?"),
                    (today,),
                )
                conn.execute(
                    self._q("DELETE FROM checkin_state WHERE date_str=?"),
                    (today,),
                )
            count = cur.rowcount
        with self._lock:
            if student_id:
                self._global_marked.get(today, set()).discard(student_id)
            else:
                self._global_marked.pop(today, None)
        logger.info(f"Cleared {count} attendance record(s) for {today}" +
                    (f" (student {student_id})" if student_id else " (all students)"))
        return count

    def retry_failed(self) -> int:
        now = datetime.now().isoformat()
        with self._db() as conn:
            cur = conn.execute(
                self._q(
                    "UPDATE attendance_queue "
                    "SET status='PENDING', attempts=0, last_error=NULL, "
                    "    next_retry_at=?, updated_at=? "
                    "WHERE status='FAILED'"
                ),
                (now, now),
            )
            count = cur.rowcount
        logger.info(f"Reset {count} FAILED records to PENDING.")
        return count

    # ── Embedding cache ───────────────────────────────────────────────────────

    def get_local_embeddings(self, student_id: str) -> list:
        """Return all cached embeddings for a student [{source, embedding, det_score, photo_url}]."""
        with self._db() as conn:
            rows = conn.execute(
                self._q(
                    "SELECT source, embedding, det_score, photo_url FROM face_embeddings "
                    "WHERE student_id=? ORDER BY source"
                ),
                (student_id,),
            ).fetchall()
        return [
            {
                "source":    r["source"],
                "embedding": r["embedding"],
                "det_score": r["det_score"],
                "photo_url": r["photo_url"],
            }
            for r in rows
        ]

    def save_local_embedding(
        self,
        student_id: str,
        embedding_json: str,
        source: str = "enrollment",
        det_score: Optional[float] = None,
        photo_url: Optional[str] = None,
    ) -> None:
        """Upsert a JSON embedding for (student_id, source)."""
        now = datetime.now().isoformat()
        with self._db() as conn:
            conn.execute(
                self._q("""
                    INSERT INTO face_embeddings (student_id, source, embedding, det_score, photo_url, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(student_id, source) DO UPDATE
                        SET embedding=excluded.embedding,
                            det_score=excluded.det_score,
                            photo_url=excluded.photo_url,
                            updated_at=excluded.updated_at
                """),
                (student_id, source, embedding_json, det_score, photo_url, now),
            )

    def clear_enrollment_embeddings(self) -> int:
        """Delete ALL enrollment embeddings (used when no scope is known)."""
        with self._db() as conn:
            cur = conn.execute(
                "DELETE FROM face_embeddings WHERE source IN ('enrollment', 'no_photo')"
            )
            count = cur.rowcount
        logger.info(f"Cleared {count} enrollment/no_photo embeddings from local cache.")
        return count

    def clear_enrollment_embeddings_for_scope(self, scope_key: str) -> int:
        """
        Delete enrollment embeddings only for students in the given scope.
        Prevents a refresh of one centre from wiping another centre's embeddings.
        """
        with self._db() as conn:
            rows = conn.execute(
                self._q("SELECT student_id FROM student_cache WHERE scope_key=?"),
                (scope_key,),
            ).fetchall()
        student_ids = [r["student_id"] for r in rows]
        if not student_ids:
            return 0
        ph = ", ".join([self._ph] * len(student_ids))
        with self._db() as conn:
            cur = conn.execute(
                f"DELETE FROM face_embeddings "
                f"WHERE source IN ('enrollment', 'no_photo') AND student_id IN ({ph})",
                tuple(student_ids),
            )
            count = cur.rowcount
        logger.info(f"Cleared {count} enrollment embeddings for scope '{scope_key}'.")
        return count

    # ── Persistent student cache (Option A: survive restarts) ────────────────────

    def save_students_to_db(self, scope_key: str, students: list) -> None:
        """
        Persist student metadata for a scope key so cold starts can skip Zoho API calls.
        Embeddings are already stored in face_embeddings; this stores id/name/number only.
        """
        now = datetime.now().isoformat()
        with self._db() as conn:
            conn.execute(self._q("DELETE FROM student_cache WHERE scope_key=?"), (scope_key,))
            for s in students:
                conn.execute(self._q("""
                    INSERT INTO student_cache (student_id, scope_key, name, student_number, has_embedding, batch_id, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """), (s["id"], scope_key, s.get("name", ""), s.get("student_number", ""),
                       True, s.get("batch_id", ""), now))
        logger.info(f"Saved {len(students)} students to local DB for scope '{scope_key}'.")

    def save_no_photo_students(self, scope_key: str, students: list) -> None:
        """
        Persist students WITHOUT embeddings so we know they exist in the system
        but haven't uploaded a photo yet. Stored with has_embedding=False.
        No TTL — lives forever until a webhook encodes their photo.
        These rows are excluded from face matching but prevent unnecessary re-fetching.
        """
        if not students:
            return
        now = datetime.now().isoformat()
        with self._db() as conn:
            for s in students:
                conn.execute(self._q("""
                    INSERT INTO student_cache (student_id, scope_key, name, student_number, has_embedding, batch_id, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(student_id, scope_key) DO NOTHING
                """), (s["id"], scope_key, s.get("name", ""), s.get("student_number", ""),
                       False, s.get("batch_id", ""), now))
        logger.info(f"Stored {len(students)} no-photo students for scope '{scope_key}'.")

    def is_scope_fully_catalogued(self, scope_key: str) -> bool:
        """
        Returns True if we have ever done a full Zoho scan for this scope and saved
        all students (with and without photos). If True, subsequent preloads can skip
        the Zoho API call entirely — the webhook handles updates.
        """
        return self.get_daily_cache(f"catalogued:{scope_key}") is not None

    def mark_scope_catalogued(self, scope_key: str) -> None:
        """Mark that this scope has been fully scanned. No expiry — persists indefinitely."""
        import json as _json
        now = datetime.now().isoformat()
        key = f"catalogued:{scope_key}"
        with self._db() as conn:
            if self._is_postgres:
                conn.execute(self._q("""
                    INSERT INTO daily_cache (cache_key, value_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE
                        SET value_json=excluded.value_json,
                            updated_at=excluded.updated_at
                """), (key, _json.dumps(True), now))
            else:
                conn.execute(self._q("""
                    INSERT OR REPLACE INTO daily_cache (cache_key, value_json, updated_at)
                    VALUES (?, ?, ?)
                """), (key, _json.dumps(True), now))

    def load_students_from_db(self, scope_key: str) -> list | None:
        """
        Reconstruct student list from local DB (student_cache + face_embeddings).
        Returns list of dicts with raw_embeddings (JSON strings) for the caller to decode,
        or None if no data exists for this scope.
        """
        with self._db() as conn:
            rows = conn.execute(
                self._q("SELECT student_id, name, student_number FROM student_cache WHERE scope_key=?"),
                (scope_key,),
            ).fetchall()
        if not rows:
            return None
        result = []
        for row in rows:
            sid = row["student_id"]
            emb_rows = self.get_local_embeddings(sid)
            valid_embs = [
                e for e in emb_rows
                if e["source"] != "no_photo" and e["embedding"]
            ]
            if not valid_embs:
                continue
            result.append({
                "id":             sid,
                "name":           row["name"],
                "student_number": row["student_number"],
                "raw_embeddings": valid_embs,
            })
        return result if result else None

    def get_all_scope_keys(self) -> list:
        """Return all scope keys that have students stored in local DB."""
        with self._db() as conn:
            rows = conn.execute("SELECT DISTINCT scope_key FROM student_cache").fetchall()
        return [r["scope_key"] for r in rows]

    def upsert_student_in_scope(self, scope_key: str, student: dict) -> None:
        """Add or update a single student row in student_cache for the given scope key."""
        now = datetime.now().isoformat()
        with self._db() as conn:
            conn.execute(self._q("""
                INSERT INTO student_cache (student_id, scope_key, name, student_number, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(student_id, scope_key) DO UPDATE
                    SET name=excluded.name,
                        student_number=excluded.student_number,
                        updated_at=excluded.updated_at
            """), (student["id"], scope_key, student.get("name", ""), student.get("student_number", ""), now))
        logger.debug(f"Upserted student {student['id']} in scope '{scope_key}'.")

    def update_student_name_everywhere(self, student_id: str, name: str, student_number: str = "") -> int:
        """
        Update name and student_number for a student across ALL scopes in student_cache.
        Called by the webhook after encoding so the new name is persisted even when no
        active in-memory cache exists (0 scope(s) patched path).
        Returns the number of rows updated.
        """
        now = datetime.now().isoformat()
        with self._db() as conn:
            cur = conn.execute(self._q("""
                UPDATE student_cache
                SET name=?, student_number=?, updated_at=?
                WHERE student_id=?
            """), (name, student_number, now, student_id))
            count = cur.rowcount if hasattr(cur, "rowcount") else 0
        if count:
            logger.info(f"Updated name to '{name}' for student {student_id} across {count} scope(s).")
        return count

    def clear_student_scope(self, scope_key: str) -> int:
        """Remove all student metadata for a scope (called on manual refresh)."""
        with self._db() as conn:
            cur = conn.execute(
                self._q("DELETE FROM student_cache WHERE scope_key=?"), (scope_key,)
            )
            count = cur.rowcount
        logger.info(f"Cleared {count} students from local DB for scope '{scope_key}'.")
        return count

    def clear_all_embeddings_for_student(self, student_id: str) -> int:
        """Delete every face_embeddings row for a student (enrollment + no_photo + all verified_N)."""
        with self._db() as conn:
            cur = conn.execute(
                self._q("DELETE FROM face_embeddings WHERE student_id=?"),
                (student_id,),
            )
            count = cur.rowcount
        logger.info(f"Cleared all {count} embedding row(s) for student {student_id}")
        return count

    def clear_verified_embeddings(self, student_id: str) -> int:
        """
        Delete all verified_N live-capture embeddings for a student.
        Called when the enrollment photo changes so stale live captures
        from the previous person don't pollute the new identity.
        """
        with self._db() as conn:
            cur = conn.execute(
                self._q(
                    "DELETE FROM face_embeddings "
                    "WHERE student_id=? AND source IN ('verified_1','verified_2','verified_3')"
                ),
                (student_id,),
            )
            count = cur.rowcount
        if count:
            logger.info(f"Cleared {count} stale verified embedding(s) for student {student_id} (photo changed)")
        return count

    def add_verified_embedding(self, student_id: str, embedding_json: str) -> None:
        """
        Persist a live-capture embedding for future angle-variant matching.
        Rotates through verified_1 → verified_2 → verified_3, then wraps back to verified_1.
        Called after every successful attendance mark so the system self-improves.
        """
        with self._db() as conn:
            rows = conn.execute(
                self._q(
                    "SELECT source FROM face_embeddings "
                    "WHERE student_id=? AND source IN ('verified_1','verified_2','verified_3')"
                ),
                (student_id,),
            ).fetchall()
        existing = {r["source"] for r in rows}

        # Fill empty slot first
        for i in range(1, 4):
            slot = f"verified_{i}"
            if slot not in existing:
                self.save_local_embedding(student_id, embedding_json, source=slot)
                logger.debug(f"Saved live capture as {slot} for student {student_id}")
                return

        # All 3 full — rotate: overwrite verified_1 (oldest, by convention)
        self.save_local_embedding(student_id, embedding_json, source="verified_1")
        logger.debug(f"Rotated verified_1 embedding for student {student_id}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _mark_in_memory(self, student_id: str, date_str: str):
        """Must be called while holding self._lock."""
        if date_str not in self._global_marked:
            self._global_marked[date_str] = set()
            # Purge keys older than today to prevent unbounded growth
            for old_key in [k for k in self._global_marked if k != date_str]:
                del self._global_marked[old_key]
        self._global_marked[date_str].add(student_id)

    # ── Background drain loop ─────────────────────────────────────────────────

    def _drain_loop(self):
        consecutive_errors = 0
        while True:
            try:
                self._drain()
                consecutive_errors = 0
                time.sleep(WORKER_POLL_INTERVAL)
            except Exception as e:
                consecutive_errors += 1
                backoff = min(WORKER_POLL_INTERVAL * (2 ** consecutive_errors), 60)
                logger.error(f"Queue drain error (attempt {consecutive_errors}): {e} — retrying in {backoff}s")
                time.sleep(backoff)

    def _drain(self):
        now_iso = datetime.now().isoformat()
        stale_iso = (datetime.now() - timedelta(minutes=5)).isoformat()

        with self._db() as conn:
            conn.execute(
                self._q(
                    "UPDATE attendance_queue "
                    "SET status='PENDING', next_retry_at=?, updated_at=? "
                    "WHERE status='PROCESSING' AND updated_at < ?"
                ),
                (now_iso, now_iso, stale_iso),
            )

        with self._db() as conn:
            rows = conn.execute(
                self._q(
                    "SELECT id, student_id, student_name, date_str, attempts, environment, action_field, checkin_time "
                    "FROM attendance_queue "
                    "WHERE status='PENDING' AND next_retry_at <= ? "
                    "ORDER BY created_at ASC LIMIT 10"
                ),
                (now_iso,),
            ).fetchall()

        for row in rows:
            rec_id = row["id"]
            with self._db() as conn:
                cur = conn.execute(
                    self._q(
                        "UPDATE attendance_queue SET status='PROCESSING', updated_at=? "
                        "WHERE id=? AND status='PENDING'"
                    ),
                    (now_iso, rec_id),
                )
            if cur.rowcount == 0:
                continue

            name         = row["student_name"]
            student_id   = row["student_id"]
            attempts     = row["attempts"]
            environment  = row["environment"]
            action_field = row["action_field"]  if "action_field"  in row.keys() else ""
            checkin_time = row["checkin_time"]  if "checkin_time"  in row.keys() else ""
            try:
                result = self._zoho.post_attendance(
                    student_id=student_id,
                    student_name=name,
                    verification_type="face_blink_verified",
                    env=environment,
                    action_field=action_field,
                    checkin_time=checkin_time,
                )
                if result.get("success"):
                    self._set_posted(rec_id)
                    zoho_id = result.get("data", {}).get("data", {}).get("ID", "")
                    if zoho_id:
                        self.update_zoho_record_id(student_id, row["date_str"], zoho_id)
                        if callable(self.on_record_posted):
                            try:
                                self.on_record_posted(student_id, name, zoho_id, environment)
                            except Exception as _cb_err:
                                logger.warning(f"on_record_posted callback error: {_cb_err}")
                    logger.info(f"Queue: synced {name} → Zoho (#{rec_id}) zoho_id='{zoho_id}'")
                else:
                    self._handle_failure(rec_id, attempts, result.get("error", "Zoho returned failure"))
            except Exception as e:
                self._handle_failure(rec_id, attempts, str(e))

    def _set_posted(self, rec_id: int):
        now = datetime.now().isoformat()
        with self._db() as conn:
            conn.execute(
                self._q(
                    "UPDATE attendance_queue SET status='POSTED', updated_at=? "
                    "WHERE id=? AND status='PROCESSING'"
                ),
                (now, rec_id),
            )

    def _handle_failure(self, rec_id: int, attempts: int, error: str):
        now = datetime.now().isoformat()
        new_attempts = attempts + 1
        if new_attempts >= MAX_ATTEMPTS:
            with self._db() as conn:
                conn.execute(
                    self._q(
                        "UPDATE attendance_queue "
                        "SET status='FAILED', attempts=?, last_error=?, updated_at=? "
                        "WHERE id=? AND status='PROCESSING'"
                    ),
                    (new_attempts, error[:500], now, rec_id),
                )
            logger.error(f"Queue: #{rec_id} permanently FAILED after {MAX_ATTEMPTS} attempts: {error[:150]}")
        else:
            delay = 5 * (3 ** attempts)
            next_retry = (datetime.now() + timedelta(seconds=delay)).isoformat()
            with self._db() as conn:
                conn.execute(
                    self._q(
                        "UPDATE attendance_queue "
                        "SET status='PENDING', attempts=?, last_error=?, "
                        "    updated_at=?, next_retry_at=? "
                        "WHERE id=? AND status='PROCESSING'"
                    ),
                    (new_attempts, error[:500], now, next_retry, rec_id),
                )
            logger.warning(
                f"Queue: #{rec_id} attempt {new_attempts}/{MAX_ATTEMPTS}, "
                f"retry in {delay}s — {error[:100]}"
            )
