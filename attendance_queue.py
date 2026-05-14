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
                updated_at     TEXT NOT NULL,
                UNIQUE(student_id, scope_key)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sc_scope "
            "ON student_cache(scope_key)"
        )

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
            else:
                try:
                    conn.execute("ALTER TABLE attendance_queue ADD COLUMN environment TEXT NOT NULL DEFAULT ''")
                except Exception:
                    pass  # Column already exists (SQLite doesn't abort the txn)
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

    def _rebuild_dedup_from_db(self):
        today = datetime.now().strftime("%d-%b-%Y")
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
                              date_str: str, environment: str = "") -> tuple:
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
                 status, attempts, created_at, updated_at, next_retry_at, environment)
            VALUES (?, ?, ?, 'PENDING', 0, ?, ?, ?, ?)
        """)
        if self._is_postgres:
            sql += " RETURNING id"
        with self._db() as conn:
            cur = conn.execute(sql, (student_id, student_name, date_str, now, now, now, environment))
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

    def get_today_attendance(self, date_str: str) -> list:
        """Return all attendance records for date_str that are not FAILED."""
        with self._db() as conn:
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
        today = datetime.now().strftime("%d-%b-%Y")
        with self._db() as conn:
            if student_id:
                cur = conn.execute(
                    self._q("DELETE FROM attendance_queue WHERE date_str=? AND student_id=?"),
                    (today, student_id),
                )
            else:
                cur = conn.execute(
                    self._q("DELETE FROM attendance_queue WHERE date_str=?"),
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
                    INSERT INTO student_cache (student_id, scope_key, name, student_number, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """), (s["id"], scope_key, s.get("name", ""), s.get("student_number", ""), now))
        logger.info(f"Saved {len(students)} students to local DB for scope '{scope_key}'.")

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

    def clear_student_scope(self, scope_key: str) -> int:
        """Remove all student metadata for a scope (called on manual refresh)."""
        with self._db() as conn:
            cur = conn.execute(
                self._q("DELETE FROM student_cache WHERE scope_key=?"), (scope_key,)
            )
            count = cur.rowcount
        logger.info(f"Cleared {count} students from local DB for scope '{scope_key}'.")
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
                    "WHERE student_id=? AND source LIKE 'verified_%'"
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
                    "SELECT id, student_id, student_name, date_str, attempts, environment "
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

            name        = row["student_name"]
            student_id  = row["student_id"]
            attempts    = row["attempts"]
            environment = row["environment"]
            try:
                result = self._zoho.post_attendance(
                    student_id=student_id,
                    student_name=name,
                    verification_type="face_blink_verified",
                    env=environment,
                )
                if result.get("success"):
                    self._set_posted(rec_id)
                    logger.info(f"Queue: synced {name} → Zoho (#{rec_id})")
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
