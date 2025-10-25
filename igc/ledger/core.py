import os
from typing import Dict, List, Literal, Optional, Tuple

import psycopg

Status = Literal["queued", "running", "written", "failed"]


# -------------------------------
# Connection helper (env-driven)
# -------------------------------
def _connect():
    """
    Uses PG* env vars or PGDSN if set.
    Required envs on the host:
      PGHOST, PGUSER, PGPASSWORD, PGDATABASE (and optionally PGPORT),
    or a full DSN in PGDSN.
    """
    dsn = os.getenv("PGDSN")
    if dsn:
        return psycopg.connect(dsn)
    return psycopg.connect()  # psycopg will read PG* env vars


# ============================================================
# 1) UNIFIED JOB LEDGER ACCESS  (full_job_ledger_extended_view)
# ============================================================
def fetch_job_ledger_record(
    *,
    job_id: int | None = None,
    sim_id: int | None = None,
    group_id: int | None = None,
    metric_id: int | None = None,
    step_id: int | None = None,
    frame: int | None = None,
) -> list[dict]:
    """
    Query full_job_ledger_extended_view using lowercase column names (job_id, sim_id, etc.)
    """
    sql = """
        SELECT * FROM full_job_ledger_extended_view
        WHERE (%(job_id)s::int   IS NULL OR job_id   = %(job_id)s::int)
          AND (%(sim_id)s::int   IS NULL OR sim_id   = %(sim_id)s::int)
          AND (%(group_id)s::int IS NULL OR group_id = %(group_id)s::int)
          AND (%(metric_id)s::int IS NULL OR metric_id = %(metric_id)s::int)
          AND (%(step_id)s::int  IS NULL OR step_id  = %(step_id)s::int)
          AND (%(frame)s::int    IS NULL OR job_frame = %(frame)s::int)
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, {
            "job_id": job_id, "sim_id": sim_id, "group_id": group_id,
            "metric_id": metric_id, "step_id": step_id, "frame": frame
        })
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    return rows
def create_jobs_for_sim(
    *,
    sim_id: int,
    jobtype: str,
    jobsubtype: str,
    priority: int,
) -> None:
    """
    INSERT INTO "SimMetricJobs" ... SELECT ... FROM job_seed_view WHERE simid = $1
    (Direct port of Swift SQL.)
    """
    sql = """
        INSERT INTO "SimMetricJobs"
            ("simID", "metricID", "groupID", "stepID", "status", "jobtype",
             "jobsubtype", "priority", "createdate")
        SELECT
            js.simid,
            js.metricid,
            js.groupid,
            js.stepid,
            'queued',
            %(jobtype)s,
            %(jobsubtype)s,
            %(priority)s,
            NOW()
        FROM job_seed_view js
        WHERE js.simid = %(sim_id)s
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, {
            "sim_id": sim_id,
            "jobtype": jobtype,
            "jobsubtype": jobsubtype,
            "priority": priority,
        })
        conn.commit()


# =========================
# 3) JOB STATUS UPDATE (v1)
# =========================
def update_job_status_single(
    *,
    job_id: int,
    to_status: Status,
    error_message: Optional[str] = None,
    set_start: bool = False,
    set_finish: bool = False,
    priority: Optional[int] = None,
) -> None:
    set_start_sql = "startdate = NOW()," if set_start else ""
    set_finish_sql = "finishdate = NOW()," if set_finish else ""
    sql = f"""
        UPDATE "SimMetricJobs"
        SET
            status = %(status)s,
            errormessage = %(err)s,
            {set_start_sql}
            {set_finish_sql}
            priority = %(pri)s
        WHERE jobid = %(job_id)s
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, {
            "status": to_status,
            "err": error_message or "",
            "pri": priority or 0,
            "job_id": job_id,
        })
        conn.commit()


def update_job_status_group(
    *,
    sim_id: int,
    group_name: str,
    frame: int,
    to_status: Status,
    error_message: Optional[str] = None,
    set_start: bool = False,
    set_finish: bool = False,
    priority: Optional[int] = None,
) -> None:
    set_start_sql = "startdate = NOW()," if set_start else ""
    set_finish_sql = "finishdate = NOW()," if set_finish else ""
    sql = f"""
        UPDATE "SimMetricJobs"
        SET
            status = %(status)s,
            errormessage = %(err)s,
            {set_start_sql}
            {set_finish_sql}
            priority = %(pri)s
        WHERE "simID" = %(sim_id)s
          AND group_name = %(group_name)s
          AND job_frame = %(frame)s
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, {
            "status": to_status,
            "err": error_message or "",
            "pri": priority or 0,
            "sim_id": sim_id,
            "group_name": group_name,
            "frame": frame,
        })
        conn.commit()


# ======================
# 4) JOB EXECUTION LOGS
# ======================
def log_execution(
    *,
    job: dict,
    runtime_ms: int,
    queue_wait_ms: int,
    was_aliased: Optional[bool] = None,
    reused_step_id: Optional[int] = None,
    reuse_metric_id: Optional[int] = None,
    learning_note: Optional[str] = None,
) -> None:
    """v1: disabled; reserved for future run-manifest logging."""
    return
def g(*names, default=None):
        for n in names:
            v = job.get(n)
            if v is not None:
                return v
        return default

    payload = {
        "jobid":          int(g("job_id", "jobID")),
        "simid":          int(g("sim_id", "simID")),
        "metricid":       int(g("metric_id", "metricID")),
        "groupid":        int(g("group_id", "groupID", default=0)),
        "stepid":         int(g("step_id", "stepID")),
        "phase":          g("job_phase", "phase", default=0),
        "frame":          g("job_frame", "frame", default=0),
        "precision":      g("sim_precision", "precision"),
        "status":         g("job_status", "status"),
        # mimic Swift: errormessage stored from subtype (if any)
        "errormessage":   g("job_subtype", "jobSubType", default=""),
        "id_f32":         g("metric_id_f32", "metricIDF32"),
        "id_f64":         g("metric_id_f64", "metricIDF64"),
        "jobtype":        g("job_type", "jobType"),
        "jobsubtype":     g("job_subtype", "jobSubType"),
        "priority":       g("job_priority", "jobPriority", default=0),
        "output_path":    g("output_path", "outputPath", default=""),
        "runtime_ms":     int(runtime_ms),
        "queue_wait_ms":  int(queue_wait_ms),
        "was_aliased":    bool(was_aliased) if was_aliased is not None else False,
        "reused_step_id": reused_step_id if reused_step_id is not None else -1,
        "reuse_metric_id":reuse_metric_id if reuse_metric_id is not None else -1,
        "learning_note":  learning_note or "",
    }

    sql = """
        INSERT INTO "JobExecutionLog" (
            jobid, simid, metricid, groupid, stepid, phase, frame,
            precision, status, errormessage,
            id_f32, id_f64, jobtype, jobsubtype, priority, output_path,
            runtime_ms, queue_wait_ms, recorded_at,
            was_aliased, reused_step_id, reuse_metric_id, learning_note
        ) VALUES (
            %(jobid)s, %(simid)s, %(metricid)s, %(groupid)s, %(stepid)s,
            %(phase)s, %(frame)s,
            %(precision)s, %(status)s, %(errormessage)s,
            %(id_f32)s, %(id_f64)s, %(jobtype)s, %(jobsubtype)s, %(priority)s, %(output_path)s,
            %(runtime_ms)s, %(queue_wait_ms)s, NOW(),
            %(was_aliased)s, %(reused_step_id)s, %(reuse_metric_id)s, %(learning_note)s
        )
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, payload)
        conn.commit()
def log_error(*, job: Dict, message: str) -> None:
    """
    Append to ErrorLog using snake_case keys from the ledger view.
    """
    def g(*names, default=None):
        for n in names:
            v = job.get(n)
            if v is not None:
                return v
        return default

    payload = {
        "jobid":        int(g("job_id", "jobID")),
        "simid":        int(g("sim_id", "simID")),
        "metricid":     int(g("metric_id", "metricID", default=0)),
        "stepid":       int(g("step_id", "stepID", default=0)),
        "groupid":      int(g("group_id", "groupID", default=0)),
        "fieldid":      g("field_id", "fieldID", default=None),
        "jobtype":      g("job_type", "jobType", default=""),
        "jobsubtype":   g("job_subtype", "jobSubType", default=""),
        "phase":        g("job_phase", "phase", default=0),
        "frame":        g("job_frame", "frame", default=0),
        "priority":     g("job_priority", "jobPriority", default=0),
        "output_path":  g("output_path", "outputPath", default=""),
        "message":      message,
    }

    sql = """
        INSERT INTO "ErrorLog" (
            jobid, simid, metricid, stepid, groupid, fieldid,
            jobtype, jobsubtype, phase, frame, priority,
            output_path, message, timestamp
        ) VALUES (
            %(jobid)s, %(simid)s, %(metricid)s, %(stepid)s, %(groupid)s, %(fieldid)s,
            %(jobtype)s, %(jobsubtype)s, %(phase)s, %(frame)s, %(priority)s,
            %(output_path)s, %(message)s, NOW()
        )
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, payload)
        conn.commit()
def update_seeded_job(
    *,
    job_id: int,
    group_id: int,
    step_id: int,
    job_type: str,
    job_phase: int,
    job_frame: int,
    output_path: str,
) -> None:
    """
    Direct port of Swift updateSeededJob().
    """
    sql = """
        UPDATE "SimMetricJobs"
        SET
            "groupID"   = %(group_id)s,
            "stepID"    = %(step_id)s,
            jobtype     = %(job_type)s,
            phase       = %(job_phase)s,
            frame       = %(job_frame)s,
            output_path = %(output_path)s
        WHERE jobid = %(job_id)s
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, {
            "group_id": group_id,
            "step_id": step_id,
            "job_type": job_type,
            "job_phase": job_phase,
            "job_frame": job_frame,
            "output_path": output_path,
            "job_id": job_id,
        })
        conn.commit()


# ==================
# 7) FRAME STATISTICS
# ==================
def fetch_frame_stats(*, sim_id: int) -> Dict[str, int]:
    """
    Returns:
      { "maxCompleted": int, "maxSeeded": int }
    Direct port of Swift fetchFrameStats().
    """
    completed_sql = """
        SELECT COALESCE(MAX(frame), -1) AS max_completed
        FROM (
            SELECT frame,
                   SUM(CASE WHEN "stepID" = 3 THEN 1 ELSE 0 END) AS total_final,
                   SUM(CASE WHEN "stepID" = 3 AND status = 'written' THEN 1 ELSE 0 END) AS written_final
            FROM "SimMetricJobs"
            WHERE "simID" = %(sim_id)s
            GROUP BY frame
        ) s
        WHERE s.total_final > 0 AND s.written_final = s.total_final
    """
    seeded_sql = """
        SELECT COALESCE(MAX(frame), -1) AS max_seeded
        FROM "SimMetricJobs"
        WHERE "simID" = %(sim_id)s
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(completed_sql, {"sim_id": sim_id})
        max_completed = cur.fetchone()[0] if cur.rowcount != 0 else -1
        cur.execute(seeded_sql, {"sim_id": sim_id})
        max_seeded = cur.fetchone()[0] if cur.rowcount != 0 else -1
    return {"maxCompleted": int(max_completed), "maxSeeded": int(max_seeded)}


# ===========================================
# 8) FRAME SEEDING FROM TEMPLATE (idempotent)
# ===========================================
class LedgerDBError(Exception):
    pass


def insert_jobs_for_frames_like_frame(
    *,
    sim_id: int,
    start_frame: int,
    end_frame: int,
    template_frame: int = 0,
) -> int:
    """
    Direct port of Swift insertJobsForFramesLikeFrame().
    Returns inserted count.
    """
    if start_frame > end_frame:
        raise LedgerDBError(f"startFrame({start_frame}) > endFrame({end_frame})")

    with _connect() as conn, conn.cursor() as cur:
        # Ensure template exists
        cur.execute(
            """
            SELECT COUNT(*)::int AS cnt
            FROM "SimMetricJobs"
            WHERE "simID" = %(sim_id)s AND frame = %(template_frame)s
            """,
            {"sim_id": sim_id, "template_frame": template_frame},
        )
        cnt = cur.fetchone()[0]
        if cnt == 0:
            raise LedgerDBError(f"No template rows for sim {sim_id} at frame {template_frame}")

        # Transaction + advisory xact lock to serialize seeding per sim
        cur.execute("BEGIN")
        try:
            cur.execute("SELECT pg_advisory_xact_lock(%(sim_id)s)", {"sim_id": sim_id})
            insert_sql = """
                
                WITH template AS (
                    SELECT DISTINCT "metricID", "groupID", "stepID", phase, priority
                    FROM "SimMetricJobs"
                    WHERE "simID" = %(sim_id)s AND frame = %(template_frame)s
                ),
                missing AS (
                    SELECT %(sim_id)s AS "simID",
                           t."metricID", t."groupID", t."stepID",
                           f.frame AS frame,
                           'created'::text AS status,
                           t.phase,
                           t.priority,
                           md5(CONCAT_WS(':',
                               %(sim_id)s::text,
                               t."metricID"::text,
                               t."groupID"::text,
                               t."stepID"::text,
                               f.frame::text,
                               t.phase::text
                           )) AS spec_hash,
                           NOW()::text AS createdate
                    FROM template t
                    JOIN generate_series(%(start_frame)s::int, %(end_frame)s::int) AS f(frame) ON TRUE
                    LEFT JOIN "SimMetricJobs" s
                      ON s."simID"   = %(sim_id)s
                     AND s."metricID"= t."metricID"
                     AND s."groupID" = t."groupID"
                     AND s."stepID"  = t."stepID"
                     AND s.frame     = f.frame
                    WHERE s.jobid IS NULL
                )
                INSERT INTO "SimMetricJobs"
                    ("simID","metricID","groupID","stepID", frame, status, phase, priority, spec_hash, createdate)
                SELECT "simID","metricID","groupID","stepID", frame, status, phase, priority, spec_hash, createdate
                FROM missing
                RETURNING 1

            """
            cur.execute(insert_sql, {
                "sim_id": sim_id,
                "template_frame": template_frame,
                "start_frame": start_frame,
                "end_frame": end_frame,
            })
            # count returned rows:
            inserted = cur.rowcount if cur.rowcount is not None else 0
            cur.execute("COMMIT")
            return int(inserted)
        except Exception:
            cur.execute("ROLLBACK")
            raise