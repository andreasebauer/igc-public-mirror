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
        WHERE (%(job_id)s   IS NULL OR job_id   = %(job_id)s)
          AND (%(sim_id)s   IS NULL OR sim_id   = %(sim_id)s)
          AND (%(group_id)s IS NULL OR group_id = %(group_id)s)
          AND (%(metric_id)s IS NULL OR metric_id = %(metric_id)s)
          AND (%(step_id)s  IS NULL OR step_id  = %(step_id)s)
          AND (%(frame)s    IS NULL OR job_frame = %(frame)s)
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
        WHERE simid = %(sim_id)s
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
    job: Dict,
    runtime_ms: int,
    queue_wait_ms: int,
    was_aliased: Optional[bool] = None,
    reused_step_id: Optional[int] = None,
    reuse_metric_id: Optional[int] = None,
    learning_note: Optional[str] = None,
) -> None:
    """
    Direct port of Swift INSERT into "JobExecutionLog".
    `job` must include keys used below (as in MetricJobLedgerRecord).
    """
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
    payload = {
        "jobid": job["jobID"],
        "simid": job["simID"],
        "metricid": job["metricID"],
        "groupid": job["groupID"],
        "stepid": job["stepID"],
        "phase": job["jobPhase"],
        "frame": job["jobFrame"],
        "precision": job.get("simPrecision"),
        "status": job.get("jobStatus"),
        "errormessage": job.get("jobSubType", ""),  # matches Swift line (errormessage from jobSubType)
        "id_f32": job.get("metricIDF32"),
        "id_f64": job.get("metricIDF64"),
        "jobtype": job.get("jobType"),
        "jobsubtype": job.get("jobSubType"),
        "priority": job.get("jobPriority"),
        "output_path": job.get("outputPath"),
        "runtime_ms": runtime_ms,
        "queue_wait_ms": queue_wait_ms,
        "was_aliased": bool(was_aliased) if was_aliased is not None else False,
        "reused_step_id": reused_step_id if reused_step_id is not None else -1,
        "reuse_metric_id": reuse_metric_id if reuse_metric_id is not None else -1,
        "learning_note": learning_note or "",
    }
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, payload)
        conn.commit()


# ===============
# 5) ERROR LOGGING
# ===============
def log_error(*, job: Dict, message: str) -> None:
    """
    Direct port of Swift INSERT into "ErrorLog".
    """
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
    payload = {
        "jobid": job["jobID"],
        "simid": job["simID"],
        "metricid": job["metricID"],
        "stepid": job["stepID"],
        "groupid": job["groupID"],
        "fieldid": job.get("fieldID"),
        "jobtype": job.get("jobType"),
        "jobsubtype": job.get("jobSubType"),
        "phase": job.get("jobPhase"),
        "frame": job.get("jobFrame"),
        "priority": job.get("jobPriority"),
        "output_path": job.get("outputPath"),
        "message": message,
    }
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, payload)
        conn.commit()


# ==============================================
# 6) UPDATE SEEDED JOB (finalization by OE pass-2)
# ==============================================
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
            groupid     = %(group_id)s,
            stepid      = %(step_id)s,
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
                   SUM(CASE WHEN stepid = 3 THEN 1 ELSE 0 END) AS total_final,
                   SUM(CASE WHEN stepid = 3 AND status = 'written' THEN 1 ELSE 0 END) AS written_final
            FROM "SimMetricJobs"
            WHERE simid = %(sim_id)s
            GROUP BY frame
        ) s
        WHERE s.total_final > 0 AND s.written_final = s.total_final
    """
    seeded_sql = """
        SELECT COALESCE(MAX(frame), -1) AS max_seeded
        FROM "SimMetricJobs"
        WHERE simid = %(sim_id)s
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
            WHERE simid = %(sim_id)s AND frame = %(template_frame)s
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
                    SELECT DISTINCT metricid, groupid, stepid, phase, jobpriority
                    FROM "SimMetricJobs"
                    WHERE simid = %(sim_id)s AND frame = %(template_frame)s
                ),
                missing AS (
                    SELECT %(sim_id)s AS simid,
                           t.metricid, t.groupid, t.stepid,
                           f.frame,
                           'created'::text AS status,
                           t.phase,
                           t.jobpriority,
                           md5(CONCAT_WS(':',
                               %(sim_id)s::text,
                               t.metricid::text,
                               t.groupid::text,
                               t.stepid::text,
                               f.frame::text,
                               t.phase::text
                           )) AS spec_hash,
                           NOW() AS created_at
                    FROM template t
                    JOIN generate_series(%(start_frame)s, %(end_frame)s) AS f(frame) ON TRUE
                    LEFT JOIN "SimMetricJobs" s
                      ON s.simid    = %(sim_id)s
                     AND s.metricid = t.metricid
                     AND s.groupid  = t.groupid
                     AND s.stepid   = t.stepid
                     AND s.frame    = f.frame
                    WHERE s.jobid IS NULL
                )
                INSERT INTO "SimMetricJobs"
                    (simid, metricid, groupid, stepid, frame, status, phase, jobpriority, spec_hash, created_at)
                SELECT simid, metricid, groupid, stepid, frame, status, phase, jobpriority, spec_hash, created_at
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