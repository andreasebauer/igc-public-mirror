import os
from typing import Dict, List, Literal, Optional, Tuple

import psycopg

Status = Literal["queued", "running", "written", "failed"]


# -------------------------------
# Connection helper (env-driven)
# -------------------------------
def _connect():
    """Connect using PG* / PGDSN; otherwise fall back to local socket DSN."""
    dsn = os.getenv("PGDSN") or os.getenv("IGC_DATABASE_URL") or os.getenv("DATABASE_URL")
    if dsn:
        return psycopg.connect(dsn)
    # Fallback: libpq defaults (PG*), else local socket to current DB/user
    try:
        return psycopg.connect()  # PG* in env
    except Exception:
        # last-resort: common local DB name
        return psycopg.connect("postgresql:///igc")



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
    Query big_view using lowercase column names (job_id, sim_id, etc.)
    """
    sql = """
        SELECT
            -- map big_view columns to OE-expected keys (no steps as jobs)
            smj_jobid        AS job_id,
            smj_simid        AS sim_id,
            smj_metricid     AS metric_id,
            smj_frame        AS job_frame,
            smj_phase        AS job_phase,
            smj_status       AS job_status,
            smj_output_path  AS output_path,
            COALESCE(smj_output_type, '') AS output_type,
            COALESCE(met_name, smj_output_type, 'metric_' || smj_metricid::text) AS metric_name,            
            -- optional/job metadata useful for paths; no step/group enforcement
            COALESCE(met_group_id, 0)   AS group_id,
            COALESCE(met_outputtypes, '') AS metric_outputtypes
        FROM public.big_view
        WHERE (%(job_id)s::int    IS NULL OR smj_jobid    = %(job_id)s::int)
          AND (%(sim_id)s::int    IS NULL OR smj_simid    = %(sim_id)s::int)
          AND (%(metric_id)s::int IS NULL OR smj_metricid = %(metric_id)s::int)
          AND (%(group_id)s::int  IS NULL OR met_group_id = %(group_id)s::int)
          AND (%(frame)s::int     IS NULL OR smj_frame    = %(frame)s::int)
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, {
            "job_id": job_id, "sim_id": sim_id, "group_id": group_id,
            "metric_id": metric_id, "step_id": step_id, "frame": frame
        })
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    return rows

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
        UPDATE public.simmetjobs
        SET
            status = %(status)s,
            error_message = %(err)s,
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
        UPDATE public.simmetjobs
        SET
            status = %(status)s,
            error_message = %(err)s,
            {set_start_sql}
            {set_finish_sql}
            priority = %(pri)s
        WHERE simid = %(sim_id)s
          AND frame = %(frame)s
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
def log_execution(*, job: dict, runtime_ms: int, queue_wait_ms: int, was_aliased: bool | None = None, reused_step_id: int | None = None, reuse_metric_id: int | None = None, learning_note: str | None = None) -> None:
    """v1: disabled; reserved for future run-manifest logging."""
    return
def log_error(*, job: Dict, message: str) -> None:
    """
    Append to ErrorLog using snake_case keys from the ledger view.

    This is called from OE/core.run() when a job fails. It should NEVER raise;
    logging failures must not crash the runner.
    """
    def g(*names, default=None):
        for n in names:
            v = job.get(n)
            if v is not None:
                return v
        return default

    try:
        jobid = g("job_id", "smj_jobid")
        simid = g("sim_id", "smj_simid")
        metricid = g("metric_id", "smj_metricid")
        stepid = g("step_id", "ms_mim_step")
        groupid = g("group_id", "met_group_id")
        fieldid = None  # not currently exposed via fetch_job_ledger_record
        phase = g("job_phase", "smj_phase")
        frame = g("job_frame", "smj_frame")
        priority = g("smj_priority", "priority", default=0)
        output_path = g("output_path", "smj_output_path", "path", default="")

        # Simple jobtype classification: metric vs sim
        jobtype = "metric" if metricid is not None else "sim"
        # Try to get a useful subtype (metric name or output_type)
        jobsubtype = g("met_name", "metric_name", "output_type", default=None)

        if jobid is None or simid is None:
            # Without jobid/simid, we skip logging; nothing to key on.
            return

        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.errorlog
                    (jobid, simid, metricid, stepid, groupid, fieldid,
                     jobtype, jobsubtype, phase, frame, priority, output_path, message)
                VALUES
                    (%(jobid)s, %(simid)s, %(metricid)s, %(stepid)s, %(groupid)s, %(fieldid)s,
                     %(jobtype)s, %(jobsubtype)s, %(phase)s, %(frame)s, %(priority)s, %(output_path)s, %(message)s)
                """,
                {
                    "jobid": int(jobid),
                    "simid": int(simid),
                    "metricid": int(metricid) if metricid is not None else None,
                    "stepid": int(stepid) if stepid is not None else None,
                    "groupid": int(groupid) if groupid is not None else None,
                    "fieldid": fieldid,
                    "jobtype": jobtype,
                    "jobsubtype": jobsubtype,
                    "phase": int(phase) if phase is not None else None,
                    "frame": int(frame) if frame is not None else None,
                    "priority": int(priority) if priority is not None else 0,
                    "output_path": output_path,
                    "message": message,
                },
            )
            conn.commit()
    except Exception:
        # Logging must never crash the runner; swallow all errors here.
        return

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
        UPDATE public.simmetjobs
        SET
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

def create_compute_template_for_sim(*, sim_id: int) -> int:
    """
    First-seeding for compute: insert a single frame=0, phase=0 job with output_type='state'.
    Idempotent via unique index (simid, frame, phase, output_type, metricid).
    Returns number of rows inserted (0 or 1).
    """
    with _connect() as conn, conn.cursor() as cur:
        # Always start with a clean job table for every fresh run.
        # simmetjobs is a pure work queue; history is stored in jobexecutionlog.
        cur.execute("TRUNCATE TABLE public.simmetjobs")
        conn.commit()        
        cur.execute("""
            INSERT INTO public.simmetjobs
              (simid, metricid, frame, phase, status, priority, createdate,
               output_extension, output_type, mime_type)
            SELECT %s, NULL, 0, 0, 'created', 0, NOW(),
                   '.frame', 'state', 'application/octet-stream'
            WHERE NOT EXISTS (
              SELECT 1 FROM public.simmetjobs s
              WHERE s.simid=%s AND s.frame=0 AND s.phase=0
                AND COALESCE(s.output_type,'')='state'
                AND s.metricid IS NULL
            )
            RETURNING jobid
        """, (sim_id, sim_id))
        row = cur.fetchone()
        if row:
            cur.execute(
                "INSERT INTO public.jobexecutionlog (jobid, simid, status) VALUES (%s, %s, %s)",
                (int(row[0]), sim_id, "created")
            )
            conn.commit()
            return 1
        conn.commit()
        return 0

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
                   COUNT(*) AS total_rows,
                   SUM(CASE WHEN status = 'written' THEN 1 ELSE 0 END) AS written_rows
            FROM public.simmetjobs
            WHERE simid = %(sim_id)s
            GROUP BY frame
        ) s
        WHERE s.total_rows > 0 AND s.written_rows = s.total_rows
    """
    seeded_sql = """
        SELECT COALESCE(MAX(frame), -1) AS max_seeded
        FROM public.simmetjobs
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
            FROM public.simmetjobs
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
                    SELECT DISTINCT
                        metricid,
                        output_type,
                        phase,
                        priority
                    FROM public.simmetjobs
                    WHERE simid = %(sim_id)s
                      AND frame = %(template_frame)s
                ),
                missing AS (
                    SELECT
                        %(sim_id)s AS simid,
                        t.metricid,
                        t.output_type,
                        f.frame        AS frame,
                        'created'::text AS status,
                        t.phase,
                        t.priority,
                        md5(CONCAT_WS(':',
                            %(sim_id)s::text,
                            COALESCE(t.metricid::text,''),
                            COALESCE(t.output_type,''),
                            f.frame::text,
                            t.phase::text
                        ))            AS spec_hash,
                        NOW()::timestamp AS createdate
                    FROM template t
                    JOIN generate_series(%(start_frame)s::int, %(end_frame)s::int) AS f(frame) ON TRUE
                    LEFT JOIN public.simmetjobs s
                      ON s.simid = %(sim_id)s
                     AND s.phase = t.phase
                     AND s.frame = f.frame
                     AND COALESCE(s.output_type,'')    = COALESCE(t.output_type,'')
                     AND COALESCE(s.metricid::text,'') = COALESCE(t.metricid::text,'')
                    WHERE s.jobid IS NULL
                )
                INSERT INTO public.simmetjobs
                    (simid, metricid, output_type, frame, phase, status, priority, spec_hash, createdate)
                SELECT
                    simid, metricid, output_type, frame, phase, status, priority, spec_hash, createdate
                FROM missing
                RETURNING jobid
            """
            cur.execute(insert_sql, {
                "sim_id": sim_id,
                "template_frame": template_frame,
                "start_frame": start_frame,
                "end_frame": end_frame,
            })
            # fetch job IDs and write created logs
            rows = cur.fetchall() or []
            ids = [r[0] for r in rows]
            if ids:
                cur.executemany(
                    "INSERT INTO public.jobexecutionlog (jobid, simid, status) VALUES (%s, %s, %s)",
                    [(jid, sim_id, "created") for jid in ids]
                )
            inserted = len(ids)
            cur.execute("COMMIT")
            return int(inserted)
        except Exception:
            cur.execute("ROLLBACK")
            raise

# --- Metrics attach & seeding helpers ---
def upsert_pathregistry_simroot(sim_id: int, abs_path: str):
    """Store or refresh sim root path under key 'simroot:<sim_id>' in public.pathregistry."""
    key = f"simroot:{sim_id}"
    sql = """
    INSERT INTO public.pathregistry(key, frametemplate)
    VALUES (%s, %s)
    ON CONFLICT (key) DO UPDATE SET frametemplate = EXCLUDED.frametemplate
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (key, abs_path))
        conn.commit()

def sim_exists(sim_id: int) -> bool:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM public.simulations WHERE id=%s", (sim_id,))
        return cur.fetchone() is not None

def load_metric_catalog_grouped():
    """Return [{id,name,metrics:[{id,name,description,outputs,requires,has_steps}]}] grouped by metgroup."""
    q_groups = "SELECT id, name FROM public.metgroup ORDER BY id"
    q_metrics = "SELECT m.id, m.name, m.description, m.outputtypes, m.group_id FROM public.metrics m ORDER BY m.group_id, m.id"
    q_steps  = "SELECT DISTINCT met_id FROM public.metrics_steps"
    q_req = """
        SELECT mf.metricid, array_agg(DISTINCT f2.name ORDER BY f2.name) AS req
        FROM public.metricfieldmatcher mf
        JOIN public.fields f2 ON f2.id = mf.fieldid
        GROUP BY mf.metricid
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(q_groups); groups = cur.fetchall()
        cur.execute(q_metrics); mets = cur.fetchall()
        cur.execute(q_steps); steps = {r[0] for r in cur.fetchall()}
        cur.execute(q_req); reqs  = {r[0]: r[1] for r in cur.fetchall()}
    gm = {gid: {"id": gid, "name": gname, "metrics": []} for gid, gname in groups}
    for mid, name, desc, outs, gid in mets:
        gm.setdefault(gid, {"id": gid, "name": f"Group {gid}", "metrics": []})
        gm[gid]["metrics"].append({
            "id": mid,
            "name": name,
            "description": desc or "",
            "outputs": (outs or "").split("|") if outs else [],
            "requires": reqs.get(mid, []),
            "has_steps": (mid in steps),
        })
    return [gm[k] for k in sorted(gm.keys())]

def load_selected_metric_ids(sim_id: int):
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT metric_id FROM public.simmetricmatcher WHERE sim_id=%s ORDER BY metric_id", (sim_id,))
        return [r[0] for r in cur.fetchall()]

def overwrite_simmetricmatcher(sim_id: int, metric_ids: list[int]):
    """Atomic overwrite: delete existing rows for sim_id, then insert new set."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM public.simmetricmatcher WHERE sim_id=%s", (sim_id,))
            if metric_ids:
                cur.executemany(
                    "INSERT INTO public.simmetricmatcher(sim_id, metric_id) VALUES (%s, %s)",
                    [(sim_id, mid) for mid in metric_ids]
                )
        conn.commit()
