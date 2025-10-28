from __future__ import annotations
import os
import psycopg
from contextlib import contextmanager

@contextmanager
def cx():
    # uses PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD from environment
    conn = psycopg.connect()
    try:
        yield conn
    finally:
        conn.close()

def fetchall_dict(conn, sql: str, params: tuple|None=None):
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(sql, params or ())
        return cur.fetchall()

def fetchone_dict(conn, sql: str, params: tuple|None=None):
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(sql, params or ())
        return cur.fetchone()

def execute(conn, sql: str, params: tuple|None=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
    conn.commit()
