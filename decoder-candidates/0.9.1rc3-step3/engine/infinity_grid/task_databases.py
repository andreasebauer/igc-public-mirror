"""Single engine-owned SQLite inventory for checkpoint/completion sealing."""
TASK_DATABASE_NAMES=frozenset({'partition.sqlite3','state_store.sqlite3','mapping.sqlite3'})


def is_task_database(relative):
    from pathlib import Path
    relative=Path(relative)
    return (relative.parts[:2]==('chain','decoder_stage_runtime')
            and relative.name in TASK_DATABASE_NAMES)
