"""Observe the host ceiling; do not mistake a historical job budget for RAM."""
from pathlib import Path


def host_memory_ceiling_bytes():
    limits=[]
    for name in ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        p=Path(name)
        if p.is_file():
            raw=p.read_text().strip()
            if raw!='max':
                n=int(raw)
                if 0<n<2**60:limits.append(n)
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemTotal:'):
            limits.append(int(line.split()[1])*1024)
            break
    if not limits:raise RuntimeError('HOST_MEMORY_CEILING_UNKNOWN')
    return min(limits)


def process_rss_bytes(pid):
    """Current RSS from statm; unavailable telemetry is explicit, never zero."""
    import os
    proc = Path('/proc/self') if int(pid)==os.getpid() else Path('/proc')/str(pid)
    raw=(proc/'statm').read_text().split()
    if len(raw)<2:raise RuntimeError('PROCESS_RSS_UNKNOWN')
    resident=int(raw[1])*os.sysconf('SC_PAGE_SIZE')
    if resident<=0:raise RuntimeError('PROCESS_RSS_UNKNOWN')
    return resident


def process_tree_snapshot(root_pid=None):
    import os
    if root_pid is None or int(root_pid)==os.getpid():
        # /proc may expose host PIDs while getpid() is namespaced.
        root_pid=int(Path('/proc/self/stat').read_text().split(' ',1)[0])
    else:
        root_pid=int(root_pid)
    parents={};errors=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            raw=(p/'stat').read_text();tail=raw[raw.rfind(')')+2:].split()
            parents[int(p.name)]=int(tail[1])
        except (OSError,ValueError,IndexError):continue
    descendants={root_pid}
    changed=True
    while changed:
        added={pid for pid,parent in parents.items() if parent in descendants}-descendants
        changed=bool(added);descendants.update(added)
    processes=[]
    for pid in sorted(descendants):
        try:processes.append({'pid':pid,'rss_bytes':process_rss_bytes(pid)})
        except (OSError,ValueError,RuntimeError) as exc:
            if pid==root_pid:raise RuntimeError('CONTROLLER_RSS_UNKNOWN') from exc
            errors.append({'pid':pid,'error':type(exc).__name__})
    return {'root_pid':root_pid,'processes':processes,'aggregate_rss_bytes':sum(x['rss_bytes'] for x in processes),
            'rss_accounting':'SUM_PROCESS_RSS_INCLUDES_SHARED_PAGES_MULTIPLE_TIMES','disappeared_or_unreadable':errors}


def shared_capacity_snapshot():
    from .execution import detect_effective_cpu_count
    root=Path('/sys/fs/cgroup')
    out={'memory_ceiling_bytes':host_memory_ceiling_bytes(),'cpu_core_equivalents':detect_effective_cpu_count()}
    try:out['shared_memory_current_bytes']=int((root/'memory.current').read_text())
    except OSError:out['shared_memory_current_bytes']=None
    try:out['cpu_quota']= (root/'cpu.max').read_text().strip()
    except OSError:out['cpu_quota']=None
    return out
