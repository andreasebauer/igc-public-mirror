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


def _process_memory(proc):
    import os
    raw=(proc/'statm').read_text().split()
    if len(raw)<2:raise RuntimeError('PROCESS_RSS_UNKNOWN')
    rss=int(raw[1])*os.sysconf('SC_PAGE_SIZE')
    if rss<0:raise RuntimeError('PROCESS_RSS_UNKNOWN')
    if rss==0:return 0,0,'ZERO_RESIDENT_PAGES_OBSERVED'
    try:
        lines=(proc/'smaps_rollup').read_text().splitlines()
        pss=next(int(x.split()[1])*1024 for x in lines if x.startswith('Pss:'))
        if pss<=0:return rss,rss,'RSS_FALLBACK'
        return rss,pss,'PSS'
    except (OSError,StopIteration):
        # Explicit conservative fallback includes shared pages more than once.
        return rss,rss,'RSS_FALLBACK'


def process_tree_snapshot(root_pid=None):
    import os
    if root_pid is None or int(root_pid)==os.getpid():
        root_pid=int(Path('/proc/self/stat').read_text().split(' ',1)[0])
    else:root_pid=int(root_pid)
    parents={};identities={};errors=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            raw=(p/'stat').read_text();tail=raw[raw.rfind(')')+2:].split()
            parents[int(p.name)]=int(tail[1]);identities[int(p.name)]=tail[19]
        except (OSError,ValueError,IndexError):continue
    if root_pid not in parents:raise RuntimeError('CONTROLLER_MEMORY_UNKNOWN')
    descendants={root_pid}
    while True:
        added={pid for pid,parent in parents.items() if parent in descendants}-descendants
        if not added:break
        descendants.update(added)
    processes=[]
    for pid in sorted(descendants):
        proc=Path('/proc')/str(pid)
        try:
            tail=(proc/'stat').read_text().rsplit(')',1)[1].split()
            if tail[0] in {'Z','X'}:
                errors.append({'pid':pid,'error':'PROCESS_EXITED_DURING_SAMPLE'});continue
            rss,accounted,method=_process_memory(proc)
            tail=(proc/'stat').read_text().rsplit(')',1)[1].split()
            if pid==root_pid and rss==0:raise RuntimeError('CONTROLLER_MEMORY_UNKNOWN')
            if tail[19]!=identities[pid]:raise RuntimeError('PROCESS_PID_RECYCLED')
            processes.append({'pid':pid,'rss_bytes':rss,'accounted_memory_bytes':accounted,'accounting':method})
        except (OSError,ValueError,RuntimeError,IndexError) as exc:
            try:exited=(proc/'stat').read_text().rsplit(')',1)[1].split()[0] in {'Z','X'}
            except (OSError,IndexError):exited=not proc.exists()
            if pid==root_pid or not exited:raise RuntimeError('LIVE_PROCESS_MEMORY_UNKNOWN') from exc
            errors.append({'pid':pid,'error':'PROCESS_EXITED_DURING_SAMPLE'})
    return {'root_pid':root_pid,'processes':processes,
            'aggregate_rss_bytes':sum(x['rss_bytes'] for x in processes),
            'accounted_memory_bytes':sum(x['accounted_memory_bytes'] for x in processes),
            'memory_accounting':'PSS_WITH_EXPLICIT_CONSERVATIVE_RSS_FALLBACK' if any(x['accounting']!='PSS' for x in processes) else 'SUM_PROCESS_PSS',
            'rss_accounting':'SUM_PROCESS_RSS_INCLUDES_SHARED_PAGES_MULTIPLE_TIMES',
            'disappeared_or_unreadable':errors}


def shared_capacity_snapshot():
    from .execution import detect_effective_cpu_count
    root=Path('/sys/fs/cgroup')
    out={'memory_ceiling_bytes':host_memory_ceiling_bytes(),'cpu_core_equivalents':detect_effective_cpu_count(ignore_override=True),
         'requested_cpu_budget':__import__('os').environ.get('IG_DECODER_CPU_BUDGET')}
    for name,key,convert in [('memory.current','shared_memory_current_bytes',int),('cpu.max','cpu_quota',str),
                             ('memory.pressure','memory_pressure',str)]:
        try:out[key]=convert((root/name).read_text().strip())
        except OSError:out[key]=None
    for name in ['memory.stat','memory.events']:
        try:out[name.replace('.','_')]={line.split()[0]:int(line.split()[1]) for line in (root/name).read_text().splitlines()}
        except OSError:out[name.replace('.','_')]=None
    stats=out['memory_stat'];current=out['shared_memory_current_bytes']
    if stats is not None and current is not None:
        # inactive file pages and reclaimable slab are observational, not a
        # promise of instant reclaim. No arbitrary cache-pressure stop threshold.
        reclaimable=stats.get('inactive_file',0)+stats.get('slab_reclaimable',0)
        out['estimated_reclaimable_bytes']=reclaimable
        out['shared_nonreclaimable_estimate_bytes']=max(0,current-reclaimable)
    return out
