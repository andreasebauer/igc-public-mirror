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
