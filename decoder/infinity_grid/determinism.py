from __future__ import annotations
import argparse, hashlib, json, locale, math, multiprocessing as mp, os, platform, sys
from importlib.resources import files
from typing import Any

from . import __version__, build_meta
from .core.canonical import canonical_sha256, canonical_text, CanonicalEncodingError
from .core.boundary import canonical_boundary_record, compose_boundary_records, bridge_ports, record_sha256
from .core.invariants import weakest_link_score, joined_label, retained_port_count, selected_tied_max
from .core.l2 import composite_abc, relation_digests
from .contracts import create_envelope
from .resources import load_json

PROBE_ID='IG_DET001_SEMANTIC_PROBE_V1'

class _TinyTables:
    tri=(0,0,0)
    cid=(0,)
    oc=(1,)
    ot=(0,)*10
    os=(1,)*10
    om=(0,)*10
    on=(0,)*10
    rk=(2,)*10
    lookup={(0,0,0):0}
    rep={0:0}
    def triple(self,tid): return (0,0,0)
    @staticmethod
    def at(a,s,i): return a[s*10+i]

def _task(i:int)->dict[str,Any]:
    # Build a mapping from set iteration on purpose. Canonical JSON must erase hash-order effects.
    pairs={(f'k{(i+j)%7}', (i*17+j*13)%101) for j in range(7)}
    m={k:v for k,v in pairs}
    m['unicode']='ψΔไทย'
    m['nested']={k:v for k,v in {(f'n{j}',j*j+i) for j in range(5)}}
    canon=canonical_sha256(m)
    # Build port order via a set on purpose; boundary canonicalization must erase it.
    raw_ports={(1,0),(4,1),(8,0)}
    a=canonical_boundary_record((1|((i&3)<<1), tuple(raw_ports), 2-(i%2), (3,1), 1))
    b=canonical_boundary_record((2, ((1,4),(4,0),(16,0)), 1, 7, 1))
    lawful=[]
    for sa in range(len(a[1])):
      for sb in range(len(b[1])):
        if bridge_ports(a[1][sa],b[1][sb]):
          c=compose_boundary_records(a,sa,b,sb)
          lawful.append(record_sha256(c))
    return {'i':i,'canonical_sha256':canon,'a_sha256':record_sha256(a),'lawful_children':sorted(lawful)}

def _exception_signature()->dict[str,str]:
    out={}
    try: canonical_sha256({'bad':math.nan})
    except Exception as exc: out['nonfinite']=type(exc).__name__
    try: retained_port_count(0,1)
    except Exception as exc: out['bad_arity']=type(exc).__name__
    return out

def _synthetic_l2()->dict[str,Any]:
    rel,diag=composite_abc(_TinyTables(),0,0,0)
    a,b=relation_digests(rel)
    return {'record_count':len(rel),'diagnostics':diag,'sha256_sorted_repr':a,'sha256_line_serialization':b}

def _fixed_semantic_tail()->dict[str,Any]:
    records=[
      canonical_boundary_record((1, ((1,0),(4,1),(8,0)), 2, (3,1), 1)),
      canonical_boundary_record((2, ((1,4),(4,0),(16,0)), 1, 7, 1)),
      canonical_boundary_record((4, ((2,8),(8,0),(32,2)), 2, (5,2), 0)),
    ]
    sel=selected_tied_max(records)
    payload=load_json('decoder/O_REGIME_EARNED_LAW_REGISTRY_v1.json')
    e1=create_envelope(payload,semantic_profile_id='DET001_PROBE',operational_metadata={'wall_seconds':1.0,'cwd':'A'})
    e2=create_envelope(payload,semantic_profile_id='DET001_PROBE',operational_metadata={'wall_seconds':99.0,'cwd':'B'})
    return {
      'record_sha256':[record_sha256(x) for x in records],
      'selected_tied_max_sha256':[record_sha256(x) for x in sel],
      'weakest_link':weakest_link_score(2,1,2),
      'joined_label':joined_label(1,2,4),
      'retained_ports':retained_port_count(3,3),
      'l2':_synthetic_l2(),
      'exception_signature':_exception_signature(),
      'artifact_content_sha256_equal':e1['content_sha256']==e2['content_sha256'],
      'artifact_content_sha256':e1['content_sha256'],
    }

def semantic_payload(*,workers:int=1,start_method:str='spawn')->dict[str,Any]:
    tasks=list(range(48))
    if workers==1:
        rows=[_task(i) for i in tasks]
    else:
        ctx=mp.get_context(start_method)
        with ctx.Pool(processes=workers) as pool:
            rows=pool.map(_task,tasks,chunksize=max(1,len(tasks)//(workers*3)))
    rows=sorted(rows,key=lambda x:x['i'])
    meta=build_meta()
    return {
      'schema_id':'IG_DET001_SEMANTIC_PAYLOAD_V1',
      'probe_id':PROBE_ID,
      'decoder_version':__version__,
      'source_sha256':meta.get('source_sha256'),
      'canonical_rows_sha256':canonical_sha256(rows),
      'task_count':len(rows),
      'tail':_fixed_semantic_tail(),
    }

def run_probe(*,workers:int=1,start_method:str='spawn')->dict[str,Any]:
    payload=semantic_payload(workers=workers,start_method=start_method)
    text=canonical_text(payload)
    return {
      'schema_id':'IG_DET001_PROBE_RESULT_V1',
      'status':'PASS',
      'semantic_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),
      'semantic_payload_sha256':canonical_sha256(payload),
      'semantic_payload':payload,
      'operational':{
        'workers':workers,
        'start_method':start_method,
        'python':platform.python_version(),
        'python_optimize':sys.flags.optimize,
        'locale':locale.setlocale(locale.LC_ALL,None),
        'cwd':os.getcwd(),
      },
    }

def main(argv=None)->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--start-method',default='spawn')
    ns=ap.parse_args(argv)
    out=run_probe(workers=ns.workers,start_method=ns.start_method)
    print(json.dumps(out,sort_keys=True,separators=(',',':'),ensure_ascii=False))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
