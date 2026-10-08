"""Exact incremental DAG partitions; hashes never replace full node equality."""
import json,hashlib,gzip
from pathlib import Path
from infinity_grid.canon import canonical_sha256

def read(path,digest):
 raw=Path(path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('PARTITION_BYTES')
 return json.loads(gzip.decompress(raw) if raw[:2]==b"\x1f\x8b" else raw)

def assemble(nodes,roots,science):
 reached={};active=set()
 def visit(k):
  if k in reached:return
  if k in active:raise ValueError('DAG_CYCLE')
  active.add(k);v=nodes[k]
  if v['construction_digest']!=k:raise ValueError('NODE_KEY')
  for c in v.get('children',[]):visit(c)
  reached[k]=v;active.remove(k)
 for k in roots:visit(k)
 dag={'schema_id':'IG_MATURATION_STATE_DAG_V1','nodes':reached,'roots':roots}
 if canonical_sha256(dag)!=science:raise ValueError('DAG_SCIENCE')
 dag['science_sha256']=science;return dag

def load_parent(path,digest):
 manifest=read(path,digest)
 if 'dag' in manifest:return manifest
 if manifest['schema_id']!='IG_G1_PARTITION_MANIFEST_V1':raise ValueError('MANIFEST_SCHEMA')
 base=read(manifest['base']['path'],manifest['base']['sha256']);nodes=dict(base['dag']['nodes'])
 for ref in manifest['partitions']:
  part=read(ref['path'],ref['sha256'])
  for k,v in part['nodes'].items():
   if k in nodes and nodes[k]!=v:raise ValueError('NODE_COLLISION')
   nodes[k]=v
 return {'level':manifest['level'],'dag':assemble(nodes,manifest['roots'],manifest['science_sha256']),'selected_count':manifest['selected_count'],'candidate_count':manifest['candidate_count']}

def delta(parent,current):
 old=parent['dag']['nodes'];new=current['dag']['nodes'];added={}
 for k,v in new.items():
  if k in old:
   if old[k]!=v:raise ValueError('NODE_COLLISION')
  else:added[k]=v
 out={k:v for k,v in current.items() if k!='dag'}
 out.update(nodes=added,roots=current['dag']['roots'],science_sha256=current['dag']['science_sha256'],parent_science_sha256=parent['dag']['science_sha256'])
 assert assemble(dict(old,**added),out['roots'],out['science_sha256'])==current['dag']
 return out
